from abc import ABC, abstractmethod
import os
import h5py
import scipy.io
import math
import numpy as np

try:
    import torch
except ModuleNotFoundError:
    print('PyTorch module is not found in the current environment')
    pass

try:
    from jax import grad
except ModuleNotFoundError:
    print('JAX module is not found in the current environment')
    pass

try:
    import tensorflow as tf
except ModuleNotFoundError:
    print('Tensorflow module is not found in the current environment')
    pass

from utils import get_grid, reaction, diffusion


class PdeBase(ABC):
    def __init__(self, framework='pytorch', postfix=''):
        self.framework = framework
        self.postfix = postfix

    def get_u_t(self, u, x, t, **kwargs):
        if self.framework == 'jax':
            neural_net = kwargs['neural_net']
            params = kwargs['params']
            u_t = grad(neural_net, argnums=1)(params, t, x)
        elif self.framework == 'tf':
            u_t = tf.gradients(u, t)[0]
        elif self.framework == 'tf2':
            u_t = tf.gradients(u, t)
        else:
            u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        return u_t

    def get_u_x(self, u, x, t, **kwargs):
        if self.framework == 'jax':
            neural_net = kwargs['neural_net']
            params = kwargs['params']
            u_x = grad(neural_net, argnums=2)(params, t, x)
        elif self.framework == 'tf':
            u_x = tf.gradients(u, x)[0]
        elif self.framework == 'tf2':
            u_x = tf.gradients(u, x)
        else:
            u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        return u_x

    def get_u_xx(self, u, x, t, u_x=None, **kwargs):
        if u_x is None and self.framework != 'jax':
            u_x = self.get_u_x(u, x, t)

        if self.framework == 'jax':
            neural_net = kwargs['neural_net']
            params = kwargs['params']
            u_xx = grad(grad(neural_net, argnums=2), argnums=2)(params, t, x)
        elif self.framework == 'tf':
            u_xx = tf.gradients(u_x, x)[0]
        elif self.framework == 'tf2':
            u_xx = tf.gradients(u_x, x)
        else:
            u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[
                0]
        return u_xx

    def get_u_tt(self, u, x, t, u_t=None, **kwargs):
        if u_t is None and self.framework != 'jax':
            u_t = self.get_u_t(u, x, t)

        if self.framework == 'jax':
            neural_net = kwargs['neural_net']
            params = kwargs['params']
            u_tt = grad(grad(neural_net, argnums=1), argnums=1)(params, t, x)
        elif self.framework == 'tf':
            u_tt = tf.gradients(u_t, t)[0]
        elif self.framework == 'tf2':
            u_tt = tf.gradients(u_t, t)
        else:
            u_tt = torch.autograd.grad(u_t, t, grad_outputs=torch.ones_like(u_t), retain_graph=True, create_graph=True)[
                0]
        return u_tt

    def save_solution(self, num_x=256, num_t=100, x0=0, t0=0, xN=2 * np.pi, tN=1, path='data'):
        x, t, u = self.solution(num_x, num_t, x0, t0, xN, tN)

        filename = self.get_filename()
        data_path = os.path.join(path, filename)
        with h5py.File(data_path, 'w') as f:
            f.create_dataset('x', data=x.reshape(-1, 1))
            f.create_dataset('t', data=t)
            f.create_dataset('u', data=u)
            f.create_dataset('grid_bounds', data=np.array([x0, xN, t0, tN]))
        print(f'Saved data to the file {filename}')
        return

    def get_filename(self):
        return 'data.hdf5'

    def get_result_name(self):
        return self.name + self.postfix

    def wrap_const(self, const_list):
        if self.framework == 'tf2':
            for i in range(len(const_list)):
                const_list[i] = tf.constant(const_list[i], dtype=tf.float32)
        return const_list

    @property
    @abstractmethod
    def name(self):
        return NotImplemented

    @property
    @abstractmethod
    def bound_der_loss(self):
        return NotImplemented

    @property
    @abstractmethod
    def zero_bc(self):
        return NotImplemented

    @abstractmethod
    def u0(self, x):
        return NotImplemented

    @abstractmethod
    def pde(self, u, x, t, **kwargs):
        return NotImplemented

    @abstractmethod
    def ic(self):
        return NotImplemented

    @abstractmethod
    def bc(self):
        return NotImplemented

    @abstractmethod
    def solution(self, num_x=200, num_t=200, x0=0, t0=0, xN=1, tN=1):
        return NotImplemented


class AllenCahn(PdeBase):
    def __init__(self, framework='pytorch', postfix='', c1=0.0001, c2=5.):
        super().__init__(framework, postfix)
        c1, c2 = self.wrap_const([c1, c2])
        self.c1 = c1
        self.c2 = c2

    def u0(self, x):
        u0 = (x ** 2) * np.cos(np.pi * x)
        u0 = u0.reshape(x.shape)
        return u0

    def pde(self, u, x, t, **kwargs):
        u_t = self.get_u_t(u, x, t, **kwargs)
        u_xx = self.get_u_xx(u, x, t, **kwargs)
        f = u_t - self.c1 * u_xx + self.c2 * u * (u - 1) * (u + 1)  # 5*u**3 - 5*u
        return f

    def ic(self):
        return

    def bc(self):
        return

    def solution(self, num_x=200, num_t=200, x0=0, t0=0, xN=1, tN=1):
        # upload from the file and save in the right format afterwards
        data = scipy.io.loadmat('data/AC.mat')
        u = data['uu'].T.flatten()
        x = data['xx']
        t = data['tt']
        return x, t, u

    def get_filename(self):
        name = f'{self.name}.hdf5'
        return name

    @property
    def name(self):
        return 'ac'

    @property
    def bound_der_loss(self):
        return True

    @property
    def zero_bc(self):
        return False


class Convection(PdeBase):
    def __init__(self, beta, scale=2*np.pi, framework='pytorch', postfix=''):
        super().__init__(framework, postfix)
        scale, beta = self.wrap_const([scale, beta])
        self.scale = scale
        self.beta = beta
        self.source = 0

    def set_scale(self, scale):
        self.scale = self.wrap_const([scale])[0]

    def u0(self, x):
        u_val = np.sin(self.scale * x)
        return u_val

    def pde(self, u, x, t, **kwargs):
        u_t = self.get_u_t(u, x, t, **kwargs)
        u_x = self.get_u_x(u, x, t, **kwargs)
        f = u_t + self.beta / self.scale * u_x
        return f

    def ic(self):
        return

    def bc(self):
        return

    # the function is adopted from https://github.com/a1k12/characterizing-pinns-failure-modes
    def solution(self, num_x=200, num_t=200, x0=0, t0=0, xN=1, tN=1):
        h = xN / num_x
        x = np.arange(x0, xN, h)  # not inclusive of the last point (why?)
        t = np.linspace(t0, tN, num_t).reshape(-1, 1)
        X, T = np.meshgrid(x, t)

        u0_val = self.u0(x)
        G = np.full_like(u0_val, fill_value=self.source)

        IKX_pos = 1j * np.arange(0, num_x / 2 + 1, 1)
        IKX_neg = 1j * np.arange(-num_x / 2 + 1, 0, 1)
        IKX = np.concatenate((IKX_pos, IKX_neg))

        uhat0 = np.fft.fft(u0_val)
        nu_factor = np.exp(-self.beta * IKX * T)
        A = uhat0 - np.fft.fft(G) * 0  # at t=0, second term goes away
        uhat = A * nu_factor + np.fft.fft(G) * T  # for constant, fft(p) dt = fft(p)*t
        u = np.real(np.fft.ifft(uhat))

        u = u.flatten()
        return x, t, u

    def get_filename(self):
        name = f'{self.name}_beta_{int(self.beta)}.hdf5'
        return name

    @property
    def name(self):
        return 'convection'

    @property
    def bound_der_loss(self):
        return False

    @property
    def zero_bc(self):
        return False


class Diffusion(PdeBase):
    def __init__(self, d, framework='pytorch', postfix=''):
        super().__init__(framework, postfix)
        d = self.wrap_const([d])[0]
        self.d = d
        self.scale = 1
        self.source = 0

    def u0(self, x):
        b = self.d * self.scale
        u_val = np.sin(b * x)
        return u_val

    def pde(self, u, x, t, **kwargs):
        u_t = self.get_u_t(u, x, t, **kwargs)
        u_xx = self.get_u_xx(u, x, t, **kwargs)
        d_prime = 1 / (self.d * self.scale) ** 2
        # f = u_t - self.d * u_xx
        f = u_t - d_prime * u_xx
        return f

    def ic(self):
        return

    def bc(self):
        return

    def solution(self, num_x=256, num_t=100, x0=0, t0=0, xN=2 * np.pi, tN=1):
        nu = 1 / (self.d * self.scale) ** 2

        h = xN / num_x
        x = np.arange(x0, xN, h)  # not inclusive of the last point
        t = np.linspace(t0, tN, num_t).reshape(-1, 1)
        X, T = np.meshgrid(x, t)

        u0_val = self.u0(self.scale * x)
        G = np.full_like(u0_val, fill_value=self.source)

        IKX_pos = 1j * np.arange(0, num_x / 2 + 1, 1)
        IKX_neg = 1j * np.arange(-num_x / 2 + 1, 0, 1)
        IKX = np.concatenate((IKX_pos, IKX_neg))
        IKX2 = IKX * IKX

        uhat0 = np.fft.fft(u0_val)
        nu_factor = np.exp(nu * IKX2 * T)
        A = uhat0 - np.fft.fft(G) * 0  # at t=0, second term goes away
        uhat = A * nu_factor + np.fft.fft(G) * T  # for constant, fft(p) dt = fft(p)*t
        u = np.real(np.fft.ifft(uhat))
        u = u.flatten()

        return x, t, u

    def get_filename(self):
        name = f'{self.name}_d_{int(self.d)}.hdf5'
        return name

    @property
    def name(self):
        return 'diffusion'

    @property
    def bound_der_loss(self):
        return True

    @property
    def zero_bc(self):
        return False


class Heat(PdeBase):
    def __init__(self, d, framework='pytorch', postfix=''):
        super().__init__(framework, postfix)
        d, scale = self.wrap_const([d, 1.])
        self.d = d
        self.scale = scale

    def u0(self, x):
        b = self.d * self.scale
        u_val = np.sin(b * x)
        return u_val

    def pde(self, u, x, t, **kwargs):
        u_t = self.get_u_t(u, x, t, **kwargs)
        u_xx = self.get_u_xx(u, x, t, **kwargs)
        # f = u_t - self.nu * u_xx + self.beta * u_x
        d_prime = 1 / (self.d * self.scale) ** 2
        f = u_t - d_prime * u_xx
        return f

    def ic(self):
        return

    def bc(self):
        return

    def solution(self, num_x=256, num_t=100, x0=0, t0=0, xN=2 * np.pi, tN=1):
        x, t, X, T = get_grid(x0, xN, num_x, t0, tN, num_t)

        # solution from page 20 of this lecture notes
        # https://sites.ualberta.ca/~niksirat/ODE/chapter-7ode.pdf
        # and https://tutorial.math.lamar.edu/classes/de/solvingheatequation.aspx

        u = np.exp(-T) * self.u0(X)
        return x, t, u

    def get_filename(self):
        name = f'{self.name}_d_{int(self.d)}.hdf5'
        return name

    @property
    def name(self):
        return 'heat'

    @property
    def bound_der_loss(self):
        return False

    @property
    def zero_bc(self):
        return True


class Reaction(PdeBase):
    def __init__(self, rho, framework='pytorch', postfix=''):
        super().__init__(framework, postfix)
        rho = self.wrap_const([rho])[0]
        self.rho = rho

    def u0(self, x):
        x0 = np.pi
        sigma = np.pi / 4
        u0_val = np.exp(-np.power((x - x0) / sigma, 2.) / 2.)
        return u0_val

    def pde(self, u, x, t, **kwargs):
        u_t = self.get_u_t(u, x, t, **kwargs)
        # f = u_t + self.rho * u * (u - 1)  # - self.rho * u + self.rho * u ** 2
        f = u_t - self.rho * u + self.rho * u ** 2
        return f

    def ic(self):
        return

    def bc(self):
        return

    # the function is adopted from https://github.com/a1k12/characterizing-pinns-failure-modes
    def solution(self, num_x=256, num_t=100, x0=0, t0=0, xN=2 * np.pi, tN=1):
        """ Computes the discrete solution of the reaction-diffusion PDE using
            pseudo-spectral operator splitting.
        """
        dx = xN / num_x
        x = np.arange(x0, xN, dx)
        t = np.linspace(t0, tN, num_t).reshape(-1, 1)
        X, T = np.meshgrid(x, t)

        u0_val = self.u0(x)
        u = reaction(u0_val, self.rho, T)
        u = u.flatten()
        return x, t, u

    def get_filename(self):
        name = f'{self.name}_rho_{int(self.rho)}.hdf5'
        return name

    @property
    def name(self):
        return 'reaction'

    @property
    def bound_der_loss(self):
        return False

    @property
    def zero_bc(self):
        return False


class Rd(PdeBase):
    def __init__(self, nu, rho, framework='pytorch', postfix=''):
        super().__init__(framework, postfix)
        nu, rho = self.wrap_const([nu, rho])
        self.nu = nu
        self.rho = rho

    def u0(self, x):
        x0 = np.pi
        sigma = np.pi / 4
        u0_val = np.exp(-np.power((x - x0) / sigma, 2.) / 2.)
        return u0_val

    def pde(self, u, x, t, **kwargs):
        u_t = self.get_u_t(u, x, t, **kwargs)
        u_xx = self.get_u_xx(u, x, t, **kwargs)
        f = u_t - self.nu * u_xx + self.rho * u * (u - 1)  # - self.rho * u + self.rho * u ** 2
        return f

    def ic(self):
        return

    def bc(self):
        return

    # the function is adopted from https://github.com/a1k12/characterizing-pinns-failure-modes
    def solution(self, num_x=256, num_t=100, x0=0, t0=0, xN=2 * np.pi, tN=1):
        """ Computes the discrete solution of the reaction-diffusion PDE using
            pseudo-spectral operator splitting.
        """
        dx = xN / num_x
        dt = tN / num_t
        x = np.arange(x0, xN, dx)  # not inclusive of the last point
        t = np.linspace(t0, tN, num_t).reshape(-1, 1)
        u = np.zeros((num_x, num_t))

        IKX_pos = 1j * np.arange(0, num_x / 2 + 1, 1)
        IKX_neg = 1j * np.arange(-num_x / 2 + 1, 0, 1)
        IKX = np.concatenate((IKX_pos, IKX_neg))
        IKX2 = IKX * IKX

        u0_val = self.u0(x)
        u[:, 0] = u0_val
        u_ = u0_val
        for i in range(num_t - 1):
            u_ = reaction(u_, self.rho, dt)
            u_ = diffusion(u_, self.nu, dt, IKX2)
            u[:, i + 1] = u_

        u = u.T
        u = u.flatten()
        return x, t, u

    def get_filename(self):
        name = f'{self.name}_nu_{int(self.nu)}_rho_{int(self.rho)}.hdf5'
        return name

    @property
    def name(self):
        return 'rd'

    @property
    def bound_der_loss(self):
        return True

    @property
    def zero_bc(self):
        return False


class Wave(PdeBase):
    def __init__(self, c, framework='pytorch', postfix=''):
        super().__init__(framework, postfix)
        c = self.wrap_const([c])[0]
        self.c = c  # the constant in the equation is c**2
        self.u0_type = 'wave0'

    def u0(self, x):
        # function corresponding to initial conditions for displacemment
        if self.u0_type == 'wave0':
            u0_val = np.sin(np.pi * x) + 0.5 * np.sin(4 * np.pi * x)
        else:
            u0_val = np.sin(np.pi * x) + np.sin(2 * np.pi * x)
        return u0_val

    def pde(self, u, x, t, **kwargs):
        u_tt = self.get_u_tt(u, x, t, **kwargs)
        u_xx = self.get_u_xx(u, x, t, **kwargs)
        f = u_tt - self.c ** 2 * u_xx
        return f

    def ic(self):
        return

    def bc(self):
        return

    def solution(self, num_x=200, num_t=200, x0=0, t0=0, xN=1, tN=1):
        """Calculate the u solution for 1D wave equation.
        """
        x, t, X, T = get_grid(x0, xN, num_x, t0, tN, num_t)

        # D’Alembert’s solution to the wave equation
        # https://personal.math.ubc.ca/~ward/teaching/m316/lecture21.pdf
        # Initial velocity v0 is equal to zero
        u = 0.5 * (self.u0(X - self.c * T) + self.u0(X + self.c * T))
        return x, t, u

    def get_filename(self):
        name = f'{self.name}_c_{int(self.c)}.hdf5'
        return name

    @property
    def name(self):
        return 'wave'

    @property
    def bound_der_loss(self):
        return False

    @property
    def zero_bc(self):
        return False


class Helmholtz(PdeBase):
    def __init__(self, n, framework='pytorch', postfix=''):
        super().__init__(framework, postfix)
        n = self.wrap_const([n])[0]
        self.n = n

    def u0(self, x):
        u_val = 0.0
        return u_val

    def pde(self, u, x, t, **kwargs):
        k0 = 2 * math.pi * self.n
        u_tt = self.get_u_tt(u, x, t, **kwargs)
        u_xx = self.get_u_xx(u, x, t, **kwargs)
        f1 = k0 ** 2 * torch.sin(k0 * x) * torch.sin(k0 * t)
        f = u_tt + u_xx + k0 ** 2 * u + f1
        return f

    def ic(self):
        return

    def bc(self):
        return

    def solution(self, num_x=200, num_t=200, x0=0, t0=0, xN=1, tN=1):
        # the same equation as here
        # https://deepxde.readthedocs.io/en/latest/demos/helmholtz.2d.dirichlet.html#helmholtz-equation-over-a-2d-square-domain
        # it has analytical real solution
        x, t, X, T = get_grid(x0, xN, num_x, t0, tN, num_t)

        k0 = 2 * np.pi * self.n
        u = np.sin(k0 * X) * np.sin(k0 * T)
        return x, t, u

    def get_filename(self):
        name = f'{self.name}_n_{int(self.n)}.hdf5'
        return name

    @property
    def name(self):
        return 'helmholtz'

    @property
    def bound_der_loss(self):
        return False

    @property
    def zero_bc(self):
        return False
