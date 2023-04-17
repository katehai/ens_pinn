from abc import ABC, abstractmethod
import os
import h5py
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

from pde.pde_utils import get_grid, reaction, diffusion


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

    def save_solution(self, num_x=256, num_t=100, x0=0, t0=0, xn=2 * np.pi, tn=1, path='data'):
        x, t, u = self.solution(num_x, num_t, x0, t0, xn, tn)

        if not os.path.exists(path):
            os.makedirs(path)
        filename = self.get_filename()
        data_path = os.path.join(path, filename)
        with h5py.File(data_path, 'w') as f:
            f.create_dataset('x', data=x.reshape(-1, 1))
            f.create_dataset('t', data=t)
            f.create_dataset('u', data=u)
            f.create_dataset('grid_bounds', data=np.array([x0, xn, t0, tn]))
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
    def solution(self, num_x=200, num_t=200, x0=0, t0=0, xn=1, tn=1):
        return NotImplemented


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
    def solution(self, num_x=200, num_t=200, x0=0, t0=0, xn=1, tn=1):
        h = xn / num_x
        x = np.arange(x0, xn, h)  # not inclusive of the last point (why?)
        t = np.linspace(t0, tn, num_t).reshape(-1, 1)
        xx, tt = np.meshgrid(x, t)

        u0_val = self.u0(x)
        g = np.full_like(u0_val, fill_value=self.source)

        ikx_pos = 1j * np.arange(0, num_x / 2 + 1, 1)
        ikx_neg = 1j * np.arange(-num_x / 2 + 1, 0, 1)
        ikx = np.concatenate((ikx_pos, ikx_neg))

        u_hat0 = np.fft.fft(u0_val)
        nu_factor = np.exp(-self.beta * ikx * tt)
        a = u_hat0 - np.fft.fft(g) * 0  # at t=0, second term goes away
        u_hat = a * nu_factor + np.fft.fft(g) * tt  # for constant, fft(p) dt = fft(p)*t
        u = np.real(np.fft.ifft(u_hat))

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

    def solution(self, num_x=256, num_t=100, x0=0, t0=0, xn=2 * np.pi, tn=1):
        nu = 1 / (self.d * self.scale) ** 2

        h = xn / num_x
        x = np.arange(x0, xn, h)  # not inclusive of the last point
        t = np.linspace(t0, tn, num_t).reshape(-1, 1)
        xx, tt = np.meshgrid(x, t)

        u0_val = self.u0(self.scale * x)
        g = np.full_like(u0_val, fill_value=self.source)

        ikx_pos = 1j * np.arange(0, num_x / 2 + 1, 1)
        ikx_neg = 1j * np.arange(-num_x / 2 + 1, 0, 1)
        ikx = np.concatenate((ikx_pos, ikx_neg))
        ikx2 = ikx * ikx

        u_hat0 = np.fft.fft(u0_val)
        nu_factor = np.exp(nu * ikx2 * tt)
        a = u_hat0 - np.fft.fft(g) * 0  # at t=0, second term goes away
        u_hat = a * nu_factor + np.fft.fft(g) * tt  # for constant, fft(p) dt = fft(p)*t
        u = np.real(np.fft.ifft(u_hat))
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

    def solution(self, num_x=256, num_t=100, x0=0, t0=0, xn=2 * np.pi, tn=1):
        x, t, xx, tt = get_grid(x0, xn, num_x, t0, tn, num_t)

        # solution from page 20 of this lecture notes
        # https://sites.ualberta.ca/~niksirat/ODE/chapter-7ode.pdf
        # and https://tutorial.math.lamar.edu/classes/de/solvingheatequation.aspx

        u = np.exp(-tt) * self.u0(xx)
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
        f = u_t - self.rho * u + self.rho * u ** 2
        return f

    def ic(self):
        return

    def bc(self):
        return

    # the function is adopted from https://github.com/a1k12/characterizing-pinns-failure-modes
    def solution(self, num_x=256, num_t=100, x0=0, t0=0, xn=2 * np.pi, tn=1):
        """ Computes the discrete solution of the reaction-diffusion PDE using
            pseudo-spectral operator splitting.
        """
        dx = xn / num_x
        x = np.arange(x0, xn, dx)
        t = np.linspace(t0, tn, num_t).reshape(-1, 1)
        xx, tt = np.meshgrid(x, t)

        u0_val = self.u0(x)
        u = reaction(u0_val, self.rho, tt)
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
    def solution(self, num_x=256, num_t=100, x0=0, t0=0, xn=2 * np.pi, tn=1):
        """ Computes the discrete solution of the reaction-diffusion PDE using
            pseudo-spectral operator splitting.
        """
        dx = xn / num_x
        dt = tn / num_t
        x = np.arange(x0, xn, dx)  # not inclusive of the last point
        t = np.linspace(t0, tn, num_t).reshape(-1, 1)
        u = np.zeros((num_x, num_t))

        ikx_pos = 1j * np.arange(0, num_x / 2 + 1, 1)
        ikx_neg = 1j * np.arange(-num_x / 2 + 1, 0, 1)
        ikx = np.concatenate((ikx_pos, ikx_neg))
        ikx2 = ikx * ikx

        u0_val = self.u0(x)
        u[:, 0] = u0_val
        u_ = u0_val
        for i in range(num_t - 1):
            u_ = reaction(u_, self.rho, dt)
            u_ = diffusion(u_, self.nu, dt, ikx2)
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
