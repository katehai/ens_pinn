# the functions are adopted from https://github.com/a1k12/characterizing-pinns-failure-modes
import numpy as np


def reaction(u, rho, dt):
    """ du/dt = rho*u*(1-u)
    """
    factor_1 = u * np.exp(rho * dt)
    factor_2 = (1 - u)
    u = factor_1 / (factor_2 + factor_1)
    return u


def diffusion(u, nu, dt, ikx2):
    """ du/dt = nu*d2u/dx2
    """
    factor = np.exp(nu * ikx2 * dt)
    u_hat = np.fft.fft(u)
    u_hat *= factor
    u = np.real(np.fft.ifft(u_hat))
    return u


def get_grid(x0, xn, num_x, t0, tn, num_t):
    h = xn / num_x
    x = np.arange(x0, xn, h)  # not inclusive of the last point
    t = np.linspace(t0, tn, num_t).reshape(-1, 1)
    xx, tt = np.meshgrid(x, t)
    return x, t, xx, tt
