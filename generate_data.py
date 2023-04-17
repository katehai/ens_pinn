import numpy as np
from pde.pde import Rd, Reaction, Convection, Heat, Diffusion
from plot_generated_data import plot_systems


def generate_rd(path='data'):
    x0, xn, num_x = 0.0, 2*np.pi, 256
    t0, tn, num_t = 0.0, 1., 100

    rho = 5
    nus = [2, 3, 4]
    for nu in nus:
        pde = Rd(nu=nu, rho=rho)
        pde.save_solution(num_x, num_t, x0, t0, xn, tn, path=path)


def generate_reaction(path='data'):
    x0, xn, num_x = 0.0, 2*np.pi, 256
    t0, tn, num_t = 0.0, 1., 100

    rhos = [5, 6, 7]
    for rho in rhos:
        pde = Reaction(rho=rho)
        pde.save_solution(num_x, num_t, x0, t0, xn, tn, path=path)


def generate_convection(path='data'):
    x0, xn, num_x = 0.0, 1, 256  # should use scale in the Convection eq. if this is used
    # x0, xn, num_x = 0.0, 2*np.pi, 256
    t0, tn, num_t = 0.0, 1., 100

    betas = [30, 40]
    for beta in betas:
        pde = Convection(beta=beta)
        pde.save_solution(num_x, num_t, x0, t0, xn, tn, path=path)


def generate_heat(path='data'):
    # x0, xn, num_x = 0.0, 2*np.pi, 256
    x0, xn, num_x = 0.0, 1, 256
    t0, tn, num_t = 0.0, 1., 100

    ds = [5, 7, 10]
    for d in ds:
        pde = Heat(d=d)
        pde.save_solution(num_x, num_t, x0, t0, xn, tn, path=path)


def generate_diffusion(path='data'):
    x0, xn, num_x = 0.0, 2*np.pi, 256
    t0, tn, num_t = 0.0, 1., 100

    ds = [5, 7, 10]
    for d in ds:
        pde = Diffusion(d=d)
        pde.save_solution(num_x, num_t, x0, t0, xn, tn, path=path)


def generate_systems(path='data'):
    print('Generate RD')
    generate_rd(path)

    print("Generate reaction")
    generate_reaction(path)

    print('Generate convection')
    generate_convection(path)

    print("Generate heat")
    generate_heat(path)

    print("Generate diffusion")
    generate_diffusion(path)


if __name__ == "__main__":
    save_path = 'data1'
    plot_generated = False
    generate_systems(save_path)

    if plot_generated:
        plot_systems(save_path)
