from pde.pde import Rd, Reaction, Convection, Heat, Diffusion
from plot_utils import plot_data
from data_utils import load_solution


def get_img_name(filename, is_pdf=True):
    ext = 'pdf' if is_pdf else 'png'
    img_name = filename.replace('hdf5', ext)
    return img_name


def plot_gen(system, path):
    filename = system.get_filename()
    x, t, u, grid_bounds = load_solution(path, filename)
    img_name = get_img_name(filename, is_pdf=True)
    plot_data(x, t, u, path, img_name)
    return


def plot_generated_rd(path='data'):
    rho = 5
    nus = [2, 3, 4]
    for nu in nus:
        pde = Rd(nu=nu, rho=rho)
        plot_gen(pde, path)
    return


def plot_generated_reaction(path='data'):
    rhos = [5, 6, 7]
    for rho in rhos:
        pde = Reaction(rho=rho)
        plot_gen(pde, path)
    return


def plot_generated_convection(path='data'):
    betas = [30, 40]
    for beta in betas:
        pde = Convection(beta=beta)
        plot_gen(pde, path)
    return


def plot_generated_heat(path='data'):
    ds = [5, 7, 10]
    for d in ds:
        pde = Heat(d=d)
        plot_gen(pde, path)
    return


def plot_generated_diffusion(path='data'):
    ds = [5, 7, 10]
    for d in ds:
        pde = Diffusion(d=d)
        plot_gen(pde, path)
    return


def plot_systems(path='data'):
    print('Plot RD')
    plot_generated_rd(path)

    print("Plot reaction")
    plot_generated_reaction(path)

    print("Plot convection")
    plot_generated_convection(path)

    print("Plot heat")
    plot_generated_heat(path)

    print("Plot diffusion")
    plot_generated_diffusion(path)


if __name__ == "__main__":
    save_path = 'data'
    plot_systems(save_path)
