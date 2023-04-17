import os
import h5py
import numpy as np


def get_prob_inner(n, x_grid):
    p = np.ones(n)
    if x_grid is not None:
        p[0:x_grid] = 0.  # initial conditions
        p[::x_grid] = 0.  # lower bound (e.g. we can have inner points on the upper bound
    return p


def get_prob_bound(n, x_grid, use_bc=True):
    p = np.zeros(n)
    if x_grid is not None:
        p[-x_grid:] = 1.  # points on t=1 for all X
        if use_bc:
            p[::x_grid] = 1.  # lower bound
    return p


def sample_random(xt_all, num_points, return_idx=False, x_grid=None, inner=True, use_bc=True, seed=0):
    np.random.seed(seed)
    p = get_prob_inner(len(xt_all), x_grid) if inner else get_prob_bound(len(xt_all), x_grid, use_bc)
    p /= np.sum(p)

    idx = np.random.choice(xt_all.shape[0], num_points, replace=False, p=p)
    xt_sampled = xt_all[idx, :]

    if return_idx:
        return idx, xt_sampled
    return xt_sampled


def get_xt_f(xt_star, args):
    xt_f = sample_random(xt_star, args.num_f, return_idx=False, x_grid=args.num_x)
    return xt_f


def get_ic(system, grid_bounds, num_x):
    x0_grid, x1_grid, t0_grid, t1_grid = grid_bounds
    h = (x1_grid - x0_grid) / num_x
    x0 = np.arange(x0_grid, x1_grid, h)
    x0 = x0.flatten()[:, None]
    u0 = system.u0(x0)

    # stack with t0
    t0 = t0_grid * np.ones_like(x0)
    xt = np.hstack((x0, t0))

    return xt, u0


def add_extra_sup(xt_star, u_star, num_points, use_bc, x1):
    # select points from boundary
    idx_sup, xt_sup = sample_random(xt_star, num_points, return_idx=True, x_grid=x1, inner=False, use_bc=use_bc)
    u_sup = u_star[idx_sup]

    return xt_sup, u_sup


def get_sup_points(system, xt_star, u_star, grid_bounds, args):
    # IC
    xt_u, u = get_ic(system, grid_bounds, args.num_x)

    # add extra supervision points
    if args.add_sup > 0:
        x0_grid, x1_grid, t0_grid, t1_grid = grid_bounds
        xt_sup, u_sup = add_extra_sup(xt_star, u_star, args.add_sup, args.use_bc, x1_grid)

        # add to initial conditions
        xt_u = np.concatenate([xt_u, xt_sup], axis=0)
        u = np.concatenate([u, u_sup], axis=0)

    return xt_u, u


def get_point_collection(system, args):
    filename = system.get_filename()
    x_star, t_star, u_star, grid_bounds = load_solution(args.data_path, filename)

    # get mesh
    xx, tt = np.meshgrid(x_star, t_star)
    xt_star = np.stack((xx, tt), axis=2)
    xt_star_flat = xt_star.reshape(-1, 2)

    # PDE points
    xt_f = get_xt_f(xt_star_flat, args)
    xt_u, u = get_sup_points(system, xt_star_flat, u_star, grid_bounds, args)

    return xt_star, u_star, xt_f, xt_u, u, grid_bounds


def load_solution(path, filename):
    full_path = os.path.join(path, filename)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"No such file {full_path}")

    with h5py.File(full_path, 'r') as f:
        # return numpy arrays from the file
        u = f['u'][()].reshape(-1, 1)
        x, t = f['x'][()], f['t'][()]
        xx, tt = np.meshgrid(x, t)
        grid_bounds = f['grid_bounds'][()].tolist()

        assert xx.reshape(-1, 1).shape == u.shape, f"Shapes of x and u are not compatible: " \
                                                   f"{xx.reshape(-1, 1).shape} and {u.shape}"
        x = x.reshape(-1)
        t = t.reshape(-1)
        print(f"Solution shape is {u.shape}")
        return x, t, u, grid_bounds
