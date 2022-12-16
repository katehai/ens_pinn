import os
import math
import numpy as np
import random
import torch
import h5py
import wandb
import matplotlib.pyplot as plt
from plot_utils import plot_2d


# the function is adopted from https://github.com/a1k12/characterizing-pinns-failure-modes
def reaction(u, rho, dt):
    """ du/dt = rho*u*(1-u)
    """
    factor_1 = u * np.exp(rho * dt)
    factor_2 = (1 - u)
    u = factor_1 / (factor_2 + factor_1)
    return u


# the function is adopted from https://github.com/a1k12/characterizing-pinns-failure-modes
def diffusion(u, nu, dt, IKX2):
    """ du/dt = nu*d2u/dx2
    """
    factor = np.exp(nu * IKX2 * dt)
    u_hat = np.fft.fft(u)
    u_hat *= factor
    u = np.real(np.fft.ifft(u_hat))
    return u


def get_grid(x0, xN, num_x, t0, tN, num_t):
    h = xN / num_x
    x = np.arange(x0, xN, h)  # not inclusive of the last point
    t = np.linspace(t0, tN, num_t).reshape(-1, 1)
    X, T = np.meshgrid(x, t)
    return x, t, X, T


def load_solution(path, filename):
    full_path = os.path.join(path, filename)
    assert os.path.exists(full_path), f"No such file {full_path}"
    
    with h5py.File(full_path, 'r') as f:
        # return numpy arrays from the file
        u = f['u'][()].reshape(-1, 1)
        x, t = f['x'][()], f['t'][()]
        X, T = np.meshgrid(x, t)
        grid_bounds = f['grid_bounds'][()].tolist()

        assert X.reshape(-1, 1).shape == u.shape, f"Shapes of x and u are not compatible: " \
                                                  f"{X.reshape(-1, 1).shape} and {u.shape}"
        return x, t, u, grid_bounds


def check_outdir(outdir):
    print(f"Check output dir {outdir}")    
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
        print("Created directory...")
    else:
        print("The directory exists.")
    
def upd_save_path(seed, args):
    if args.save_path is not None:
        new_path = os.path.join(args.save_path, str(seed))
        args.save_path = new_path
        
        # check if the directory exists and create otherwise
        check_outdir(new_path)

def get_log_filepath(save_path, metric, model_name, system, postfix):
    file = f"{metric}_{model_name}_{system}{postfix}.csv"
    filepath = os.path.join(save_path, file)
    return filepath

def get_lossfile_headers():
    loss_names = ['sup', 'bc', 'dbc', 'lb', 'ub', 'f', 'sup_ex', 'l_sup_ex', 'l_sup', 
                  'l_bc', 'l_sup_w', 'l_ut_w', 'l_sup_total', 'l_f', 'total']
    return loss_names

def get_errorfile_headers():
    error_names = ['err_rel', 'err_abs']
    return error_names

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# def get_sobol_points(l, r, n_samples):
#     sampler = skopt.sampler.Sobol(skip=0, randomize=False)
#     space = [(0.0, 1.0)]
#     x_points0 = np.array(sampler.generate(space, n_samples + 2)[2:], dtype=np.float32)
#     x_points = (r - l) * x_points0 + l
#     return x_points
    
# def sample_sobol(t0, tN, x0, xN, n_samples):
#     x = get_sobol_points(x0, xN, n_samples)
#     t = get_sobol_points(t0, tN, n_samples)
#     t = np.random.default_rng().permutation(t)
#     return np.hstack((x, t))

def get_prob_inner(n, xgrid):
    p = np.ones(n)
    if xgrid is not None:
        p[0:xgrid] = 0. ## initial conditions
        p[::xgrid] = 0. ## lower bound (e.g. we can have inner points on the upper bound
    return p
        
def get_prob_bound(n, xgrid, use_bc=True):
    p = np.zeros(n)
    if xgrid is not None:
        p[-xgrid:] = 1. # points on t=1 for all X
        if use_bc:
            p[::xgrid] = 1. # lower bound
            # p[xgrid-1::xgrid] = 1. # upper bound -> upper bound is not included in X_star
    return p

def sample_random(X_all, N, seed=0, return_idx=False, xgrid=None, inner=True,
                  use_bc=True, p=None, fix_X_f=False):
    if fix_X_f:
        # use the same X_f for different random initialization of network weights
        # it is needed for test purposes
        st0 = np.random.get_state()
        np.random.seed(seed)
    
    if p is None:
        if inner:
            p = get_prob_inner(len(X_all), xgrid)
        else:
            p = get_prob_bound(len(X_all), xgrid, use_bc)
            print('Sum of prob is ', np.sum(p))
    
    # normalize prob vector
    p /= np.sum(p)
    
    idx = np.random.choice(X_all.shape[0], N, replace=False, p=p)
    X_sampled = X_all[idx, :]
    
    if fix_X_f:
        # revert seed
        np.random.set_state(st0)

    if return_idx:
        return idx, X_sampled
    return X_sampled


# the function is adopted from https://github.com/a1k12/characterizing-pinns-failure-modes
def choose_nonlinearity(name):
    nl = None
    if name == 'tanh':
        nl = torch.tanh
    elif name == 'relu':
        nl = torch.relu
    elif name == 'sigmoid':
        nl = torch.sigmoid
    elif name == 'softplus':
        nl = torch.nn.functional.softplus
    elif name == 'celu':
        nl = torch.nn.functional.celu
    elif name == 'selu':
        nl = torch.nn.functional.selu
    elif name == 'gelu':
        nl = torch.nn.functional.gelu
    elif name == 'elu':
        nl = torch.nn.functional.elu
    elif name == 'swish':
        nl = lambda x: x * torch.sigmoid(x)
    elif name == 'sin':
        nl = torch.sin
    else:
        raise ValueError("nonlinearity not recognized")
    return nl


def save_model(model, folder='run_models/', model_name=''):
    path = os.path.join(folder, f'net_ensemble{model_name}.pth')
    torch.save(model.net.dnn.state_dict(), path)


def plot_mean_and_var(u_pred_list, X, T, X_f_train):
    u_pred_arr = np.stack(u_pred_list, axis=0)
    u_pred_mean = np.mean(u_pred_arr, axis=0)
    u_pred_var = np.var(u_pred_arr, axis=0)
    
    plot_2d(X, T, u_pred_mean, X_f_train=X_f_train, title='U mean')
    
    u_pred_min, u_pred_max = np.min(u_pred_var), np.max(u_pred_var)
    plot_2d(X, T, u_pred_var, u_pred_min, u_pred_max, X_f_train=X_f_train, title='U var')


def load_model(model, folder, filename, is_ens=True, load_optimizer=True, postfix=''):    
    # load model weights
    model.load_state_dict(torch.load(os.path.join(folder, filename)))
    
    filename_opt = get_opt_path(filename)
    path_opt = os.path.join(folder, filename_opt)
    
    if is_ens:      
        device = model.X_star.device
        print(device)

        with open(f'{folder}/idx_fixed{postfix}.txt', 'r') as f:
            idx_fixed = np.loadtxt(f, delimiter = ',').astype(dtype=np.int64)
            model.idx_fixed = torch.tensor(idx_fixed).to(device)
            # print(model.idx_fixed)

        with open(f'{folder}/pde_idx_fixed{postfix}.txt', 'r') as f:
            pde_idx_fixed = np.loadtxt(f, delimiter = ',').astype(dtype=np.int64)
            model.pde_idx_fixed = torch.tensor(pde_idx_fixed).to(device)
            # print(model.pde_idx_fixed)

        with open(f'{folder}/bc_idx_fixed{postfix}.txt', 'r') as f:
            bc_idx_fixed = np.loadtxt(f, delimiter = ',').astype(dtype=np.int64)
            model.bc_idx_fixed = torch.tensor(bc_idx_fixed).to(device)
            # print(model.bc_idx_fixed)

        with open(f'{folder}/x_u_extra{postfix}.txt', 'r') as f:
            x_u_extra = np.loadtxt(f, delimiter = ',').astype(dtype=np.float32)
            model.x_u_extra = torch.tensor(x_u_extra).unsqueeze(-1).to(device)
            # print(model.x_u_extra) 

        with open(f'{folder}/t_u_extra{postfix}.txt', 'r') as f:
            t_u_extra = np.loadtxt(f, delimiter = ',').astype(dtype=np.float32)
            model.t_u_extra = torch.tensor(t_u_extra).unsqueeze(-1).to(device)
            # print(model.t_u_extra) 

        with open(f'{folder}/u_extra{postfix}.txt', 'r') as f:
            u_extra = np.loadtxt(f, delimiter = ',').astype(dtype=np.float32)
            model.u_extra = torch.tensor(u_extra).unsqueeze(-1).to(device)
            # print(model.u_extra)
        model.update_sup_pde_points()
        if load_optimizer:
            model.net.optimizer.load_state_dict(torch.load(path_opt))
    else:
        if load_optimizer:
            model.optimizer.load_state_dict(torch.load(path_opt))
    return model


def save_ens_info(model, folder='runs', save_wandb=False):
    postfix = f"_{model.net.system.name}{model.postfix}_ep_{model.net.iter}"
    filename = f"{model.model_name}{postfix}.tar"
    save_model_info(model, folder, filename, postfix=postfix, save_wandb=save_wandb)


def save_model_info(model, folder, filename='ens.tar', is_ens=True, postfix='',
                    save_wandb=False):
    if is_ens:
        with open(f'{folder}/idx_fixed{postfix}.txt', 'w') as f:
            if model.idx_fixed is not None:
                np.savetxt(f, to_np(model.idx_fixed), delimiter = ',')

        with open(f'{folder}/pde_idx_fixed{postfix}.txt', 'w') as f:
            np.savetxt(f, to_np(model.pde_idx_fixed), delimiter = ',')

        with open(f'{folder}/bc_idx_fixed{postfix}.txt', 'w') as f:
            np.savetxt(f, to_np(model.bc_idx_fixed), delimiter = ',')

        with open(f'{folder}/x_u_extra{postfix}.txt', 'w') as f:
            if model.x_u_extra is not None:
                np.savetxt(f, to_np(model.x_u_extra), delimiter = ',')

        with open(f'{folder}/t_u_extra{postfix}.txt', 'w') as f:
            if model.t_u_extra is not None:
                np.savetxt(f, to_np(model.t_u_extra), delimiter = ',')

        with open(f'{folder}/u_extra{postfix}.txt', 'w') as f:
            if model.u_extra is not None:
                np.savetxt(f, to_np(model.u_extra), delimiter = ',')
        
    # save model_weights
    path_model = os.path.join(folder, filename)
    torch.save(model.state_dict(), path_model)
    
    filename_opt = get_opt_path(filename)
    path_opt = os.path.join(folder, filename_opt)
    optimizer = model.net.optimizer if is_ens else model.optimizer
    torch.save(optimizer.state_dict(), path_opt)
    
    # log to wandb
    if save_wandb:
        wandb.save(f'{folder}/idx_fixed{postfix}.txt')
        wandb.save(f'{folder}/pde_idx_fixed{postfix}.txt')
        wandb.save(f'{folder}/bc_idx_fixed{postfix}.txt')
        wandb.save(f'{folder}/x_u_extra{postfix}.txt')
        wandb.save(f'{folder}/t_u_extra{postfix}.txt')
        wandb.save(f'{folder}/u_extra{postfix}.txt')
        wandb.save(path_model)
        wandb.save(path_opt)

    
def save_results(errors, save_path, system):
    filepath = os.path.join(save_path, f'result_{system}.txt')
    with open(filepath, "w") as out:
        out.write(' '.join([str(err) for err in errors]))
        out.write('\n')
    wandb.save(filepath)
    return


def calc_error(u_pred, u_sol):
    error_u_rel = np.linalg.norm(u_pred - u_sol) / np.linalg.norm(u_sol)
    error_u_abs = np.mean(np.abs(u_pred - u_sol))
    error_u_linf = np.linalg.norm(u_sol - u_pred, np.inf) / np.linalg.norm(u_sol, np.inf)
    return error_u_rel, error_u_abs, error_u_linf


def log_error(u_pred, u_sol):
    error_u_rel, error_u_abs, error_u_linf = calc_error(u_pred, u_sol)
    
    wandb.log({'err_rel': error_u_rel,
               'err_abs': error_u_abs,
               'err_linf': error_u_linf})


def get_error(u_pred, u_sol, system, path=None):   
    error_u_rel, error_u_abs, error_u_linf = calc_error(u_pred, u_sol)

    print(f'Error u rel: {error_u_rel:.3e}')
    print(f'Error u abs: {error_u_abs:.3e}')
    print(f'Error u linf: {error_u_linf:.3e}')
    
    wandb.log({'err_rel': error_u_rel,
               'err_abs': error_u_abs,
               'err_linf': error_u_linf})

    # save results
    if path is not None:
        errors = [error_u_rel, error_u_abs, error_u_linf]
        save_results(errors, path, system)
    return


def save_plot(t, x, u, system, path=None):
    print("saving the last plot")
    T, X = np.meshgrid(t, x)
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    mappable = ax.pcolor(T, X, u, cmap='jet')
    fig.colorbar(mappable, ax=ax)
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    
    wandb.log({'u_plt': wandb.Image(fig)})

    if path is not None:
        plot_name = f"{system}_pred.pdf"
        full_path = os.path.join(path, plot_name)
        fig.savefig(full_path)
    plt.close(fig)
    

def save_plots(t, x, u_pred, u_sol, system, path=None, postfix='',
               save_interm=False, disable_latex=False):
    if disable_latex:
        # to support older versions without latex
        plt.rc('text', usetex=False)
    
    T, X = np.meshgrid(t, x)
    
    fig = plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(T, X, u_sol, cmap='jet')
    plt.colorbar()
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title(r'Exact u(x)')
    plt.tight_layout()
    
    u_sol_min = np.min(u_sol)
    u_sol_max = np.max(u_sol)

    plt.subplot(1, 3, 2)
    plt.pcolor(T, X, u_pred, vmin=u_sol_min, vmax=u_sol_max, cmap='jet')
    plt.colorbar()
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title(r'Predicted u(x)')
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.pcolor(T, X, np.abs(u_sol - u_pred), cmap='jet')
    plt.colorbar()
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('Absolute error')
    plt.tight_layout()
    
    if save_interm:
        wandb.log({'u_error_plt': wandb.Image(fig)})
    
    if path is not None:
        plot_name = f"{system}_pred_error{postfix}.pdf"    
        full_path = os.path.join(path, plot_name)
        fig.savefig(full_path)
    plt.close(fig)

    
def save_u_pred(u_pred, save_path, sys_name):
    path_u_pred = os.path.join(save_path, f"{sys_name}_u_pred.csv")
    np.savetxt(path_u_pred, u_pred, delimiter=',')
    wandb.save(path_u_pred)
    return

    
def get_opt_path(model_filename):
    return f"{model_filename.split('.')[0]}_opt.tar"
    
def to_np(tensor):
    return tensor.cpu().detach().numpy()
