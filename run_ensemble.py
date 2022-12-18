import torch
import matplotlib.pyplot as plt
import json
import h5py
from pyDOE import lhs

from net import *
from utils import *
from plot_utils import *

from solution_nets import Normalize
        
        
def get_all_points(X, T):
    return np.hstack((X.flatten()[:, None], T.flatten()[:, None])) # all the x,t "test" data
        
        
def get_X_f(X_star, grid_bounds, args, fix_X_f=True):
    x0_grid, x1_grid, t0_grid, t1_grid = grid_bounds
    # sample collocation points only from the area (where the PDE is enforced)
    if args.lhs_sampling:
        lb = np.array([x0_grid, t0_grid])
        ub = np.array([x1_grid, t1_grid])
        X_f_train = lb + (ub - lb)*lhs(2, args.N_f)
    else:
        X_f_train = sample_random(X_star, args.N_f, return_idx=False,
                                  xgrid=args.N_x, fix_X_f=fix_X_f)
    return X_f_train


def get_bc(grid_bounds, N_t):
    x0_grid, x1_grid, t0_grid, t1_grid = grid_bounds
    t = np.linspace(t0_grid, t1_grid, N_t)[:,None]
    x_lb = x0_grid * np.ones_like(t)
    x_ub = x1_grid * np.ones_like(t)
    
    lb = np.hstack((x_lb, t))
    ub = np.hstack((x_ub, t))

    return lb, ub

def get_ic(system, grid_bounds, N_x):
    x0_grid, x1_grid, t0_grid, t1_grid = grid_bounds
    h = (x1_grid - x0_grid) / N_x
    # x0 = np.linspace(x0_grid, x1_grid, N_x)
    x0 = np.arange(x0_grid, x1_grid, h)
    x0 = x0.flatten()[:,None]
    u0 = system.u0(x0)
    
    # stack with t0
    t0 = t0_grid * np.ones_like(x0)
    X0 = np.hstack((x0, t0))
    
    return X0, u0


def parse_layers(in_dim, layers_str):
    layers = [int(item) for item in layers_str.split(',')]
    layers.insert(0, in_dim)
    return layers


def add_extra_sup(X_u_train, u_train, X_star, u_star, N, use_bc, x1):
    # select points from boundary
    idx_sup, X_sup = sample_random(X_star, N, return_idx=True, xgrid=x1, 
                                   inner=False, use_bc=use_bc)
    u_sup = u_star[idx_sup]

    # add to initial conditions
    X_u_train = np.concatenate([X_u_train, X_sup], axis=0)
    u_train = np.concatenate([u_train, u_sup], axis=0)
    return X_u_train, u_train


def get_mapping(grid_bounds, norm, norm_min_max=False):
    x0_grid, x1_grid, t0_grid, t1_grid = grid_bounds
    if norm:        
        if norm_min_max:
            # normalize [0, 1]
            mean_x = x0_grid
            mean_t = t0_grid
            std_x = x1_grid - x0_grid 
            std_t = t1_grid - t0_grid
        else:
            # include normalizing mapping
            mean_x = std_x = x1_grid / 2.
            mean_t = std_t = t1_grid / 2.

        mean = torch.tensor([mean_x, mean_t], dtype=torch.float32)
        std = torch.tensor([std_x, std_t], dtype=torch.float32)
        print('norm', mean, std)
        mapping = Normalize(mean, std)
    else:
        mapping = None
    return mapping
    

def main(system, x_star, t_star, u_star, grid_bounds, args, model=None, do_plot=True):
    # CUDA support
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    postfix = args.postfix 
    
    set_seed(args.seed)
    
    # get mesh
    x0_grid, x1_grid, t0_grid, t1_grid = grid_bounds
    X, T = np.meshgrid(x_star, t_star)
    X_star = get_all_points(X, T)
    
    # PDE points
    X_f_train = get_X_f(X_star, grid_bounds, args)
    
    # IC & BC
    X_u_train, u_train = get_ic(system, grid_bounds, args.N_x)
    bc_lb, bc_ub = get_bc(grid_bounds, args.N_t)

    args.n_bc = len(bc_lb)

    # add extra supervision points
    if args.add_sup > 0:
        X_u_train, u_train = add_extra_sup(X_u_train, u_train, X_star, u_star,
                                           args.add_sup, args.use_bc, x1_grid)

    mapping = get_mapping(grid_bounds, args.norm, args.norm_min_max)

    # get net layers
    in_dim = X_u_train.shape[-1]
    layers = parse_layers(in_dim, args.layers)

    n_bc = len(bc_lb)
    norm_x, norm_t = x1_grid, t1_grid
    optimizer_name = 'Adam'
    if model is None:
        if args.model_type == 'pinn':
            model = BaselinePINN(system, args.N_f, args.n_bc, X_u_train, u_train, X_f_train, bc_lb, bc_ub, layers,
                                 optimizer_name, args.lr, args.L,
                                 args.activation, mapping=mapping,
                                 save_path=args.save_path, postfix=postfix,
                                 pde_initial=args.pde_initial,
                                 pde_boundary=args.pde_boundary,
                                 use_w=args.use_w, 
                                 use_two_opt=args.use_lbfgs,
                                 use_lbfgs_wrapper=args.use_lbfgs_wrapper
                                )
        elif args.model_type == 'ens':
            model = PINNMultParallel2D(system, args.N_f, args.n_bc, args.n_nets, 
                                       X, T, X_u_train, u_train, X_f_train,
                                       bc_lb, bc_ub, norm_x, norm_t, layers,
                                       optimizer_name, args.lr, args.L,
                                       args.activation,
                                       mapping=mapping, distance=args.distance,
                                       pde_tmax=args.pde_tmax,
                                       save_path=args.save_path, postfix=postfix, 
                                       log_every=args.log_every,
                                       include_extra=args.include_extra,
                                       pde_initial=args.pde_initial, 
                                       pde_boundary=args.pde_boundary,
                                       log_model=args.log_model, 
                                       use_lbfgs=args.use_lbfgs,
                                       threshold=args.threshold, 
                                       delta=args.delta,
                                       use_w=args.use_w,
                                       upd_targets=args.upd_targets,
                                       use_sup_lbfgs=args.use_sup_lbfgs,
                                       use_lbfgs_wrapper=args.use_lbfgs_wrapper
                                      )
        else:
            print(f'The model type is not identified')
            return

    u_star_tensor = torch.tensor(u_star)
    if args.model_type == 'pinn':
        x_star_tensor = torch.tensor(X_star).float()
        model.train_adam(n_epochs=args.n_epochs, x_star=x_star_tensor, 
                         u_star=u_star_tensor)
    else:
        model.train(n_iter=args.n_iter, n_epochs=args.n_epochs,
                    n_epochs_1st=args.n_epochs_1st, 
                    epochs_max=args.epochs_max,
                    u_star=u_star_tensor)

    model.eval()
    u_pred = model.predict(X_star)

    if do_plot:
        # plot fit results
        u_pred1 = u_pred.reshape(*X.shape)
        u_star1 = u_star.reshape(*X.shape)

        path_gt = os.path.join(args.save_path, f"{model.model_name}_{system.name}{postfix}_ugt.pdf")
        path_pred = os.path.join(args.save_path, f"{model.model_name}_{system.name}{postfix}_upred.pdf")

        bound_all = np.concatenate([bc_lb, bc_ub], axis=0)
        idx_train = np.arange(len(X_f_train)) if args.model_type == 'pinn' else model.pde_idx_fixed
        idx_fixed = None if args.model_type == 'pinn' else model.idx_fixed
        bound = bound_all if args.model_type == 'pinn' else np.concatenate([model.bc_lb.detach().cpu().numpy(), 
                                                                            model.bc_ub.detach().cpu().numpy()],
                                                                           axis=0)
        fig_gt = plot_2d_sets(X, T, u_star1, u_pred_min=np.min(u_star1), u_pred_max=np.max(u_star1),
                              X_f_all=X_f_train, idx_train=idx_train, idx_fixed=idx_fixed,
                              x_sup=X_u_train[:, 0], t_sup=X_u_train[:, 1], 
                              bound=bound, bound_all=bound_all,
                              title='U GT', save_path=path_gt,
                              save_png=True)

        fig_pred = plot_2d_sets(X, T, u_pred1, u_pred_min=np.min(u_star1), u_pred_max=np.max(u_star1),
                                X_f_all=X_f_train, idx_train=idx_train, idx_fixed=idx_fixed,
                                x_sup=X_u_train[:, 0], t_sup=X_u_train[:, 1], 
                                bound=bound, bound_all=bound_all,
                                title='U predicted', save_path=path_pred,
                                save_png=True)
        wandb.log({'u_gt': wandb.Image(fig_gt)})
        wandb.log({'u_pred': wandb.Image(fig_pred)})

    sys_name = f'{system.name}{postfix}_{model.model_name}'
    get_error(u_pred, u_star, sys_name, path=args.save_path)
    save_u_pred(u_pred, args.save_path, sys_name)

    if args.model_type == 'ens':
        save_ens_info(model, folder=args.save_path, save_wandb=True)
        wandb.log({'n_total_iter': model.net.iter})
        print(f"Number of total epochs is {model.net.iter}")

    if args.model_type == 'pinn':
        path_model = os.path.join(args.save_path, f'{sys_name}.pth')
        torch.save(model.dnn.state_dict(), path_model)
        wandb.save(path_model)
        
        wandb.log({'n_total_iter': model.iter})
        print(f"Number of total epochs is {model.iter}")

    return model
