import torch
import os
import wandb

from data_utils import get_point_collection
from model.pinn import BaselinePINN, PINNMultParallel2D
from utils import set_seed, save_u_pred, save_ens_info
from error_utils import log_error
from plot_utils import plot_results, PlotMeanVar
from model.solution_nets import Normalize


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


def run_training(system, args, do_plot=True):
    # load data
    xt_star, u_star, xt_f, xt_u, u, grid_bounds = get_point_collection(system, args)
    mapping = get_mapping(grid_bounds, args.norm, args.norm_min_max)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    set_seed(args.seed)

    print(f"Model is {args.model_type}")
    if args.model_type == 'pinn':
        model = BaselinePINN(system, xt_star, u_star, xt_u, u, xt_f, args.num_t, args.num_f, args.layers, args.lr,
                             args.activation, args.m, mapping=mapping, pde_initial=args.pde_initial,
                             pde_boundary=args.pde_boundary, use_w=args.use_w, use_two_opt=args.use_two_opt,
                             grid_bounds=grid_bounds)
        model.to_device(device)

        model.train(n_epochs=args.n_epochs)

    elif args.model_type == 'ens':
        model = PINNMultParallel2D(system, args.n_nets, xt_star, u_star, xt_u,
                                   u, xt_f, args.num_t, args.num_f,
                                   args.layers, args.lr, args.activation, args.m, mapping=mapping,
                                   pde_initial=args.pde_initial, pde_boundary=args.pde_boundary, use_w=args.use_w,
                                   use_two_opt=args.use_two_opt, grid_bounds=grid_bounds, distance=args.distance,
                                   pde_tmax=args.pde_tmax, include_pl=args.include_pl, threshold=args.threshold,
                                   delta=args.delta, save_path=args.save_path, log_model=args.log_model,
                                   log_every=args.log_every)

        model.to_device(device)
        n_inner = xt_f.shape[0]
        plotting = PlotMeanVar(u_star, n_inner, args.save_path, args.postfix)
        model.train(n_iter=args.n_iter, n_epochs=args.n_epochs, n_epochs_1st=args.n_epochs_1st, plotting=plotting)
    else:
        print(f'The model type is not identified')
        return

    model.eval()
    u_pred = model.predict(xt_star.reshape(-1, 2))
    sys_name = f'{system.name}{args.postfix}_{model.model_name}'
    # save_u_pred(u_pred, args.save_path, sys_name)

    if args.model_type == 'ens':
        log_error(u_pred, u_star, sys_name, args.save_path, verbose=True)
        save_ens_info(model, folder=args.save_path, postfix=args.postfix, save_wandb=True)
        wandb.log({'n_total_iter': model.net.iter})
        print(f"Number of total epochs is {model.net.iter}")

    if args.model_type == 'pinn':
        log_error(u_pred, u_star, sys_name, args.save_path, verbose=True)
        path_model = os.path.join(args.save_path, f'{sys_name}.pth')
        torch.save(model.dnn.state_dict(), path_model)
        wandb.save(path_model)

        wandb.log({'n_total_iter': model.iter})
        print(f"Number of total epochs is {model.iter}")

    if do_plot:
        plot_results(args, model, args.postfix, system, u_pred, u_star)

    return
