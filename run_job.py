import csv
import os
import argparse
import numpy as np
import wandb

from run_ensemble import main
from utils import load_solution, upd_save_path
from run_utils import check_conv_param
from pde import *


def parse_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--sys', default='convection', type=str, help='the name of the system')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--model_type', default="ens", help='model for the run: ens or pinn')
    parser.add_argument('--layers', default='50,50,50,50,1', help='network architecture')
    parser.add_argument('--activation', default='tanh', help='network activation function')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--n_nets', default=5, type=int, help='number of networks in the ensemble')
    parser.add_argument('--N_f', default=1000, type=int, help='number of points for PDE loss')
    parser.add_argument('--N_x', default=256, type=int, help='number of points for IC loss')
    parser.add_argument('--N_t', default=100, type=int, help='number of points for BC loss')
    parser.add_argument('--threshold', default=0.0004, type=float, help='variance threshold for ensemble')
    parser.add_argument('--delta', default=0.001, type=float, help='upper bound for allowed error for fixing a target')
    parser.add_argument('--pde_initial', default=True, type=bool, help='if IC points are included in PDE loss as well')
    parser.add_argument('--pde_boundary', action='store_true', help='if BC points are included in PDE loss as well')
    parser.add_argument('--upd_targets', action='store_true', help='defines if pseudo-labels are updated')
    parser.add_argument('--use_sup_lbfgs', action='store_true', help='defines if pseudo-labels are used in LBFGS updates')
    parser.add_argument('--use_lbfgs_wrapper', action='store_true', help='defines if LBFGS-B optimizer is used from scipy instead of Pytorch implementation')
    parser.add_argument('--lhs_sampling', action='store_true', help='use Latin hypercube sampling for X_f points')
    parser.add_argument('--auto_iter', action='store_true', help='defines n_iter dynamically')
    parser.add_argument('--no_w', action='store_true', help='identifies if ratio of points are used as weights')
    parser.add_argument('--L', default=1., type=float, help='supervision loss multiplier')
    parser.add_argument('--n_epochs', default=1000, type=int, help='number of networks in the ensemble')
    parser.add_argument('--n_epochs_1st', default=5000, type=int, help='number of networks in the ensemble')
    parser.add_argument('--epochs_max', action='store_true', help='stop training after a certain number of epochs')
    parser.add_argument('--n_iter', default=0, type=int, help='number of iterations for each system is 0, set from hardcoded values')
    parser.add_argument('--pde_tmax', default=0.1, type=float, help='distance from the closest point with a target to add to PDE points')
    parser.add_argument('--distance', default=0.05, type=float, help='distance from the closest point with a target to fix the target')   
    parser.add_argument('--add_sup', default=0, type=int, help='number of extra supervision points added')
    parser.add_argument('--use_bc', action='store_true', help='identifies if sample supervision points from boundary conditions.')
    parser.add_argument('--postfix', default='', help='identifier of parameters to the experiment')
    parser.add_argument('--log_every', default=10, type=int, help='integer k which identifies how often to save ensemble progress, every k-th iterations.')
    parser.add_argument('--zero_bias', action='store_true', help='identifies if zero bias should be used in convection runs.')
    parser.add_argument('--norm', default=True, type=bool, help='identifies if inputs should be normalized.')
    parser.add_argument('--norm_min_max', action='store_true', help='identifies if inputs are normalized between 0 and 1.')
    parser.add_argument('--exclude_extra', action='store_true', help='identifies if loss term with pseudo-labels is used in optimization.')
    parser.add_argument('--use_lbfgs', action='store_true', help='identifier if the second optimizer is used in the model training')
    parser.add_argument('--data_path', default='data', type=str, help='path for data input')
    parser.add_argument('--no_wandb', action='store_true', help='identifier if wandb library is disabled')
    parser.add_argument('--wandb_usr', default='your_usr', type=str, help='the name of wandb user if enabled')
    parser.add_argument('--wandb_online', default=True, type=bool, help='wandb online mode identifier')
    parser.add_argument('--wandb_save_interm', default=True, type=bool, help='if log images and model weights on the intermediate steps')
    
    args = parser.parse_known_args()[0]
    args.run_name = args.model_type  # wandb name of the run
    return args

def get_n_iter(sys, const, args):
    n_iter = 1
    if sys == 'convection':
        beta = const
        if beta < 30:
            n_iter = 60
        elif beta < 40:
            n_iter = 100
        else:
            n_iter = 150
    elif sys == 'heat' or sys == 'diffusion':
        n_iter = 100 if const >= 10 else 80
    elif sys == 'rd':
        n_iter = 60
    elif sys == 'reaction':
        n_iter = 60
        
#     if args.n_epochs != 1000:
#         ratio = 1000 / args.n_epochs
#         n_iter = int(n_iter * ratio)
        
    return n_iter


def run(args):
    print(f"Run {args.sys}.")
    framework = 'pytorch'
    system = get_system(args, framework)
    args.log_model = False  # for now    
    
    const = get_const_param(args.postfix)
    if args.auto_iter:
        args.n_iter = -1 
    elif args.n_iter == 0:
        args.n_iter = get_n_iter(args.sys, const, args)
        
    print(f"Number of iteration is {args.n_iter}")      
    
    if args.exclude_extra:
        args.postfix = f'{args.postfix}_no_sup'
    
    args.include_extra = not args.exclude_extra   # the loss term is used for pseudo-labels
    args.use_w = not args.no_w                    # switch off weights for ablations

    # create a separate folder for each seed
    upd_save_path(args.seed, args)
    
    if args.model_type == 'pinn':
        args.n_epochs = (args.n_iter - 1) * args.n_epochs + args.n_epochs_1st

    print(f"Model is {args.model_type}")
        
    # log args to wandb, n_iter and postfix are overridden
    wandb.config.update(args, allow_val_change=True)
    
    # load data
    filename = system.get_filename()
    x_star, t_star, u_sol, grid_bounds = load_solution(args.data_path, filename)
    system = check_conv_param(system, args, x_star) 

    x_star = x_star.reshape(-1)
    t_star = t_star.reshape(-1)
    X, T = np.meshgrid(x_star, t_star)
    print(f"Solution shape is {u_sol.shape}")
    
    model = main(system, x_star, t_star, u_sol, grid_bounds, args, do_plot=True)
    return args


def get_save_path(args):
    save_path = 'results'
    return save_path


def get_const_param(postfix):
    res = None
    if postfix != '':
        res = eval(postfix)
        if type(res) == int or type(res) == float:
            print(f'const is {res}')
            # use parsed number as a constant
        else:
            # set postfix for const to empty
            res = None
    return res


def initialize_wandb(args):
    if args.no_wandb:
        mode = 'disabled'
    elif args.wandb_online:
        mode = 'online'
    else:
        mode = 'offline'        
        
    name = f'{args.run_name}_{args.sys}_{args.postfix}'
    
    kwargs = {'project': 'test', # test
              'entity': args.wandb_usr,
              'config': args,
              'name': name,
              'mode': mode,
              'group': f'{args.sys}_{args.postfix}',
              'tags': ['grid_x_f']
             }
    wandb.init(**kwargs)
    wandb.save('*.txt')  # not sure if it is the best option
    return


def get_system(args, framework):
    if args.sys == "convection":
        beta = get_const_param(args.postfix)
        pde = Convection(beta=beta, framework=framework, 
                         postfix=args.postfix)
    
    elif args.sys == "heat":
        const = get_const_param(args.postfix)
        pde = Heat(d=const, framework=framework, postfix=args.postfix) 
    
    elif args.sys == "diffusion":
        const = get_const_param(args.postfix)
        pde = Diffusion(d=const, framework=framework, postfix=args.postfix)
        
    elif args.sys == "reaction":
        const = get_const_param(args.postfix)
        pde = Reaction(rho=const, framework=framework, postfix=args.postfix)
    
    elif args.sys == "rd":
        const = get_const_param(args.postfix)
        pde = Rd(nu=const, rho=5, framework=framework, postfix=args.postfix)
        
    elif args.sys == "allen_cahn":
        pde = AllenCahn(framework=framework, postfix=args.postfix)    
    else:
        print("Unknown system!!!")
        
    return pde


if __name__ == "__main__":
    args = parse_args()
    args.save_path = get_save_path(args)
    print(f"Save path is {args.save_path}")
    
    initialize_wandb(args)
    args = run(args)

