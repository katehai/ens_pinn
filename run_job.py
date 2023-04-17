import argparse
import wandb

from run_ensemble import run_training
from utils import upd_save_path
from pde.pde import Rd, Reaction, Convection, Heat, Diffusion


def parse_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--sys', default='reaction', type=str, help='the name of the system')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--model_type', default="ens", help='model for the run: ens or pinn')
    parser.add_argument('--layers', default='2,50,50,50,50,1', help='network architecture')
    parser.add_argument('--activation', default='tanh', help='network activation function')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--n_nets', default=5, type=int, help='number of networks in the ensemble')
    parser.add_argument('--num_f', default=1000, type=int, help='number of points for PDE loss')
    parser.add_argument('--num_x', default=256, type=int, help='number of points for IC loss')
    parser.add_argument('--num_t', default=100, type=int, help='number of points for BC loss')
    parser.add_argument('--threshold', default=0.0004, type=float, help='variance threshold for ensemble')
    parser.add_argument('--delta', default=0.001, type=float, help='upper bound for allowed error for fixing a target')
    parser.add_argument('--pde_initial', default=True, type=bool, help='if IC points are included in PDE loss as well')
    parser.add_argument('--pde_boundary', action='store_true', help='if BC points are included in PDE loss as well')
    parser.add_argument('--use_w', action='store_true', help='identifies if ratio of points are used as loss weights')
    parser.add_argument('--m', default=1., type=float, help='supervision loss multiplier')
    parser.add_argument('--n_epochs', default=1000, type=int, help='number of networks in the ensemble')
    parser.add_argument('--n_epochs_1st', default=5000, type=int, help='number of networks in the ensemble')
    parser.add_argument('--n_iter', default=60, type=int, help='number of iterations for ensemble training')
    parser.add_argument('--pde_tmax', default=0.1, type=float, help='distance threshold to add to PDE points')
    parser.add_argument('--distance', default=0.05, type=float, help='distance threshold to fix the target')
    parser.add_argument('--add_sup', default=0, type=int, help='number of extra supervision points')
    parser.add_argument('--use_bc', action='store_true', help='identifies if extra sup. points can be from boundaries')
    parser.add_argument('--const', default=5, type=float, help='constant in the PDE')
    parser.add_argument('--log_every', default=10, type=int, help='ensemble progress is saved every k-th iterations')
    parser.add_argument('--norm', default=True, type=bool, help='identifies if inputs should be normalized')
    parser.add_argument('--norm_min_max', action='store_true', help='identifies normalization between 0 and 1')
    parser.add_argument('--include_pl', action='store_true', help='identifies if loss for pseudo-labels is included')
    parser.add_argument('--use_two_opt', action='store_true', help='identifier if LBFGS is used in training after Adam')
    parser.add_argument('--data_path', default='data', type=str, help='path for data input')
    parser.add_argument('--save_path', default='results', type=str, help='path for saving logs of the run')
    parser.add_argument('--no_wandb', action='store_true', help='identifier if wandb library is disabled')
    parser.add_argument('--wandb_usr', default='your_user', type=str, help='the name of wandb user if enabled')
    parser.add_argument('--wandb_online', default=True, type=bool, help='wandb online mode identifier')

    args = parser.parse_known_args()[0]
    args.run_name = args.model_type  # wandb name of the run
    return args


def initialize_wandb(args):
    if args.no_wandb:
        mode = 'disabled'
    elif args.wandb_online:
        mode = 'online'
    else:
        mode = 'offline'

    name = f'{args.run_name}_{args.sys}_{args.postfix}'
    kwargs = {'project': 'test',
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
    const = args.const
    if args.sys == "convection":
        return Convection(beta=const, framework=framework,
                          postfix=args.postfix)
    elif args.sys == "heat":
        return Heat(d=const, framework=framework, postfix=args.postfix)
    elif args.sys == "diffusion":
        return Diffusion(d=const, framework=framework, postfix=args.postfix)
    elif args.sys == "reaction":
        return Reaction(rho=const, framework=framework, postfix=args.postfix)
    elif args.sys == "rd":
        return Rd(nu=const, rho=5, framework=framework, postfix=args.postfix)
    else:
        raise ValueError(f"Unknown system {args.sys}")


def get_params():
    args = parse_args()

    # create a separate folder for each seed
    upd_save_path(args.seed, args)
    print(f"Save path is {args.save_path}")

    # set parameters for logging
    args.log_model = False
    args.postfix = args.const if args.include_pl else f'{args.const}_no_sup'

    # calculate the number of epochs for PINN training
    if args.model_type == 'pinn':
        args.n_epochs = (args.n_iter - 1) * args.n_epochs + args.n_epochs_1st

    return args


def run():
    args = get_params()
    initialize_wandb(args)

    print(f"Run {args.sys}.")
    framework = 'pytorch'
    system = get_system(args, framework)

    run_training(system, args, do_plot=True)
    return


if __name__ == "__main__":
    run()
