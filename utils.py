import os
import numpy as np
import random
import torch
import wandb


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_out_dir(out_dir):
    print(f"Check output dir {out_dir}")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
        print("Created directory...")
    else:
        print("The directory exists.")


def upd_save_path(seed, args):
    if args.save_path is not None:
        new_path = os.path.join(args.save_path, str(seed))
        args.save_path = new_path
        
        # check if the directory exists and create otherwise
        check_out_dir(new_path)


def load_model(model, folder, filename, is_ens=True, load_optimizer=True, postfix=''):    
    # load model weights
    model.load_state_dict(torch.load(os.path.join(folder, filename)))
    
    filename_opt = get_opt_path(filename)
    path_opt = os.path.join(folder, filename_opt)
    
    if is_ens:      
        device = model.xt_star.device
        print(device)

        with open(f'{folder}/idx_fixed{postfix}.txt', 'r') as f:
            idx_fixed = np.loadtxt(f, delimiter=',').astype(dtype=np.int64)
            model.idx_fixed = torch.tensor(idx_fixed).to(device)

        with open(f'{folder}/pde_idx_fixed{postfix}.txt', 'r') as f:
            pde_idx_fixed = np.loadtxt(f, delimiter=',').astype(dtype=np.int64)
            model.pde_idx_fixed = torch.tensor(pde_idx_fixed).to(device)

        with open(f'{folder}/bc_idx_fixed{postfix}.txt', 'r') as f:
            bc_idx_fixed = np.loadtxt(f, delimiter=',').astype(dtype=np.int64)
            model.bc_idx_fixed = torch.tensor(bc_idx_fixed).to(device)

        with open(f'{folder}/x_u_pl{postfix}.txt', 'r') as f:
            x_u_pl = np.loadtxt(f, delimiter=',').astype(dtype=np.float32)
            model.x_u_pl = torch.tensor(x_u_pl).unsqueeze(-1).to(device)

        with open(f'{folder}/t_u_pl{postfix}.txt', 'r') as f:
            t_u_pl = np.loadtxt(f, delimiter=',').astype(dtype=np.float32)
            model.t_u_pl = torch.tensor(t_u_pl).unsqueeze(-1).to(device)

        with open(f'{folder}/u_pl{postfix}.txt', 'r') as f:
            u_pl = np.loadtxt(f, delimiter=',').astype(dtype=np.float32)
            model.u_pl = torch.tensor(u_pl).unsqueeze(-1).to(device)

        model.update_sup_pde_points()
        if load_optimizer:
            model.net.optimizer.load_state_dict(torch.load(path_opt))
    else:
        if load_optimizer:
            model.optimizer.load_state_dict(torch.load(path_opt))
    return model


def save_ens_info(model, folder='runs', postfix='', save_wandb=False):
    postfix = f"_{model.net.system.name}{postfix}_ep_{model.net.iter}"
    filename = f"{model.model_name}{postfix}.tar"
    save_model_info(model, folder, filename, postfix=postfix, save_wandb=save_wandb)


def save_model_info(model, folder, filename='ens.tar', is_ens=True, postfix='', save_wandb=False):
    if is_ens:
        pc = model.net.pc
        with open(f'{folder}/idx_fixed{postfix}.txt', 'w') as f:
            if pc.idx_fixed is not None:
                np.savetxt(f, to_np(pc.idx_fixed), delimiter=',')

        with open(f'{folder}/pde_idx_fixed{postfix}.txt', 'w') as f:
            np.savetxt(f, to_np(pc.pde_idx_fixed), delimiter=',')

        with open(f'{folder}/bc_idx_fixed{postfix}.txt', 'w') as f:
            np.savetxt(f, to_np(pc.bc_idx_fixed), delimiter=',')

        with open(f'{folder}/x_u_pl{postfix}.txt', 'w') as f:
            if pc.x_u_pl is not None:
                np.savetxt(f, to_np(pc.x_u_pl), delimiter=',')

        with open(f'{folder}/t_u_pl{postfix}.txt', 'w') as f:
            if pc.t_u_pl is not None:
                np.savetxt(f, to_np(pc.t_u_pl), delimiter=',')

        with open(f'{folder}/u_pl{postfix}.txt', 'w') as f:
            if pc.u_pl is not None:
                np.savetxt(f, to_np(pc.u_pl), delimiter=',')
        
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
        wandb.save(f'{folder}/x_u_pl{postfix}.txt')
        wandb.save(f'{folder}/t_u_pl{postfix}.txt')
        wandb.save(f'{folder}/u_pl{postfix}.txt')
        wandb.save(path_model)
        wandb.save(path_opt)

    
def save_results(errors, save_path, system):
    filepath = os.path.join(save_path, f'result_{system}.txt')
    with open(filepath, "w") as out:
        out.write(' '.join([str(err) for err in errors]))
        out.write('\n')
    wandb.save(filepath)
    return


def save_u_pred(u_pred, save_path, sys_name):
    path_u_pred = os.path.join(save_path, f"{sys_name}_u_pred.csv")
    np.savetxt(path_u_pred, u_pred, delimiter=',')
    wandb.save(path_u_pred)
    return

    
def get_opt_path(model_filename):
    return f"{model_filename.split('.')[0]}_opt.tar"


def to_np(tensor):
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()

    return tensor
