import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb

from utils import to_np


def plot_data(x, t, u, path, filename):
    xx, tt = np.meshgrid(x, t)
    fig, ax = plt.subplots(1, 1, figsize=(3, 2))
    u = u.reshape(*xx.shape)

    mappable = ax.pcolor(tt.T, xx.T, u.T, cmap='RdBu', alpha=0.9)
    fig.colorbar(mappable, ax=ax)
    ax.set_xlabel('t')
    ax.set_ylabel('x')

    full_path = os.path.join(path, filename)
    fig.savefig(full_path)
        
        
def plot_2d_sets(xx, tt, u_pred, u_pred_min=-1.0, u_pred_max=1.0, xt_f_all=None,
                 idx_train=None, idx_fixed=None,
                 x_sup=None, t_sup=None, bound=None, bound_all=None, idx=None,
                 title='U predicted', save_path=None, cmap='RdBu', save_png=False):
    u_pred = to_np(u_pred)
    fig, ax = plt.subplots(1, figsize=(9, 5))
    color = ax.pcolormesh(tt.T, xx.T, u_pred.T, cmap=cmap, vmin=u_pred_min, vmax=u_pred_max, alpha=0.9)
    ax.set_title(title)
    # set the limits of the plot to the limits of the data
    ax.axis([tt.min(), tt.max(), xx.min(), xx.max(), ])
    fig.colorbar(color, ax=ax)
    
    # Down sample points for plots
    mask_all = np.ones(xt_f_all.shape[0], dtype=bool)
    if idx is None:
        num_points = 500
        idx = np.random.choice(xt_f_all.shape[0], num_points, replace=False)
    mask_all[idx] = 0
        
    if idx_train is not None:
        idx_train = to_np(idx_train)
        train_mask = np.zeros(xt_f_all.shape[0], dtype=bool)
        train_mask[idx_train] = True
    else:
        train_mask = np.zeros(xt_f_all.shape[0], dtype=np.int32)
        
    if idx_fixed is not None:
        idx_fixed = to_np(idx_fixed)
        sup_mask = np.zeros(xt_f_all.shape[0], dtype=bool)
        sup_mask[idx_fixed] = True
    else:
        sup_mask = np.zeros(xt_f_all.shape[0], dtype=np.int32)
        
    # plot initial conditions    
    if x_sup is not None and t_sup is not None:
        # (t, x) pairs
        ax.plot(to_np(t_sup[::10]), to_np(x_sup[::10]), 'ob', markersize=6, clip_on=False)    
        
    if xt_f_all is not None:
        # plot all pde points, (t, x) pairs
        xt_f_all1 = to_np(xt_f_all)[mask_all]
        ax.plot(xt_f_all1[::, 1], xt_f_all1[::, 0], 'ko', fillstyle='none')
        
        # plot used pde points, (t, x) pairs     
        xt_f = to_np(xt_f_all)[mask_all & train_mask]
        # print(xt_f)
        ax.plot(xt_f[:, 1], xt_f[:, 0], '.k', markersize=10)
        
        # plot pseudo-labels
        xt_u_pl = xt_f_all[mask_all & sup_mask]
        if np.sum(mask_all & sup_mask) > 0:
            t_u_pl, x_u_pl = xt_u_pl[:, 1], xt_u_pl[:, 0]
            ax.plot(to_np(t_u_pl), to_np(x_u_pl), 'o', color='deepskyblue', markersize=6)
        
    if bound_all is not None: 
        bound_all = to_np(bound_all)
        bound_all = bound_all.reshape(2, -1, 2)[:, ::2].reshape(-1, 2)
        ax.plot(bound_all[:, 1], bound_all[:, 0], 'o', markersize=5, fillstyle='none', color='red', clip_on=False)
        
    if bound is not None: 
        bound = to_np(bound)
        bound = bound.reshape(2, -1, 2)[:, ::2].reshape(-1, 2)
        ax.plot(bound[:, 1], bound[:, 0], 'o', markersize=6, color='red', clip_on=False)
        
    fig.set_clip_on(False)
    
    if save_path is not None:
        fig.savefig(save_path)
        
        # save png version as well
        if save_png:
            save_path1 = save_path.replace('.pdf', '.png')
            fig.savefig(save_path1)
    else:
        plt.show()
    
    plt.close(fig)
    return fig


def plot_results(args, model, postfix, system, u_pred, u_star):
    pc = model.pc if args.model_type == 'pinn' else model.net.pc

    u_pred1 = u_pred.reshape(*pc.X.shape)
    u_star1 = u_star.reshape(*pc.X.shape)
    path_gt = os.path.join(args.save_path, f"{model.model_name}_{system.name}{postfix}_ugt.pdf")
    path_pred = os.path.join(args.save_path, f"{model.model_name}_{system.name}{postfix}_upred.pdf")
    bound_all = np.concatenate([to_np(pc.bc_lb_all), to_np(pc.bc_ub_all)], axis=0)
    idx_train = np.arange(len(pc.xt_f_all)) if args.model_type == 'pinn' else pc.pde_idx_fixed
    idx_fixed = None if args.model_type == 'pinn' else pc.idx_fixed
    bound = bound_all if args.model_type == 'pinn' else np.concatenate([to_np(pc.bc_lb), to_np(pc.bc_ub)], axis=0)
    fig_gt = plot_2d_sets(pc.X, pc.T, u_star1, u_pred_min=np.min(u_star1), u_pred_max=np.max(u_star1),
                          xt_f_all=pc.xt_f_all, idx_train=idx_train, idx_fixed=idx_fixed,
                          x_sup=pc.x_u, t_sup=pc.t_u,
                          bound=bound, bound_all=bound_all,
                          title='U GT', save_path=path_gt,
                          save_png=True)
    fig_pred = plot_2d_sets(pc.X, pc.T, u_pred1, u_pred_min=np.min(u_star1), u_pred_max=np.max(u_star1),
                            xt_f_all=pc.xt_f_all, idx_train=idx_train, idx_fixed=idx_fixed,
                            x_sup=pc.x_u, t_sup=pc.t_u,
                            bound=bound, bound_all=bound_all,
                            title='U predicted', save_path=path_pred,
                            save_png=True)
    wandb.log({'u_gt': wandb.Image(fig_gt)})
    wandb.log({'u_pred': wandb.Image(fig_pred)})


class PlotMeanVar:
    def __init__(self, u_star, n_inner, save_path, postfix):
        n_plot = 500
        self.plot_idx = np.random.choice(n_inner, n_plot, replace=False)

        # plot titles
        self.u_pred_title = "U median"
        self.u_metric_title = "U var"
        self.save_path = save_path
        self.model_name = 'ens_base'
        self.postfix = postfix
        self.save_png = True

        # used for plots (to preserve the same scale for all plots)
        if u_star is not None:
            self.u_star_min = np.amin(u_star)
            self.u_star_max = np.amax(u_star)
        else:
            self.u_star_min = None
            self.u_star_max = None

    def plot_mean_var(self, model, u_pred_mean, u_pred_var):
        pc = model.net.pc
        bound = torch.cat([pc.bc_lb, pc.bc_ub], dim=0)
        bound_all = torch.cat([pc.bc_lb_all, pc.bc_ub_all], dim=0)
        u_pred_mean = u_pred_mean.reshape(*pc.X.shape)
        u_pred_var = u_pred_var.reshape(*pc.X.shape)
        pred_path = self.get_plot_path(model, self.u_pred_title)

        # to maintain the same scale across plots use min and max of the correct solution
        u_mean_min = self.u_star_min if self.u_star_min is not None else torch.min(u_pred_mean).numpy()
        u_mean_max = self.u_star_max if self.u_star_max is not None else torch.max(u_pred_mean).numpy()

        fig_mean = plot_2d_sets(pc.X, pc.T, u_pred_mean, u_mean_min, u_mean_max,
                                xt_f_all=pc.xt_f_all,
                                idx_train=pc.pde_idx_fixed,
                                idx_fixed=pc.idx_fixed,
                                x_sup=pc.x_u, t_sup=pc.t_u,
                                bound=bound, bound_all=bound_all, idx=self.plot_idx,
                                title=self.u_pred_title, save_path=pred_path,
                                save_png=self.save_png)
        wandb.log({'u_mean_curr': wandb.Image(fig_mean)})

        # plot log of var
        u_pred_var = torch.log(u_pred_var)

        u_var_min, u_var_max = torch.min(u_pred_var), torch.max(u_pred_var)
        metric_path = self.get_plot_path(model, self.u_metric_title)

        fig_var = plot_2d_sets(pc.X, pc.T, u_pred_var, u_var_min, u_var_max,
                               xt_f_all=pc.xt_f_all,
                               idx_train=pc.pde_idx_fixed,
                               idx_fixed=pc.idx_fixed,
                               x_sup=pc.x_u, t_sup=pc.t_u,
                               bound=bound, bound_all=bound_all, idx=self.plot_idx,
                               title=self.u_metric_title, save_path=metric_path, cmap='Greys',
                               save_png=self.save_png)
        wandb.log({'u_var_curr': wandb.Image(fig_var)})

    def get_plot_path(self, model, name):
        name = name.replace(' ', '').lower()
        plot_name = f"{model.model_name}_{model.net.system.name}{self.postfix}_{name}_ep_{model.net.iter}.pdf"
        full_path = os.path.join(self.save_path, plot_name)
        return full_path
