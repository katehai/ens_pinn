import os
import torch
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import matplotlib.gridspec as gridspec


def to_np(tensor):
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
        
    return tensor


def plot_2d(X, T, u_pred, u_pred_min=-1.0, u_pred_max=1.0, X_f_train=None,\
            x_sup=None, t_sup=None, var_idx=None, x_f_new=None, bound=None,
            x_u_ex=None, t_u_ex=None, u_ex=None,
            x_u_extra=None, t_u_extra=None, # points with known targets on top of initial conditions
            rand_points=None, title='U predicted', save_path=None, cmap='RdBu',
            save_png=False):
    u_pred = to_np(u_pred)
    fig, ax = plt.subplots(1, figsize=(9, 5))
    color = ax.pcolormesh(T.T, X.T, u_pred.T, cmap=cmap, vmin=u_pred_min, vmax=u_pred_max)
    ax.set_title(title)
    # set the limits of the plot to the limits of the data
    ax.axis([T.min(), T.max(), X.min(), X.max(), ])
    fig.colorbar(color, ax=ax)

    if x_f_new is not None:
        x_f_new = to_np(x_f_new)
        ax.plot(x_f_new[:, 1], x_f_new[:, 0], 'x', color='tab:green')
        
    if X_f_train is not None:
        # (t, x) pairs
        X_f_train = to_np(X_f_train)
        ax.plot(X_f_train[:, 1], X_f_train[:, 0], '.k')
        
#         if var_idx is not None:
#             var_idx = to_np(var_idx)
#             X_var = X_f_train[var_idx]
#             ax.plot(X_var[:, 1], X_var[:, 0], 'x', color='tab:orange')
        
    if x_sup is not None and t_sup is not None:
        # (t, x) pairs
        ax.plot(to_np(t_sup), to_np(x_sup), '.b', markersize=12)
    
    if x_u_extra is not None and t_u_extra is not None:
        ax.plot(to_np(t_u_extra), to_np(x_u_extra), '.b', markersize=12)
        
    if rand_points is not None:
        points = to_np(rand_points)
        ax.plot(points[:, 1], points[:, 0], '.r')
        
    if bound is not None: 
        bound = to_np(bound)
        ax.plot(bound[:, 1], bound[:, 0], '.r')
        
    if x_u_ex is not None and t_u_ex is not None and u_ex is not None:
        x_u_ex, t_u_ex, u_ex = to_np(x_u_ex), to_np(t_u_ex), to_np(u_ex)
        ax.plot(t_u_ex, x_u_ex, '.g')
    
    if save_path is not None:
        fig.savefig(save_path)
        
        # save png version as well
        if save_png:
            save_path1 = save_path.replace('.pdf', '.png')
            fig.savefig(save_path1)
        
        plt.close(fig)
    else:
        plt.show()
        
        
def plot_2d_sets(X, T, u_pred, u_pred_min=-1.0, u_pred_max=1.0, X_f_all=None, 
            idx_train=None, idx_fixed=None,
            x_sup=None, t_sup=None, bound=None, bound_all=None, idx=None,
            title='U predicted', save_path=None, cmap='RdBu', save_png=False):
    u_pred = to_np(u_pred)
    fig, ax = plt.subplots(1, figsize=(9, 5))
    color = ax.pcolormesh(T.T, X.T, u_pred.T, cmap=cmap, vmin=u_pred_min, vmax=u_pred_max, alpha=0.9)
    ax.set_title(title)
    # set the limits of the plot to the limits of the data
    ax.axis([T.min(), T.max(), X.min(), X.max(), ])
    fig.colorbar(color, ax=ax)
    
    # Downsample points for plots 
    mask_all = np.ones(X_f_all.shape[0], dtype=bool)
    if idx is None:
        N = 500
        idx = np.random.choice(X_f_all.shape[0], N, replace=False)
    mask_all[idx] = 0
        
    if idx_train is not None:
        idx_train = to_np(idx_train)
        train_mask = np.zeros(X_f_all.shape[0], dtype=bool)
        train_mask[idx_train] = True
    else:
        train_mask = np.zeros(X_f_all.shape[0], dtype=np.int32)
        
    if idx_fixed is not None:
        idx_fixed = to_np(idx_fixed)
        sup_mask = np.zeros(X_f_all.shape[0], dtype=bool)
        sup_mask[idx_fixed] = True
    else:
        sup_mask = np.zeros(X_f_all.shape[0], dtype=np.int32)
        
    # plot initial conditions    
    if x_sup is not None and t_sup is not None:
        # (t, x) pairs
        ax.plot(to_np(t_sup[::10]), to_np(x_sup[::10]), 'ob', markersize=6, clip_on=False)    
        
    if X_f_all is not None:
        # plot all pde points, (t, x) pairs
        X_f_all1 = to_np(X_f_all)[mask_all]
        ax.plot(X_f_all1[::, 1], X_f_all1[::, 0], 'ko', fillstyle='none')
        
        # plot used pde points, (t, x) pairs     
        X_f_train = to_np(X_f_all)[mask_all & train_mask]
        # print(X_f_train)
        ax.plot(X_f_train[:, 1], X_f_train[:, 0], '.k', markersize=10)
        
        # plot pseudo-labels
        X_u_extra = X_f_all[mask_all & sup_mask]
        if np.sum(mask_all & sup_mask) > 0:
            t_u_extra, x_u_extra = X_u_extra[:, 1], X_u_extra[:, 0]
            ax.plot(to_np(t_u_extra), to_np(x_u_extra), 'o', color='deepskyblue', markersize=6)
        
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
