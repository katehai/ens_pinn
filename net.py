import os
import torch
import torch.nn as nn
import numpy as np
import wandb
# from pytorch_minimize.optim import MinimizeWrapper

from choose_optimizer import *
from plot_utils import plot_2d, plot_2d_sets

from solution_nets import SolutionMLP, SolutionMLPMult
from utils import save_ens_info, log_error


# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class PhysicsInformedNN(nn.Module):
    def __init__(self, system, n_f, n_bc, X_u_train, u_train, X_f_train, bc_lb, bc_ub,
                 optimizer_name, lr, dnn, L=1,
                 X_extra=None, u_extra=None, include_extra=True,
                 pde_initial=True, pde_boundary=False, use_two_opt=False, 
                 use_w=True, use_lbfgs_wrapper=False,
                 ):
        super().__init__()
        self.system = system

        # used for supervision loss
        self.x_u = self.np_to_tensor(X_u_train[:, 0:1])
        self.t_u = self.np_to_tensor(X_u_train[:, 1:2])
        self.u = self.np_to_tensor(u_train)

        self.x_f = self.np_to_tensor(X_f_train[:, 0:1])
        self.t_f = self.np_to_tensor(X_f_train[:, 1:2])
        self.x_bc_lb = self.np_to_tensor(bc_lb[:, 0:1])
        self.t_bc_lb = self.np_to_tensor(bc_lb[:, 1:2])
        self.x_bc_ub = self.np_to_tensor(bc_ub[:, 0:1])
        self.t_bc_ub = self.np_to_tensor(bc_ub[:, 1:2])

        self.include_extra = include_extra
        if X_extra is not None and u_extra is not None:
            self.x_u_extra = self.np_to_tensor(X_extra[:, 0:1])
            self.t_u_extra = self.np_to_tensor(X_extra[:, 1:2])
            self.u_extra = self.np_to_tensor(u_extra)
        else:
            self.x_u_extra = None
            self.t_u_extra = None
            self.u_extra = None

        self.pde_initial = pde_initial
        self.pde_boundary = pde_boundary
        print(f"PDE initial is {self.pde_initial}.")
        print(f"PDE boundary is {self.pde_boundary}.")
        self.upd_pde_points() # include initial and boundary conditions if needed

        self.dnn = dnn.to(device)
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.use_two_opt = use_two_opt
        self.use_lbfgs_wrapper = use_lbfgs_wrapper
        self.optimizer = choose_optimizer(optimizer_name, self.dnn.parameters(), self.lr)
        self.optimizer_lbfgs = self.get_lbfgs()
        self.use_w = use_w
        self.iter = 0
        self.L = L

        # max number of points
        self.n_f = n_f
        self.n_pde = self.get_n_pde()
        self.n_sup = len(self.x_u)
        self.n_sup_extra = self.n_f + len(self.u_extra) if self.u_extra is not None else self.n_f
        self.n_bc = n_bc
        print(f"weights are {self.n_sup}, {self.n_pde}, {self.n_bc}")

    def np_to_tensor(self, x):
        x_tensor = torch.tensor(x, requires_grad=True).float().to(device)
        return x_tensor

    def get_lbfgs(self):
        if self.use_lbfgs_wrapper:
            self.dnn.to(torch.double)
            # self = self.double()
            minimizer_args = dict(method='L-BFGS-B', options={'maxiter': 10000,
                                                              'maxfun': 20000,
                                                              'maxcor': 50,
                                                              'maxls': 50,
                                                              'ftol' : 1.0 * np.finfo(float).eps})
            # opt = MinimizeWrapper(self.dnn.parameters(), minimizer_args)
            opt = None
        else:
            opt = torch.optim.LBFGS(self.dnn.parameters(), # lr=self.lr,
                                 max_iter=20000, max_eval=20000,
                                 history_size=50,
                                 line_search_fn='strong_wolfe',
                                 tolerance_change=1.0 * np.finfo(float).eps)
        return opt

    def get_n_pde(self):
        n_pde = self.n_f + len(self.x_u) if self.pde_initial else self.n_f
        n_pde = n_pde + len(self.x_bc_lb) if self.pde_boundary else n_pde
        return n_pde

    def forward(self, X):
        if len(X.shape) < 3:
            x, t = X[:, 0:1], X[:, 1:2]
        else:
            x, t = X[:, :, 0:1], X[:, :, 1:2]

        u = self.net_u(x, t)
        return u

    def net_u(self, x, t, n_nets=None, return_input=False):
        if n_nets is not None:
            # to take into account outputs of each net in PDE loss
            t = t.repeat(n_nets, 1, 1)
            x = x.repeat(n_nets, 1, 1)

        X = torch.cat([x, t], dim=-1)
        u = self.dnn(X)

        if return_input:
            return x, t, u
        else:
            return u

    def net_f(self, x, t, n_nets=None):
        if n_nets is not None:
            # to take into account outputs of each net in PDE loss
            t = t.repeat(n_nets, 1, 1)
            x = x.repeat(n_nets, 1, 1)

        u = self.net_u(x, t)
        f = self.system.pde(u, x, t)
        return f

    def get_du(self, u, x):
        der = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        return der

    def upd_pde_points(self):
        x, t = self.x_f, self.t_f

        if self.pde_initial:
            # always include initial conditions in PDE loss
            x = torch.cat([x, self.x_u], dim=0).detach().requires_grad_(True)
            t = torch.cat([t, self.t_u], dim=0).detach().requires_grad_(True)

        if self.pde_boundary:
            # include boundary conditions in PDE loss
            x = torch.cat([x, self.x_bc_lb, self.x_bc_ub], dim=0).detach().requires_grad_(True)
            t = torch.cat([t, self.t_bc_lb, self.t_bc_ub], dim=0).detach().requires_grad_(True)

        self.x_f, self.t_f = x, t

    def criterion(self, log_wandb=True):
        terms = {}

        # supervision loss
        u_pred = self.net_u(self.x_u, self.t_u)
        err_u = (self.u - u_pred) ** 2
        loss_u = torch.sum(torch.mean(err_u, dim=-2))
        terms['sup'] = loss_u.item()

        # if boundary points are added
        n_nets = u_pred.size(0) if len(u_pred.shape) > 2 else None
        if len(self.x_bc_lb) > 0:
            x_bc_lb, t_bc_lb, u_pred_lb = self.net_u(self.x_bc_lb, self.t_bc_lb, n_nets, return_input=True)
            x_bc_ub, t_bc_ub, u_pred_ub = self.net_u(self.x_bc_ub, self.t_bc_ub, n_nets, return_input=True)
            loss_b = torch.sum(torch.mean((u_pred_lb - u_pred_ub) ** 2, dim=-2))
            terms['bc'] = loss_b.item()

            if self.system.bound_der_loss:
                u_pred_lb_x = self.get_du(u_pred_lb, x_bc_lb)
                u_pred_ub_x = self.get_du(u_pred_ub, x_bc_ub)
                loss_dbc = torch.sum(torch.mean((u_pred_lb_x - u_pred_ub_x) ** 2, dim=-2))
                loss_b = loss_b + loss_dbc
                terms['dbc'] = loss_dbc.item()

            if self.system.zero_bc:
                loss_lb = torch.sum(torch.mean(u_pred_lb ** 2, dim=-2))
                loss_ub = torch.sum(torch.mean(u_pred_ub ** 2, dim=-2))

                loss_b = loss_b + loss_lb + loss_ub
                terms['lb'] = loss_lb.item()
                terms['ub'] = loss_ub.item()
        else:
            loss_b = torch.zeros(1, dtype=torch.float32).to(loss_u.device)
            terms['bc'] = loss_b.item()

        # if PDE points are added
        if len(self.x_f) > 0:
            f_pred = self.net_f(self.x_f, self.t_f, n_nets)
            loss_f = torch.sum(torch.mean(f_pred ** 2, dim=-2))
        else:
            loss_f = torch.zeros(1, dtype=torch.float32).to(loss_u.device)
        terms['f'] = loss_f.item()

        # if pseudo-labels are taken into account
        lambda_sup = len(self.x_u) / self.n_sup if self.use_w else 1.
        if self.u_extra is not None and self.include_extra:
            u_pred_ex = self.net_u(self.x_u_extra, self.t_u_extra)
            loss_u_ex = torch.sum(torch.mean((self.u_extra - u_pred_ex) ** 2, dim=-2))

            lambda_sup_ex = len(self.x_u_extra) / self.n_sup_extra if self.use_w else 1.
            loss_sup = lambda_sup * loss_u + lambda_sup_ex * loss_u_ex

            terms['sup_ex'] = loss_u_ex.item()
            terms['l_sup_ex'] = (lambda_sup_ex * loss_u_ex).item()
        else:
            loss_sup = lambda_sup * loss_u

        lambda_b = len(self.x_bc_lb) / self.n_bc if self.use_w else 1.
        lambda_f = len(self.x_f) / self.n_pde if self.use_w else 1.
        loss = loss_sup * self.L + lambda_b * loss_b + lambda_f * loss_f

        terms['l_sup'] = (lambda_sup * loss_u).item()
        terms['l_bc'] = (lambda_b * loss_b).item()
        terms['l_sup_total'] =  self.L * loss_sup.item()
        terms['l_f'] = (lambda_f * loss_f).item()
        terms['total'] = loss.item()

        if self.iter % 100 and log_wandb:
            wandb.log(terms, step=self.iter, commit=True)

        return loss, terms

    def loss_pinn(self, verbose=True, log_wandb=True):
        if torch.is_grad_enabled():
            self.optimizer_lbfgs.zero_grad()
            self.optimizer.zero_grad()

        loss, terms = self.criterion(log_wandb)

        if loss.requires_grad:
            loss.backward()

        grad_norm = 0
        for p in self.dnn.parameters():
            param_norm = p.grad.detach().data.norm(2)
            grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm ** 0.5

        if verbose:
            if self.iter % 100 == 0:
                print(
                    'epoch %d, gradient: %.5e, loss: %.5e, loss_u: %.5e, loss_b: %.5e, loss_f: %.5e' %
                    (self.iter, grad_norm,
                     terms['total'], terms['sup'],
                     terms['bc'], terms['f'])
                )
                if self.u_extra is not None and self.include_extra:
                    print('loss extra sup: %.5e' % terms['sup_ex'])

        self.iter += 1

        return loss

    def train(self, x_star=None, u_star=None, log_wandb=True):
        self.dnn.train()
        if self.use_two_opt:
            self.optimizer_lbfgs.step(self.loss_pinn)
        else:
            self.optimizer.step(self.loss_pinn)
        self.dnn.eval()
        with torch.no_grad():
            self.log_test_error(x_star, u_star)

    def train_adam(self, n_epochs=50, x_star=None, u_star=None, log_wandb=True):
        self.dnn.train()
        if x_star is not None and u_star is not None:
            x_star = x_star.to(device)
            u_star = u_star.to(device)

        for i in range(n_epochs):
            loss = self.loss_pinn(log_wandb)
            self.optimizer.step()
#             if i % 100:
#                 self.log_test_error(x_star, u_star)

        if self.use_two_opt:
            print("Tune with LBFGS")
            # self = self.double()
            # self.data_to_double()
            self.train(x_star, u_star, log_wandb)
        return

    def log_test_error(self, x_star, u_star, verbose=False, log_wandb=True):
        if u_star is not None and x_star is not None:
            with torch.no_grad():
                u_pred = self.forward(x_star)

                if len(u_pred.shape) > 2:
                    # for ensemble model
                    u_pred = torch.median(u_pred, dim=0)[0]

                error_u_relative = (torch.linalg.norm(u_star - u_pred, 2) / torch.linalg.norm(u_star, 2)).item()
                error_u_abs = torch.mean(torch.abs(u_star - u_pred)).item()

                errors = {'error_rel': error_u_relative,
                          'error_abs': error_u_abs
                         }

                if verbose:
                    print(errors)

                if log_wandb:
                    wandb.log(errors, step=self.iter, commit=True)

    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)

        self.dnn.eval()
        u = self.net_u(x, t)
        u = u.detach().cpu().numpy()
        return u
    
    def data_to_double(self):
        self.dnn = self.dnn.double()
        self.dnn.mlp = self.dnn.mlp.double() 
        self.x_u = self.x_u.double()
        self.t_u = self.t_u.double()
        self.u = self.u.double()

        self.x_f = self.x_f.double()
        self.t_f = self.t_f.double()
        self.x_bc_lb = self.x_bc_lb.double()
        self.t_bc_lb = self.t_bc_lb.double()
        self.x_bc_ub = self.x_bc_ub.double()
        self.t_bc_ub = self.t_bc_ub.double()

        if self.x_u_extra is not None and self.u_extra is not None:
            self.x_u_extra = self.x_u_extra.double()
            self.t_u_extra = self.t_u_extra.double()
            self.u_extra = self.u_extra.double()


class BaselinePINN(PhysicsInformedNN):
    def __init__(self, system, n_f, n_bc, X_u_train, u_train, X_f_train, bc_lb, bc_ub, layers,
                 optimizer_name, lr, L=1, activation='tanh',
                 mapping=None, X_extra=None, u_extra=None,
                 save_path=None, postfix='', pde_initial=True, pde_boundary=False,
                 use_w=True, use_two_opt=False, use_lbfgs_wrapper=False,
                ):
        dnn = SolutionMLP(layers, activation, mapping=mapping).to(device)
        self.model_name = 'vanilla_pinn'

        super().__init__(system, n_f, n_bc, X_u_train, u_train, X_f_train, bc_lb, bc_ub,
                         optimizer_name, lr, dnn, L,
                         X_extra=X_extra, u_extra=u_extra,
                         pde_initial=pde_initial, pde_boundary=pde_boundary,
                         use_w=use_w, use_two_opt=use_two_opt,
                         use_lbfgs_wrapper=use_lbfgs_wrapper
                        )


class PINNEnsemble2D(nn.Module):
    def __init__(self, n_nets, X, T, X_u_train, u_train,
                 X_f_train, bc_lb, bc_ub, norm_x, norm_t, distance=None, pde_tmax=None,
                 X_extra=None, u_extra=None,
                 save_path=None, postfix='', log_every=10, log_model=False,
                 use_lbfgs=False, threshold=0.0004, delta=0.001
                 ):
        super().__init__()
        # in this model we will use distance from the closed supervision point and variance to fix supervision targets
        self.n_nets = n_nets
        self.X = X
        self.T = T

        # all the x,t "test" data
        X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        self.X_star = torch.FloatTensor(X_star).to(device)

        # normalization for distance d calculation
        self.norm = torch.tensor([norm_x, norm_t]).to(device)
        print(f'Normalize inputs for distance by {norm_x}, {norm_t}.')

        # save boundary conditions
        self.bc_lb_all = self.np_to_tensor(bc_lb)
        self.bc_ub_all = self.np_to_tensor(bc_ub)

        self.pde_tmax = pde_tmax
        self.X_f_all = self.np_to_tensor(X_f_train)

        # save IC separately
        self.x_u = self.np_to_tensor(X_u_train[:, 0:1])
        self.t_u = self.np_to_tensor(X_u_train[:, 1:2])
        self.u = self.np_to_tensor(u_train)

        # additional points for supervision
        if X_extra is not None and u_extra is not None:
            self.x_u_extra = self.np_to_tensor(X_extra[:, 0:1])
            self.t_u_extra = self.np_to_tensor(X_extra[:, 1:2])
            self.u_extra = self.np_to_tensor(u_extra)

            X_u = torch.cat([self.x_u, self.t_u], dim=1)
            X_u_extra = torch.cat([self.x_u_extra, self.t_u_extra], dim=1)
            X_sup = torch.cat([X_u, X_u_extra], dim=0)
        else:
            self.x_u_extra = None
            self.t_u_extra = None
            self.u_extra = None
            X_sup = torch.cat([self.x_u, self.t_u], dim=1)

        # fix targets for a set of PDE points
        self.idx_fixed = None
        self.pde_idx_fixed = None
        self.bc_idx_fixed = None

        # # set PDE and boundary points
        self.set_pde_bc_points(X_sup)

        self.threshold = threshold
        self.delta = delta  # 100000 #0.0001 # 0.001
        self.distance = distance
        self.use_lbfgs = use_lbfgs

        # used for plots (to preserve the same scale for all plots)
        self.u_star_min = None
        self.u_star_max = None

        # plot titles
        self.u_pred_title = "U median"
        self.u_metric_title = "U var"
        self.save_path = save_path
        self.model_name = 'ens_base'
        self.postfix = postfix
        self.log_every = log_every  # 1
        self.log_model = log_model

        N = 500
        self.plot_idx = np.random.choice(self.X_f_all.shape[0], N, replace=False)

    def np_to_tensor(self, x):
        x_tensor = torch.tensor(x, requires_grad=True).float().to(device)
        return x_tensor

    @property
    def X_f(self):
        return self.X_f_all[self.pde_idx_fixed]

    @property
    def bc_lb(self):
        return self.bc_lb_all[self.bc_idx_fixed]

    @property
    def bc_ub(self):
        return self.bc_ub_all[self.bc_idx_fixed]

    def plot_mean_var(self, u_pred_mean, u_pred_var, curr_iter):
        bound = torch.cat([self.bc_lb, self.bc_ub], dim=0)
        bound_all = torch.cat([self.bc_lb_all, self.bc_ub_all], dim=0)
        u_pred_mean = u_pred_mean.reshape(*self.X.shape)
        u_pred_var = u_pred_var.reshape(*self.X.shape)
        verbose = self.save_path is None  # print the image by plt.show()
        pred_path = self.get_plot_path(self.u_pred_title, curr_iter)
        # if verbose or pred_path is not None:
        if True: # always log images to wandb
            # to maintain the same scale across plots use min and max of the correct solution
            u_mean_min = self.u_star_min if self.u_star_min is not None else torch.min(u_pred_mean)
            u_mean_max = self.u_star_max if self.u_star_max is not None else torch.max(u_pred_mean)
            print(f'U star min is {self.u_star_min}')

            fig_mean = plot_2d_sets(self.X, self.T, u_pred_mean, u_mean_min, u_mean_max,
                                    X_f_all=self.X_f_all,
                                    idx_train=self.pde_idx_fixed,
                                    idx_fixed=self.idx_fixed,
                                    x_sup=self.x_u, t_sup=self.t_u,
                                    bound=bound, bound_all=bound_all, idx=self.plot_idx,
                                    title=self.u_pred_title, save_path=pred_path,
                                    save_png=self.log_model)
            wandb.log({'u_mean_curr': wandb.Image(fig_mean)})

            # plot log of var
            u_pred_var = torch.log(u_pred_var)

            u_var_min, u_var_max = torch.min(u_pred_var), torch.max(u_pred_var)
            metric_path = self.get_plot_path(self.u_metric_title, curr_iter)

            fig_var = plot_2d_sets(self.X, self.T, u_pred_var, u_var_min, u_var_max,
                                   X_f_all=self.X_f_all,
                                   idx_train=self.pde_idx_fixed,
                                   idx_fixed=self.idx_fixed,
                                   x_sup=self.x_u, t_sup=self.t_u,
                                   bound=bound, bound_all=bound_all, idx=self.plot_idx,
                                   title=self.u_metric_title, save_path=metric_path, cmap='Greys',
                                   save_png=self.log_model)
            wandb.log({'u_var_curr': wandb.Image(fig_var)})

            if self.log_model:
                save_ens_info(self, folder=self.save_path)

    def get_plot_path(self, name, curr_iter):
        name = name.replace(' ', '').lower()
        print("N iter is ", self.net.iter)
        if (self.save_path is None) or ((curr_iter + 1) % self.log_every != 0 and curr_iter != 0):
            # it means that we do not save the plots on drive
            full_path = None
        else:
            plot_name = f"{self.model_name}_{self.net.system.name}{self.postfix}_{name}_ep_{self.net.iter}.pdf"
            full_path = os.path.join(self.save_path, plot_name)

        full_path = None  # do not save plots on disk
        return full_path

    def get_sup_points(self, points, values, use_extra=False):
        if use_extra:
            x_u = self.x_u_extra
            t_u = self.t_u_extra
            u = self.u_extra
        else:
            x_u = self.x_u
            t_u = self.t_u
            u = self.u

        x_u_new = torch.cat([points[:, 0:1], x_u], dim=0).requires_grad_(True)
        t_u_new = torch.cat([points[:, 1:2], t_u], dim=0).requires_grad_(True)
        u_new = torch.cat([values, u], dim=0).requires_grad_(True)

        return x_u_new, t_u_new, u_new

    def upd_extra_points(self, p, v):
        x = [p[:, 0:1]]
        t = [p[:, 1:2]]
        values = [v]

        if (self.x_u_extra is not None) and (self.t_u_extra is not None) and (self.u_extra is not None):
            x.append(self.x_u_extra)
            t.append(self.t_u_extra)
            values.append(self.u_extra)

        x_u_new = torch.cat(x, dim=0).requires_grad_(True)
        t_u_new = torch.cat(t, dim=0).requires_grad_(True)
        u_new = torch.cat(values, dim=0).requires_grad_(True)

        return x_u_new, t_u_new, u_new

    def set_pde_bc_points(self, X_sup):
        self.set_pde_points(X_sup)
        self.set_bc_points(X_sup)
        return

    def set_pde_points(self, X_sup):
        # check F points not yet included in the PDE loss
        pde_mask = self.get_pde_point_mask()
        if torch.sum(pde_mask) != 0:
            pde_idx = self.get_closest_idx(self.X_f_all, pde_mask, X_sup, self.pde_tmax)
            print('New PDE points: ', len(pde_idx))

            # update indices of included PDE points
            self.upd_pde_fixed_idx(pde_idx)

    def set_bc_points(self, X_sup):
        # the same logic with distance and fixed idx as for PDE points
        bc_mask = self.get_bc_point_mask()
        if torch.sum(bc_mask) != 0:
            bc_lb_idx = self.get_closest_idx(self.bc_lb_all, bc_mask, X_sup, self.pde_tmax)
            bc_ub_idx = self.get_closest_idx(self.bc_ub_all, bc_mask, X_sup, self.pde_tmax)

            # merge 2 arrays without duplicates
            bc_idx = torch.unique(torch.cat([bc_lb_idx, bc_ub_idx], dim=0))
            print("New boundary points", len(bc_idx))

            # update bc_idx_fixed
            self.upd_bc_fixed_idx(bc_idx)

    def get_closest_idx(self, all_points, mask, X_sup, d_t):
        points = all_points[mask]
        min_dist = self.get_min_dist(X_sup, points).values
        idx = torch.where(min_dist < d_t)[0]

        # convert from idx in subset to idx the full set
        idx_fix = self.convert_idx(all_points, mask, idx)
        return idx_fix

    def convert_idx(self, all_points, mask, fixed_idx):
        idx_all = torch.arange(len(all_points)).to(fixed_idx.device)
        idx_fix_new = idx_all[mask][fixed_idx]
        return idx_fix_new

    def upd_bc_fixed_idx(self, bc_idx):
        self.bc_idx_fixed = self.upd_fixed(bc_idx, self.bc_idx_fixed)

    def upd_pde_fixed_idx(self, pde_idx):
        self.pde_idx_fixed = self.upd_fixed(pde_idx, self.pde_idx_fixed)

    def upd_fixed_idx(self, mask, sup_idx):
        sup_idx_conv = self.convert_idx(self.X_f_all, mask, sup_idx)
        self.idx_fixed = self.upd_fixed(sup_idx_conv, self.idx_fixed)

    def upd_fixed(self, fixed_idx, saved_idx):
        if saved_idx is None:
            saved_idx = fixed_idx
        else:
            saved_idx = torch.cat([saved_idx, fixed_idx], dim=0)
        return saved_idx

    def get_bc_point_mask(self):
        # filter out idx of points with already fixed targets
        mask = torch.ones(len(self.bc_lb_all), dtype=torch.bool)
        if self.bc_idx_fixed is not None:
            mask[self.bc_idx_fixed] = False
        return mask

    def get_pde_point_mask(self):
        # filter out idx of points with already fixed targets
        mask = torch.ones(len(self.X_f_all), dtype=torch.bool)
        if self.pde_idx_fixed is not None:
            mask[self.pde_idx_fixed] = False
        return mask

    def get_fpoint_mask(self):
        # filter out idx of points with already fixed targets
        mask = torch.ones(len(self.X_f_all), dtype=torch.bool)
        if self.idx_fixed is not None:
            mask[self.idx_fixed] = False
        return mask

    def get_x_sup(self):
        if self.u_extra is not None:
            # concat extra set and main set for supervision
            X_u = torch.cat([self.x_u, self.t_u], dim=-1)
            X_extra = torch.cat([self.x_u_extra, self.t_u_extra], dim=-1)
            X_sup = torch.cat([X_u, X_extra], dim=0)
            u_sup = torch.cat([self.u, self.u_extra], dim=0)
        else:
            X_sup = torch.cat([self.x_u, self.t_u], dim=-1)
            u_sup = self.u
        return X_sup, u_sup

    def get_min_dist(self, X_sup, points):
        X_sup = X_sup / self.norm
        points = points / self.norm

        distances = torch.cdist(X_sup, points, p=2.)
        min_dist = torch.min(distances, dim=0)
        return min_dist

    def get_closest_sup(self, points):
        # get min distance to supervision set
        X_sup, u_sup = self.get_x_sup()

        min_dist = self.get_min_dist(X_sup, points)
        min_dist_v = min_dist.values

        X_sup_closest = X_sup[min_dist.indices]
        u_sup_closest = u_sup[min_dist.indices]
        return X_sup_closest, u_sup_closest, min_dist_v

    def get_select_metric(self, u_preds):
        u_pred_var = torch.var(u_preds, dim=0)
        return u_pred_var

    def select_points(self, u_preds, use_robust=True):
        # calculate mean, variance and add points to supervision loss
        if use_robust:
            u_pred_mean = torch.median(u_preds, dim=0)[0]
        else:
            u_pred_mean = torch.mean(u_preds, dim=0)

        u_pred_var = self.get_select_metric(u_preds)

        mask = self.get_fpoint_mask()
        idx = torch.where(u_pred_var[mask] < self.threshold)[0]
        sup_idx = idx
        x_u = self.X_f_all[mask][idx]
        u = u_pred_mean[mask][idx]

        if (self.distance is not None) and (len(x_u) > 0):
            # select points that are in the neighborhood of existing supervision points
            # select for initial condition supervision set
            d_t = self.distance
            delta = self.delta
            X_sup_closest, u_sup_target, min_dist_v = self.get_closest_sup(x_u)

            # check that supervision loss at the closest supervision set point is small
            u_pred = torch.mean(self.net(X_sup_closest), dim=0)
            loss = ((u_pred - u_sup_target) ** 2).squeeze(-1)

            idx_dist = torch.where((min_dist_v < d_t) & (loss < delta))[0]

            # select only points satisfying distance conditions
            sup_idx = idx[idx_dist]  # mark supervision idx for both sets
            x_u = x_u[idx_dist]
            u = u[idx_dist]

            print("Number of added x_u points ", len(x_u))

        # update indices for fixed targets
        self.upd_fixed_idx(mask, sup_idx)
        return x_u, u, u_pred_mean, u_pred_var


class PINNMultParallel2D(PINNEnsemble2D):
    def __init__(self, system, n_f, n_bc, n_nets, X, T, X_u_train, u_train,
                 X_f_train, bc_lb, bc_ub, norm_x, norm_t, layers,
                 optimizer_name, lr, L=1, activation='tanh',
                 mapping=None, distance=None, pde_tmax=None,
                 X_extra=None, u_extra=None,
                 save_path=None, postfix='', log_every=10,
                 include_extra=True, pde_initial=True, pde_boundary=False,
                 log_model=False, use_lbfgs=False,
                 threshold=0.0004, delta=0.001, use_w=True, upd_targets=False,
                 use_sup_lbfgs=False, use_lbfgs_wrapper=False
                ):
        super().__init__(n_nets, X, T, X_u_train, u_train,
                         X_f_train, bc_lb, bc_ub, norm_x, norm_t, distance, pde_tmax,
                         X_extra=X_extra, u_extra=u_extra, save_path=save_path,
                         postfix=postfix, log_every=log_every, log_model=log_model,
                         use_lbfgs=use_lbfgs, threshold=threshold, delta=delta
                         )
        dnn = SolutionMLPMult(layers, nonlin=activation, n_nets=n_nets,
                              mapping=mapping).to(device)

        # for PINN init
        X_f_train = self.X_f.detach().cpu().numpy()
        bc_lb = self.bc_lb.detach().cpu().numpy()
        bc_ub = self.bc_ub.detach().cpu().numpy()

        self.model_name = 'ens2D'
        self.upd_targets = upd_targets
        self.use_sup_lbfgs = use_sup_lbfgs

        self.net = PhysicsInformedNN(system, n_f, n_bc, X_u_train, u_train, X_f_train,
                                     bc_lb, bc_ub, optimizer_name, lr, dnn, L,
                                     X_extra=X_extra, u_extra=u_extra,
                                     include_extra=include_extra,
                                     pde_initial=pde_initial,
                                     pde_boundary=pde_boundary,
                                     use_two_opt=use_lbfgs,
                                     use_w=use_w,
                                     use_lbfgs_wrapper=use_lbfgs_wrapper,
                                     )

    def update_sup_pde_points(self, upd_optimizer=False):
        if self.u_extra is not None:
            self.net.x_u_extra = self.x_u_extra.detach().requires_grad_(True)
            self.net.t_u_extra = self.t_u_extra.detach().requires_grad_(True)
            self.net.u_extra = self.u_extra.detach().requires_grad_(True)
        else:
            # it is needed to reset points (if extra points were reset)
            self.net.x_u_extra = None
            self.net.t_u_extra = None
            self.net.u_extra = None 

        # update pde points
        self.net.x_f = self.X_f[:, 0:1].detach().requires_grad_(True)
        self.net.t_f = self.X_f[:, 1:2].detach().requires_grad_(True)

        # update boundary points
        self.net.x_bc_lb = self.bc_lb[:, 0:1].detach().requires_grad_(True)
        self.net.t_bc_lb = self.bc_lb[:, 1:2].detach().requires_grad_(True)
        self.net.x_bc_ub = self.bc_ub[:, 0:1].detach().requires_grad_(True)
        self.net.t_bc_ub = self.bc_ub[:, 1:2].detach().requires_grad_(True)

        # updates pde points if initial and/or boundary conditions are included
        self.net.upd_pde_points()

        if upd_optimizer:
            print('Optimizer is updated.')
            self.net.optimizer = choose_optimizer(self.net.optimizer_name, self.net.dnn.parameters(), self.net.lr)

    def reset_sup_points(self, threshold=None):
        self.x_u_extra = None
        self.u_u_extra = None
        self.u_extra = None

        if threshold is not None:
            self.threshold = threshold

    def train(self, n_iter=5, n_epochs=0, n_epochs_1st=None, u_star=None, epochs_max=False, verbose=True):
        # set boundaries for plots
        if u_star is not None:
            self.u_star_min = torch.min(u_star)
            self.u_star_max = torch.max(u_star)

        upd_optimizer = False  # Do not re-create optimizer every time when points are added
        if n_epochs_1st is None:
            n_epochs_1st = n_epochs

        if u_star is not None:
            u_star = u_star.to(self.x_u.device)

        # iteratively train an ensemble
        if n_iter == -1:
            # select the number of training epochs automatically
            # train until all PDE points are not added to PDE set
            # and all boundary points are not added            
            i = 0
            while (self.pde_idx_fixed is None or len(self.pde_idx_fixed) != len(self.X_f_all)) or \
                  (self.bc_idx_fixed is None or len(self.bc_idx_fixed) != len(self.bc_lb_all)):
                self.train_iter(i, n_epochs, n_epochs_1st, u_star, upd_optimizer, verbose)
                i += 1
            
            # then train for 30k iterations more
            n_epochs = 30000
            self.train_iter(i+1, n_epochs, n_epochs_1st, u_star, upd_optimizer, verbose)
        elif epochs_max:
            n_epochs_max = (n_iter - 1) * 1000
            n_iter = n_epochs_max // n_epochs + 2 # add the first epochs too
            n_epochs_last = n_epochs_max - n_epochs * (n_iter - 2) 
            n_epochs_curr = n_epochs
            for i in range(n_iter):
                if i == n_iter - 1:
                    n_epochs_curr = n_epochs_last
                
                self.train_iter(i, n_epochs_curr, n_epochs_1st, u_star, upd_optimizer, verbose)
                
        else:
            for i in range(n_iter):
                self.train_iter(i, n_epochs, n_epochs_1st, u_star, upd_optimizer, verbose)
            
        # tune with LBFGS afterwards    
        if self.use_lbfgs:
            ## marked as lbfgs2 in the experiments
            print("Tune with LBFGS")
            # reset supervision points for pseudo-label case for tuning
            if not self.use_sup_lbfgs:
                self.reset_sup_points()
            
            # add all PDE and boundary points to LBFGS to tune
            self.pde_idx_fixed = torch.arange(len(self.X_f_all))
            self.bc_idx_fixed = torch.arange(len(self.bc_lb_all))            
            self.update_sup_pde_points(upd_optimizer=False)
            
            self.net.optimizer_lbfgs = self.net.get_lbfgs()
            print(f'Number of X_f {self.net.x_f.shape}')
            self.net.train(self.X_star, u_star)
            
            wandb.log({'n_sup': len(self.idx_fixed) if self.idx_fixed is not None else 0,
                       'n_pde': len(self.pde_idx_fixed) if self.pde_idx_fixed is not None else 0,
                       'n_bound': len(self.bc_idx_fixed) if self.bc_idx_fixed is not None else 0
                      }, step=self.net.iter, commit=True)

        return
    
    def train_iter(self, i, n_epochs, n_epochs_1st, u_star, upd_optimizer, verbose):
        print(f'Iteration {i}: ')
        n_ep = n_epochs_1st if i == 0 else n_epochs
        self.net.dnn.train()

        self.net.train_adam(n_ep, self.X_star, u_star)

        # use LBFGS as an additional optimizer
        if self.use_lbfgs:
            print('Run LBFGS')
            # recreate LBFGS optimizer
            if not self.use_sup_lbfgs:
                self.net.x_u_extra = None
                self.net.t_u_extra = None
                self.net.u_extra = None
            
            self.net.optimizer_lbfgs = self.net.get_lbfgs()
            self.net.train(self.X_star, u_star)

        with torch.no_grad():
            u_preds = self.net(self.X_f_all)
            p, v, u_pred_mean, u_pred_var = self.select_points(u_preds)
            
            # update targets for existing points
            if self.upd_targets and self.u_extra is not None:
                self.u_extra = torch.median(self.net.net_u(self.x_u_extra, 
                                                           self.t_u_extra), dim=0)[0]  

            # if any points are added we should update data
            if len(p) > 0:
                self.x_u_extra, self.t_u_extra, self.u_extra = self.upd_extra_points(p, v)
                
                self.set_pde_bc_points(p)
                self.update_sup_pde_points(upd_optimizer)

            if verbose:
                print()
                print(f'Number of sup. points: {len(self.idx_fixed)}.')
                print(f'Number of PDE points: {len(self.pde_idx_fixed)}.')
                print(f'Number of boundary points: {len(self.bc_idx_fixed)}.')
            wandb.log({'n_sup': len(self.idx_fixed) if self.idx_fixed is not None else 0,
                       'n_pde': len(self.pde_idx_fixed) if self.pde_idx_fixed is not None else 0,
                       'n_bound': len(self.bc_idx_fixed) if self.bc_idx_fixed is not None else 0
                       }, step=self.net.iter, commit=True)

            u_preds = self.net(self.X_star)
            u_pred_mean = torch.median(u_preds, dim=0)[0]
            u_pred_var = torch.var(u_preds, dim=0)
            if u_star is not None:
                u_star_n = u_star.repeat(self.n_nets, 1, 1)
                error = torch.mean(torch.abs(u_star_n - u_preds), dim=1).squeeze(-1)
                print(f"Absolute error is {error}.")

                error = torch.mean(torch.abs(u_star - u_pred_mean))
                print(f"Mean absolute error is {error}.")
                log_error(torch.mean(u_preds, dim=0).detach().cpu().numpy(), u_star.detach().cpu().numpy())

            if i % 10 == 0:
                self.plot_mean_var(u_pred_mean, u_pred_var, i)


    def predict(self, X):
        self.net.dnn.eval()
        with torch.no_grad():
            X = torch.tensor(X).float().to(self.x_u.device)
            u_preds = self.net(X)  # predict for the ensemble
            u_preds = torch.mean(u_preds, dim=0)
        return u_preds.detach().cpu().numpy()

    def predict_by_net(self, X):
        self.net.dnn.eval()
        with torch.no_grad():
            if not torch.is_tensor(X):
                X = torch.tensor(X).float().to(self.x_u.device)
            u_preds = self.net(X)  # predict for the ensemble
        return u_preds.detach().cpu().numpy()

    def predict_var(self, X):
        with torch.no_grad():
            X = torch.tensor(X).float().to(self.x_u.device)
            u_preds = self.net(X)  # predict for the ensemble

            u_pred_mean = torch.mean(u_preds, dim=0)
            u_pred_var = torch.var(u_preds, dim=0)
        return u_pred_mean, u_pred_var
