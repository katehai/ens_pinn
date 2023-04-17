import torch
import torch.nn as nn
import numpy as np
import wandb

from model.solution_nets import SolutionMLP, SolutionMLPMult
from utils import save_ens_info
from error_utils import log_error


def np_to_tensor(x):
    x_tensor = torch.tensor(x, requires_grad=True).float()
    return x_tensor


class PointCollection:
    def __init__(self, xt_star, u_star, xt_u, u, xt_f, xt_pl, u_pl, n_bc, n_f,
                 pde_boundary, pde_initial, grid_bounds):
        self.grid_bounds = grid_bounds
        # save the full grid
        self.X = xt_star[:, :, 0]
        self.T = xt_star[:, :, 1]

        # The full grid (X, T) is the "test" data with the ground truth solution u_star
        xt_star = xt_star.reshape(-1, 2)
        self.xt_star = np_to_tensor(xt_star)
        self.u_star = np_to_tensor(u_star)

        # used for supervision loss
        self.x_u = np_to_tensor(xt_u[:, 0:1])
        self.t_u = np_to_tensor(xt_u[:, 1:2])
        self.u = np_to_tensor(u)
        self.n_sup = len(self.x_u)

        # PDE loss
        self.n_f = n_f
        self.x_f = np_to_tensor(xt_f[:, 0:1])
        self.t_f = np_to_tensor(xt_f[:, 1:2])
        self.xt_f_all = np_to_tensor(xt_f)

        # boundary conditions
        self.n_bc = n_bc
        bc_lb, bc_ub = self.get_bc()
        self.bc_lb_all = np_to_tensor(bc_lb)
        self.bc_ub_all = np_to_tensor(bc_ub)
        self.x_bc_lb = np_to_tensor(bc_lb[:, 0:1])
        self.t_bc_lb = np_to_tensor(bc_lb[:, 1:2])
        self.x_bc_ub = np_to_tensor(bc_ub[:, 0:1])
        self.t_bc_ub = np_to_tensor(bc_ub[:, 1:2])

        # Pseudo-labels
        if xt_pl is not None and u_pl is not None:
            self.x_u_pl = np_to_tensor(xt_pl[:, 0:1])
            self.t_u_pl = np_to_tensor(xt_pl[:, 1:2])
            self.u_pl = np_to_tensor(u_pl)
        else:
            self.x_u_pl = None
            self.t_u_pl = None
            self.u_pl = None

        self.pde_initial = pde_initial
        self.pde_boundary = pde_boundary
        print(f"PDE initial is {self.pde_initial}.")
        print(f"PDE boundary is {self.pde_boundary}.")
        self.upd_pde_points()  # include initial and boundary conditions if needed

        self.n_pde = self.get_n_pde()
        self.n_sup_pl = self.n_f + len(self.u_pl) if self.u_pl is not None else self.n_f
        print(f"weights are {self.n_sup}, {self.n_pde}, {self.n_bc}")

    def get_bc(self):
        x0_grid, x1_grid, t0_grid, t1_grid = self.grid_bounds
        t = np.linspace(t0_grid, t1_grid, self.n_bc)[:, None]
        x_lb = x0_grid * np.ones_like(t)
        x_ub = x1_grid * np.ones_like(t)

        lb = np.hstack((x_lb, t))
        ub = np.hstack((x_ub, t))
        return lb, ub

    def get_n_pde(self):
        n_pde = self.n_f + self.n_sup if self.pde_initial else self.n_f
        n_pde = n_pde + self.n_bc if self.pde_boundary else n_pde
        return n_pde

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

    def to_device(self, device):
        self.xt_star = self.xt_star.to(device)
        self.u_star = self.u_star.to(device)

        self.x_u = self.x_u.to(device)
        self.t_u = self.t_u.to(device)
        self.u = self.u.to(device)

        self.x_f = self.x_f.to(device)
        self.t_f = self.t_f.to(device)
        self.x_bc_lb = self.x_bc_lb.to(device)
        self.t_bc_lb = self.t_bc_lb.to(device)
        self.x_bc_ub = self.x_bc_ub.to(device)
        self.t_bc_ub = self.t_bc_ub.to(device)

        self.xt_f_all = self.xt_f_all.to(device)
        self.bc_lb_all = self.bc_lb_all.to(device)
        self.bc_ub_all = self.bc_ub_all.to(device)

        self.x_u_pl = self.x_u_pl.to(device) if self.x_u_pl is not None else None
        self.t_u_pl = self.t_u_pl.to(device) if self.t_u_pl is not None else None
        self.u_pl = self.u_pl.to(device) if self.u_pl is not None else None


class DynamicPointCollection(PointCollection):
    def __init__(self, xt_star, u_star, xt_u, u, xt_f, xt_pl, u_pl, n_bc, n_f,
                 pde_boundary, pde_initial, grid_bounds, pde_tmax):
        _, x1_grid, _, t1_grid = grid_bounds
        norm_x, norm_t = x1_grid, t1_grid
        self.norm = torch.tensor([norm_x, norm_t])  # normalization for distance d calculation
        self.pde_tmax = pde_tmax

        # fix targets for a set of PDE points
        self.idx_fixed = None
        self.pde_idx_fixed = None
        self.bc_idx_fixed = None

        super().__init__(xt_star, u_star, xt_u, u, xt_f, xt_pl, u_pl, n_bc, n_f,
                         pde_boundary, pde_initial, grid_bounds)
        # get all points for supervision
        xt_sup, _ = self.get_x_sup()

        # set PDE and boundary points
        self.set_pde_bc_masks(xt_sup)
        self.upd_pde_bound_points()

    @property
    def xt_f(self):
        return self.xt_f_all[self.pde_idx_fixed]

    @property
    def bc_lb(self):
        return self.bc_lb_all[self.bc_idx_fixed]

    @property
    def bc_ub(self):
        return self.bc_ub_all[self.bc_idx_fixed]

    def get_x_sup(self):
        if self.u_pl is not None:
            # concat pl set and main set for supervision
            xt_u = torch.cat([self.x_u, self.t_u], dim=-1)
            xt_pl = torch.cat([self.x_u_pl, self.t_u_pl], dim=-1)
            xt = torch.cat([xt_u, xt_pl], dim=0)
            u_sup = torch.cat([self.u, self.u_pl], dim=0)
        else:
            xt = torch.cat([self.x_u, self.t_u], dim=-1)
            u_sup = self.u
        return xt, u_sup

    def set_pde_bc_masks(self, xt_sup):
        # the method update masks for PDE and boundary points
        self.set_pde_point_mask(xt_sup)
        self.set_bc_point_mask(xt_sup)
        return

    def set_pde_point_mask(self, xt_sup):
        # check F points not yet included in the PDE loss
        pde_mask = self.get_pde_point_mask()
        if torch.sum(pde_mask) != 0:
            pde_idx = self.get_closest_idx(self.xt_f_all, pde_mask, xt_sup, self.pde_tmax)
            print('New PDE points: ', len(pde_idx))

            # update indices of included PDE points
            self.upd_pde_fixed_idx(pde_idx)

    def set_bc_point_mask(self, xt_sup):
        # the same logic with distance and fixed idx as for PDE points
        bc_mask = self.get_bc_point_mask()
        if torch.sum(bc_mask) != 0:
            bc_lb_idx = self.get_closest_idx(self.bc_lb_all, bc_mask, xt_sup, self.pde_tmax)
            bc_ub_idx = self.get_closest_idx(self.bc_ub_all, bc_mask, xt_sup, self.pde_tmax)

            # merge 2 arrays without duplicates
            bc_idx = torch.unique(torch.cat([bc_lb_idx, bc_ub_idx], dim=0))
            print("New boundary points", len(bc_idx))

            # update bc_idx_fixed
            self.upd_bc_fixed_idx(bc_idx)

    def get_bc_point_mask(self):
        # filter out idx of points with already fixed targets
        mask = torch.ones(len(self.bc_lb_all), dtype=torch.bool)
        if self.bc_idx_fixed is not None:
            mask[self.bc_idx_fixed] = False
        return mask

    def get_pde_point_mask(self):
        # filter out idx of points with already fixed targets
        mask = torch.ones(len(self.xt_f_all), dtype=torch.bool)
        if self.pde_idx_fixed is not None:
            mask[self.pde_idx_fixed] = False
        return mask

    def get_fpoint_mask(self):
        # filter out idx of points with already fixed targets
        mask = torch.ones(len(self.xt_f_all), dtype=torch.bool)
        if self.idx_fixed is not None:
            mask[self.idx_fixed] = False
        return mask

    def get_closest_idx(self, all_points, mask, xt_sup, d_t):
        points = all_points[mask]
        min_dist = self.get_min_dist(xt_sup, points).values
        idx = torch.where(min_dist < d_t)[0]

        # convert from idx in subset to idx the full set
        idx_fix = self.convert_idx(all_points, mask, idx)
        return idx_fix

    def get_min_dist(self, xt_sup, points):
        xt_sup = xt_sup / self.norm
        points = points / self.norm

        distances = torch.cdist(xt_sup, points, p=2.)
        min_dist = torch.min(distances, dim=0)
        return min_dist

    @staticmethod
    def convert_idx(all_points, mask, fixed_idx):
        idx_all = torch.arange(len(all_points)).to(fixed_idx.device)
        idx_fix_new = idx_all[mask][fixed_idx]
        return idx_fix_new

    def upd_bc_fixed_idx(self, bc_idx):
        self.bc_idx_fixed = self.upd_fixed(bc_idx, self.bc_idx_fixed)

    def upd_pde_fixed_idx(self, pde_idx):
        self.pde_idx_fixed = self.upd_fixed(pde_idx, self.pde_idx_fixed)

    def upd_fixed_idx(self, mask, sup_idx):
        sup_idx_conv = self.convert_idx(self.xt_f_all, mask, sup_idx)
        self.idx_fixed = self.upd_fixed(sup_idx_conv, self.idx_fixed)

    @staticmethod
    def upd_fixed(fixed_idx, saved_idx):
        if saved_idx is None:
            saved_idx = fixed_idx
        else:
            saved_idx = torch.cat([saved_idx, fixed_idx], dim=0)
        return saved_idx

    def get_closest_sup(self, points):
        # get min distance to supervision set
        xt_sup, u_sup = self.get_x_sup()

        min_dist = self.get_min_dist(xt_sup, points)
        min_dist_v = min_dist.values

        xt_sup_closest = xt_sup[min_dist.indices]
        u_sup_closest = u_sup[min_dist.indices]
        return xt_sup_closest, u_sup_closest, min_dist_v

    def upd_pl_points(self, p, v):
        x = [p[:, 0:1]]
        t = [p[:, 1:2]]
        values = [v]

        if (self.x_u_pl is not None) and (self.t_u_pl is not None) and (self.u_pl is not None):
            x.append(self.x_u_pl)
            t.append(self.t_u_pl)
            values.append(self.u_pl)

        self.x_u_pl = torch.cat(x, dim=0).detach().requires_grad_(True)
        self.t_u_pl = torch.cat(t, dim=0).detach().requires_grad_(True)
        self.u_pl = torch.cat(values, dim=0).detach().requires_grad_(True)

    def upd_pde_bound_points(self):
        # update pde points
        self.x_f = self.xt_f[:, 0:1].detach().requires_grad_(True)
        self.t_f = self.xt_f[:, 1:2].detach().requires_grad_(True)

        # update boundary points
        self.x_bc_lb = self.bc_lb[:, 0:1].detach().requires_grad_(True)
        self.t_bc_lb = self.bc_lb[:, 1:2].detach().requires_grad_(True)
        self.x_bc_ub = self.bc_ub[:, 0:1].detach().requires_grad_(True)
        self.t_bc_ub = self.bc_ub[:, 1:2].detach().requires_grad_(True)

        # updates pde points if initial and/or boundary conditions are included
        self.upd_pde_points()

    def to_device(self, device):
        super().to_device(device)
        self.norm = self.norm.to(device)
        self.idx_fixed = self.idx_fixed.to(device) if self.idx_fixed is not None else None
        self.pde_idx_fixed = self.pde_idx_fixed.to(device) if self.pde_idx_fixed is not None else None
        self.bc_idx_fixed = self.bc_idx_fixed.to(device) if self.bc_idx_fixed is not None else None

    def get_counts(self):
        counts = {
            'n_sup': len(self.idx_fixed) if self.idx_fixed is not None else 0,
            'n_pde': len(self.pde_idx_fixed) if self.pde_idx_fixed is not None else 0,
            'n_bound': len(self.bc_idx_fixed) if self.bc_idx_fixed is not None else 0
        }
        return counts


class PhysicsInformedNN(nn.Module):
    def __init__(self, system, pc, lr, m, dnn, include_pl=True, use_two_opt=False, use_w=True):
        super().__init__()
        self.system = system
        self.include_pl = include_pl
        self.pc = pc

        self.dnn = dnn
        self.lr = lr
        self.use_two_opt = use_two_opt
        self.optimizer = torch.optim.Adam(self.dnn.parameters(), self.lr)
        self.optimizer_lbfgs = self.get_lbfgs()
        self.use_w = use_w
        self.m = m
        self.iter = 0

    def get_lbfgs(self):
        opt = torch.optim.LBFGS(self.dnn.parameters(),
                                max_iter=20000, max_eval=20000,
                                history_size=50,
                                line_search_fn='strong_wolfe',
                                tolerance_change=1.0 * np.finfo(float).eps)
        return opt

    def forward(self, xt):
        if len(xt.shape) < 3:
            x, t = xt[:, 0:1], xt[:, 1:2]
        else:
            x, t = xt[:, :, 0:1], xt[:, :, 1:2]

        u = self.net_u(x, t)
        return u

    def net_u(self, x, t, n_nets=None, return_input=False):
        if n_nets is not None:
            # to take into account outputs of each net in PDE loss
            t = t.repeat(n_nets, 1, 1)
            x = x.repeat(n_nets, 1, 1)

        xt = torch.cat([x, t], dim=-1)
        u = self.dnn(xt)

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

    @staticmethod
    def get_du(u, x):
        der = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        return der

    @property
    def lambda_sup(self):
        return len(self.pc.x_u) / self.pc.n_sup if self.use_w else 1.

    @property
    def lambda_sup_pl(self):
        return len(self.pc.x_u_pl) / self.pc.n_sup_pl if self.use_w else 1.

    @property
    def lambda_b(self):
        return len(self.pc.x_bc_lb) / self.pc.n_bc if self.use_w else 1.

    @property
    def lambda_f(self):
        return len(self.pc.x_f) / self.pc.n_pde if self.use_w else 1.

    def criterion(self, log_wandb=True):
        terms = {}

        loss_sup, n_nets = self.get_sup_loss(terms)
        loss_b = self.get_bc_loss(n_nets, terms)
        loss_f = self.get_f_loss(n_nets, terms)

        loss = self.m * loss_sup + self.lambda_b * loss_b + self.lambda_f * loss_f

        terms['l_bc'] = (self.lambda_b * loss_b).item()
        terms['l_sup_total'] = self.m * loss_sup.item()
        terms['l_f'] = (self.lambda_f * loss_f).item()
        terms['total'] = loss.item()

        if self.iter % 100 and log_wandb:
            wandb.log(terms, step=self.iter, commit=True)

        return loss, terms

    def get_sup_loss(self, terms):
        # supervision loss
        u_pred = self.net_u(self.pc.x_u, self.pc.t_u)
        err_u = (self.pc.u - u_pred) ** 2
        loss_u = torch.sum(torch.mean(err_u, dim=-2))
        terms['sup'] = loss_u.item()

        # if pseudo-labels are taken into account
        lambda_sup = self.lambda_sup
        if self.pc.u_pl is not None and self.include_pl:
            u_pred_pl = self.net_u(self.pc.x_u_pl, self.pc.t_u_pl)
            loss_u_pl = torch.sum(torch.mean((self.pc.u_pl - u_pred_pl) ** 2, dim=-2))

            loss_sup_pl = self.lambda_sup_pl * loss_u_pl
            loss_sup = lambda_sup * loss_u + loss_sup_pl

            terms['sup_pl'] = loss_u_pl.item()
            terms['l_sup_pl'] = loss_sup_pl.item()
        else:
            loss_sup = lambda_sup * loss_u

        terms['l_sup'] = (lambda_sup * loss_u).item()
        n_nets = u_pred.size(0) if len(u_pred.shape) > 2 else None
        return loss_sup, n_nets

    def get_f_loss(self, n_nets, terms):
        # if PDE points are added
        if len(self.pc.x_f) > 0:
            f_pred = self.net_f(self.pc.x_f, self.pc.t_f, n_nets)
            loss_f = torch.sum(torch.mean(f_pred ** 2, dim=-2))
        else:
            loss_f = torch.zeros(1, dtype=torch.float32).to(self.pc.x_u.device)
        terms['f'] = loss_f.item()
        return loss_f

    def get_bc_loss(self, n_nets, terms):
        # if boundary points are added
        if len(self.pc.x_bc_lb) > 0:
            x_bc_lb, t_bc_lb, u_pred_lb = self.net_u(self.pc.x_bc_lb, self.pc.t_bc_lb, n_nets, return_input=True)
            x_bc_ub, t_bc_ub, u_pred_ub = self.net_u(self.pc.x_bc_ub, self.pc.t_bc_ub, n_nets, return_input=True)
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
            loss_b = torch.zeros(1, dtype=torch.float32).to(self.pc.x_u.device)
            terms['bc'] = loss_b.item()
        return loss_b

    def loss_pinn(self, verbose=True, log_wandb=True):
        if torch.is_grad_enabled():
            self.optimizer_lbfgs.zero_grad()
            self.optimizer.zero_grad()

        loss, terms = self.criterion(log_wandb)
        if loss.requires_grad:
            loss.backward()

        if verbose:
            if self.iter % 100 == 0:
                print(f"epoch {self.iter}, loss: {terms['total']:.5e}, "
                      f"loss_u: {terms['sup']:.5e}, loss_b: {terms['bc']:.5e}, "
                      f"loss_f: {terms['f']:.5e}")

                if self.pc.u_pl is not None and self.include_pl:
                    print(f"loss PL sup: {terms['sup_pl']:.5e}")

        self.iter += 1

        return loss

    def train_lbfgs(self, verbose=False, log_wandb=True, log_test=True):
        self.dnn.train()
        self.optimizer_lbfgs = self.get_lbfgs()
        self.optimizer_lbfgs.step(self.loss_pinn)

        if log_test:
            self.dnn.eval()
            with torch.no_grad():
                self.log_test_error(verbose=verbose, log_wandb=log_wandb)

    def train_adam(self, n_epochs=50, verbose=False, log_wandb=True, log_test=True):
        for i in range(n_epochs):
            self.dnn.train()

            self.loss_pinn(log_wandb)
            self.optimizer.step()

            if i % 1000 == 0 and log_test:
                self.dnn.eval()
                with torch.no_grad():
                    self.log_test_error(verbose=verbose, log_wandb=log_wandb)
        return

    def train(self, n_epochs=50, verbose=False, log_wandb=True):
        self.train_adam(n_epochs, log_wandb)

        if self.use_two_opt:
            print("Tune with LBFGS base")
            self.train_lbfgs(log_wandb)
        return

    def log_test_error(self, verbose=False, log_wandb=True):
        xt_star = self.pc.xt_star
        u_star = self.pc.u_star
        with torch.no_grad():
            u_pred = self.forward(xt_star)

            if len(u_pred.shape) > 2:
                # for ensemble model
                u_pred = torch.median(u_pred, dim=0)[0]

            log_error(u_pred, u_star, log_wandb=log_wandb, verbose=verbose)

    def predict(self, xt):
        x = torch.tensor(xt[:, 0:1], requires_grad=True).float().to(self.pc.x_u.device)
        t = torch.tensor(xt[:, 1:2], requires_grad=True).float().to(self.pc.x_u.device)

        self.dnn.eval()
        u = self.net_u(x, t)
        u = u.detach().cpu().numpy()
        return u

    def to_device(self, device):
        self.dnn.to(device)
        self.pc.to_device(device)


class BaselinePINN(PhysicsInformedNN):
    def __init__(self, system, xt_star, u_star, xt_u, u, xt_f, n_bc, n_f, layers, lr, activation='tanh',
                 m=1., mapping=None, xt_pl=None, u_pl=None, pde_initial=True, pde_boundary=False, use_w=True,
                 use_two_opt=False, grid_bounds=(0, 1, 0, 1)):
        self.model_name = 'vanilla_pinn'
        dnn = SolutionMLP(layers, activation, mapping=mapping)

        pc = PointCollection(xt_star, u_star, xt_u, u, xt_f, xt_pl, u_pl, n_bc, n_f,
                             pde_boundary, pde_initial, grid_bounds)
        super().__init__(system, pc, lr, m, dnn, use_two_opt=use_two_opt, use_w=use_w)


class PINNMultParallel2D(nn.Module):
    def __init__(self, system, n_nets, xt_star, u_star, xt_u, u, xt_f, n_bc, n_f, layers, lr,
                 activation='tanh', m=1., mapping=None, xt_pl=None, u_pl=None, pde_initial=True, pde_boundary=False,
                 use_w=True, use_two_opt=False, grid_bounds=None, distance=None, pde_tmax=None, include_pl=True,
                 threshold=0.0004, delta=0.001, save_path=None, log_model=False, log_every=10):
        super().__init__()

        pc = DynamicPointCollection(xt_star, u_star, xt_u, u, xt_f, xt_pl, u_pl, n_bc, n_f,
                                    pde_boundary, pde_initial, grid_bounds, pde_tmax)

        self.n_nets = n_nets
        dnn = SolutionMLPMult(layers, activation=activation, n_nets=n_nets, mapping=mapping)

        self.model_name = 'ens2D'

        # in this model we will use distance from the closed supervision point and variance to fix supervision targets
        self.threshold = threshold
        self.delta = delta
        self.distance = distance
        self.use_two_opt = use_two_opt

        self.log_model = log_model
        self.log_every = log_every
        self.save_path = save_path

        self.net = PhysicsInformedNN(system, pc, lr, m, dnn, include_pl, use_two_opt, use_w)

    def select_points(self, u_pred, use_robust=True):
        # calculate mean, variance and add points to supervision loss
        if use_robust:
            u_pred_mean = torch.median(u_pred, dim=0)[0]
        else:
            u_pred_mean = torch.mean(u_pred, dim=0)

        u_pred_var = torch.var(u_pred, dim=0)

        mask = self.net.pc.get_fpoint_mask()
        idx = torch.where(u_pred_var[mask] < self.threshold)[0]
        sup_idx = idx
        x_u = self.net.pc.xt_f_all[mask][idx]
        u = u_pred_mean[mask][idx]

        if (self.distance is not None) and (len(x_u) > 0):
            # select points that are in the neighborhood of existing supervision points
            # select for initial condition supervision set
            d_t = self.distance
            delta = self.delta
            xt_sup_closest, u_sup_target, min_dist_v = self.net.pc.get_closest_sup(x_u)

            # check that supervision loss at the closest supervision set point is small
            u_pred = torch.mean(self.net(xt_sup_closest), dim=0)
            loss = ((u_pred - u_sup_target) ** 2).squeeze(-1)

            idx_dist = torch.where((min_dist_v < d_t) & (loss < delta))[0]

            # select only points satisfying distance conditions
            sup_idx = idx[idx_dist]  # mark supervision idx for both sets
            x_u = x_u[idx_dist]
            u = u[idx_dist]

            print("Number of added x_u points ", len(x_u))

        # update indices for fixed targets
        self.net.pc.upd_fixed_idx(mask, sup_idx)
        return x_u, u

    def train(self, n_iter=5, n_epochs=0, n_epochs_1st=0, verbose=True, plotting=None):
        # iteratively train the ensemble
        for i in range(n_iter):
            print(f'Iteration {i}: ')
            n_ep = n_epochs_1st if i == 0 else n_epochs
            u_pred_mean, u_pred_var = self.train_iter(n_ep, verbose)

            if (i + 1) % self.log_every == 0 or i == 0:
                plotting.plot_mean_var(self, u_pred_mean, u_pred_var)
                if self.log_model:
                    save_ens_info(self, folder=self.save_path)

        if self.use_two_opt:
            print("Tune with LBFGS final")
            self.tune_lbfgs()

            p_counts = self.net.pc.get_counts()
            wandb.log(p_counts, step=self.net.iter, commit=True)

        return

    def train_iter(self, n_ep, verbose):
        self.net.dnn.train()
        self.net.train_adam(n_ep, verbose=verbose, log_test=False)

        # use LBFGS as an additional optimizer
        if self.use_two_opt:
            print('Run LBFGS')
            self.net.train_lbfgs(verbose=verbose, log_test=False)

        with torch.no_grad():
            u_preds = self.net(self.net.pc.xt_f_all)
            p, v = self.select_points(u_preds)

            # if any points are added we should update point collection
            if len(p) > 0:
                self.net.pc.upd_pl_points(p, v)
                self.net.pc.set_pde_bc_masks(p)
                self.net.pc.upd_pde_bound_points()

            p_counts = self.net.pc.get_counts()
            wandb.log(p_counts, step=self.net.iter, commit=True)

            if verbose:
                print()
                print(f"Number of sup. points: {p_counts['n_sup']}.")
                print(f"Number of PDE points: {p_counts['n_pde']}.")
                print(f"Number of boundary points: {p_counts['n_bound']}.")

        u_pred_mean, u_pred_var = self.log_test_error(verbose=verbose)
        return u_pred_mean, u_pred_var

    def tune_lbfgs(self):
        # add all PDE and boundary points to LBFGS to tune
        self.net.pc.pde_idx_fixed = torch.arange(len(self.net.pc.xt_f_all)).to(self.net.pc.xt_f_all.device)
        self.net.pc.bc_idx_fixed = torch.arange(len(self.net.pc.bc_lb_all)).to(self.net.pc.xt_f_all.device)
        self.net.pc.upd_pde_bound_points()

        self.net.train_lbfgs(log_test=False)

    def log_test_error(self, verbose=False, log_wandb=True):
        with torch.no_grad():
            u_pred = self.net(self.net.pc.xt_star)
            u_pred_mean = torch.mean(u_pred, dim=0)
            u_pred_var = torch.var(u_pred, dim=0)

            log_error(u_pred_mean, self.net.pc.u_star, log_wandb=log_wandb, verbose=verbose)

        return u_pred_mean, u_pred_var

    def predict(self, xt):
        self.net.dnn.eval()
        with torch.no_grad():
            xt = torch.tensor(xt).float().to(self.net.pc.x_u.device)
            u_pred = self.net(xt)  # predict for the ensemble
            u_pred = torch.mean(u_pred, dim=0)
        return u_pred.detach().cpu().numpy()

    def to_device(self, device):
        self.net.to_device(device)
