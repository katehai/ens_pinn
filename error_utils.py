import numpy as np
import wandb

from utils import save_results, to_np


def calc_error(u_pred, u_sol):
    u_pred = to_np(u_pred)
    u_sol = to_np(u_sol)

    error_u_rel = np.linalg.norm(u_pred - u_sol) / np.linalg.norm(u_sol)
    error_u_abs = np.mean(np.abs(u_pred - u_sol))
    error_u_linf = np.linalg.norm(u_sol - u_pred, np.inf) / np.linalg.norm(u_sol, np.inf)
    errors = {'err_rel': error_u_rel,
              'err_abs': error_u_abs,
              'err_linf': error_u_linf
              }

    return errors


def log_error(u_pred, u_sol, system=None, path=None, log_wandb=True, verbose=False):
    errors = calc_error(u_pred, u_sol)
    if log_wandb:
        wandb.log(errors)

    if verbose:
        print(f"Error u rel: {errors['err_rel']:.3e}")
        print(f"Error u abs: {errors['err_abs']:.3e}")
        print(f"Error u linf: {errors['err_linf']:.3e}")

    # save results
    if path is not None and system is not None:
        errors_value = errors.values()
        save_results(errors_value, path, system)
    return
