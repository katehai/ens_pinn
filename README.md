These files contain implementation of the proposed PINN ensemble algorithm. 

1. Data: the data used in our experiments are generated from the corresponding PDEs (pde.py file for the implementation) and saved to files available in data folder.
2. Main logic of the proposed algorithm is implemented in net.py file (the classes PINNMultParallel2D and PINNEnsemble2D). The implementation for the baseline PINN is adopted from one of our closest time-adaptive baselines by Krishnapriyan et al.  (https://github.com/a1k12/characterizing-pinns-failure-modes)
3. The method can be run with a followinf commands:
PINN Ensemble:  python run_job.py --sys reaction --model_type ens --postfix 5 --exclude_extra --no_wandb
Pseudo-labels version: python run_job.py --sys reaction --model_type ens --postfix 5 --no_wandb
Tuning with LBFGS optimizer is enabled by adding '--use_lbfgs' to the commands.
