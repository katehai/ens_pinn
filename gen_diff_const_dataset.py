import numpy as np
from pde import *


def generate_convection(betas, path='data'):
    x0, xN, num_x = 0.0, 1, 256  # should use scale in the Convection eq. if this is used
    t0, tN, num_t = 0.0, 1.005, 202  # implement one extra time step in order to get finite differences later on

    # t = np.linspace(t0, 1.005, 202).reshape(-1, 1)
    # print(t)

    for beta in betas:
        pde = Convection(beta=beta)
        pde.save_solution(num_x, num_t, x0, t0, xN, tN, path=path)


if __name__ == "__main__":
    print("Generate train")
    save_path = 'pde_const_data/train'
    betas_train = np.arange(1, 51, 0.05)
    print(len(betas_train))
    print(betas_train[:20])
    generate_convection(betas_train, save_path)

    print()
    print("Generate test")
    save_path_test = 'pde_const_data/test'
    betas_test = np.arange(1.025, 51.025, 0.05)
    print(len(betas_test))
    print(betas_test[:20])
    generate_convection(betas_test, save_path_test)

    # debugging set up
    # save_path_test = 'pde_const_data/test1'
    # betas_test = [20]
    # generate_convection(betas_test, save_path_test)
