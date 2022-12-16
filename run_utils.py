import numpy as np


def check_conv_param(system, line_args, x_star):
    if line_args.sys == "convection":
        scale = system.scale
        x_max = np.max(x_star)
        correct_scale = 2*np.pi if np.abs(x_max - 1.) < 0.05 else 1.
        
        if np.abs(scale - correct_scale) > 0.001:
            print("Scale for convection is not correct")
            system.set_scale(correct_scale)
            
    return system