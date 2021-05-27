from ODE_Utils3 import *
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
from functions import *

@logger.catch
def main():

    ############################ hopf norm
    print('\nPSEUDO-ARCLENGTH CONTINUATION: HOPF NORMAL FORM')
    u0 = np.array([.3, .3]) # new initial guess
    print('Init guess: ', u0)

    sys_params = {}         # set system params
    sys_params['b'] = .01    # ^what he said
    print('Params: ', sys_params)

    system = hopf_norm      # declare which system to solve

    plt.title(system.__name__)
    # Then with improve_guess=True passed to the first solution shooting!
    points = pseudo_arc_contin(system, u0, .01, vary_par='b', end_par=2., step=.01, max_steps=200, discretisation=shooting, solver=fsolve, plot=True, print_progress=False, sys_params=sys_params)

    ########################### hopf mod
    print('\n\nPSEUDO-ARCLENGTH CONTINUATION: HOPF MODIFIED')
    u0 = np.array([.3, .3]) # new initial guess
    print('Init guess: ', u0)

    sys_params = {}         # set system params
    sys_params['b'] = 2.    # ^what he said
    print('Params: ', sys_params)

    system = hopf_mod      # declare which system to solve

    plt.title(system.__name__)
    points = pseudo_arc_contin(system, u0, 2., vary_par='b', end_par=-1., step=-.01, max_steps=600, discretisation=shooting, solver=fsolve, plot=True, print_progress=False, sys_params=sys_params)

    # cubic
    print('\n\nPSEUDO-ARCLENGTH CONTINUATION: CUBIC')
    u0 = 1.5 # new initial guess

    sys_params = {}         # set system params
    sys_params['b'] = -2.    # ^what he said
    print('Params: ', sys_params)

    system = cubic      # declare which system to solve

    plt.title(system.__name__)
    points = pseudo_arc_contin(system, u0, -2., vary_par='b', step=.01, end_par=2., max_steps=800, discretisation=param_discretise, solver=fsolve, plot=True, print_progress=False, sys_params=sys_params)

if __name__ == '__main__':
    main()
