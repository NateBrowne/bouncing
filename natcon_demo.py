from ODE_Utils3 import *
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
from functions import *

@logger.catch
def main():

    # hopf norm
    print('\nNATURAL PARAM CONTINUATION: HOPF NORMAL FORM - improve_guess=True')
    u0 = np.array([.05, .05]) # new initial guess
    print('Init guess: ', u0)

    sys_params = {}         # set system params
    sys_params['b'] = .01    # ^what he said
    print('Params: ', sys_params)

    system = hopf_norm      # declare which system to solve

    # Then with improve_guess=True passed to the first solution shooting!
    plt.title(system.__name__)
    par_vals, ics = continuation(system, u0, .01, vary_par='b', end_par = 2., step=.01, max_steps=200, discretisation=shooting, solver=fsolve, plot=True, print_progress=False, improve_guess=True, sys_params=sys_params)

    # hopf mod
    print('\n\nNATURAL PARAM CONTINUATION: HOPF MODIFIED')
    u0 = np.array([.3, .3]) # new initial guess
    print('Init guess: ', u0)

    sys_params = {}         # set system params
    sys_params['b'] = 2.    # ^what he said
    print('Params: ', sys_params)

    system = hopf_mod      # declare which system to solve

    plt.title(system.__name__)
    par_vals, ics = continuation(system, u0, 2., vary_par='b', end_par=-1., step=-.01, max_steps=300, discretisation=shooting, solver=fsolve, plot=True, print_progress=False, sys_params=sys_params)
    # key observation: stops at b=-0.29

    # cubic
    print('\n\nNATURAL PARAM CONTINUATION: CUBIC')
    u0 = 1.5 # new initial guess

    sys_params = {}         # set system params
    sys_params['b'] = -2.    # ^what he said
    print('Params: ', sys_params)

    system = cubic      # declare which system to solve

    plt.title(system.__name__)
    par_vals, ics = continuation(system, u0, -2., vary_par='b', step=.01, end_par=2., max_steps=400, discretisation=param_discretise, solver=fsolve, plot=True, print_progress=False, sys_params=sys_params)
    # key observation - will stop at .4 - when the root disappears

if __name__ == '__main__':
    main()
