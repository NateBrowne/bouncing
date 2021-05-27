from ODE_Utils3 import *
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
from functions import *

@logger.catch
def main():

    ################    PSEUDO CONTINUATION    ###########################
    # # cubic
    # u0 = 1.5 # new initial guess
    #
    # sys_params = {}         # set system params
    # sys_params['b'] = -2.    # ^what he said
    #
    # system = cubic      # declare which system to solve
    #
    # points = pseudo_arc_contin(system, u0, -2., vary_par='b', step=.01, end_par=2., max_steps=800, discretisation=param_discretise, solver=fsolve, plot=True, print_progress=False, sys_params=sys_params)


    # # hopf norm
    u0 = np.array([.3, .3]) # new initial guess

    sys_params = {}         # set system params
    sys_params['b'] = .01    # ^what he said

    system = hopf_norm      # declare which system to solve

    points = pseudo_arc_contin(system, u0, .01, vary_par='b', end_par=2., step=.01, max_steps=200, discretisation=shooting, solver=fsolve, plot=True, print_progress=False, sys_params=sys_params)
    # key observation: no different to nat param contin

    # # hopf mod
    # u0 = np.array([.3, .3]) # new initial guess
    # print('Our initial guess: ', u0)
    #
    # sys_params = {}         # set system params
    # sys_params['b'] = 2.    # ^what he said
    # print('Start system parameters: ', sys_params)
    #
    # system = hopf_mod      # declare which system to solve
    # print('system: ', system.__name__, '\n')
    #
    # points = pseudo_arc_contin(system, u0, 2., vary_par='b', end_par=-1., step=-.01, max_steps=600, discretisation=shooting, solver=fsolve, plot=True, print_progress=False, sys_params=sys_params)


main()
