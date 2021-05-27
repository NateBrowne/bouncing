from ODE_Utils3 import *
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
from functions import *

@logger.catch
def main():

    print('\nSHOOTING LOTKA VOLTERRA')
    # Set up an initial value
    u0 = np.array([.9, .9])
    print('Init guess: ', u0)

    # Set up system parameters:
    sys_params = {}
    sys_params['b'] = .2
    print('Params: ', sys_params)

    system = lotka_volterra

    ##############    shooting lotka_volterra     #######################
    # Of course, shooting works faster with a close initial guess. By activating improve_guess, there is the option to increase the chance of convergence by creating a good initial guess. (Not always the right move though).
    sol = shoot_root(system, u0, sys_params=sys_params, improve_guess=True, plot=True)
    print('ICs: ', sol.ics, '  Period: ', sol.period)

    ################   plotting a phase portrait   ###############
    tl, vl = solve_to(system, 0, sol.period, sol.ics, sys_params=sys_params)
    plt.plot(vl[:, 0], vl[:, 1])
    plt.grid()
    plt.show()

    #################    TEST SHOOTING ON HOPF NORM    #####################
    print('\n\nSHOOTING HOPF NORMAL FORM with condeq=1, cond=.5, improve_guess=False')
    u0 = np.array([.5, .1]) # new initial guess
    print('Init guess: ', u0)

    sys_params = {}         # set system params
    sys_params['b'] = 2.    # ^what he said
    print('Params: ', sys_params)

    system = hopf_norm      # declare which system to solve

    # shoot some ics and a period
    sol = shoot_root(system, u0, sys_params=sys_params, condeq=1, plot=True, improve_guess=False, cond=.5)
    print('ICs: ', sol.ics, '  Period: ', sol.period)

    # solve and display a phase portrait
    tl, vl = solve_to(system, 0, sol.period, sol.ics, sys_params=sys_params)
    plt.plot(vl[:, 0], vl[:, 1])
    plt.grid()
    plt.show()

    ################    TEST SHOOTING ON HOPF MOD    #####################
    print('\n\nSHOOTING HOPF MODIFIED condeq=1, improve_guess=False')
    u0 = np.array([.6, .6]) # new initial guess
    print('Init guess: ', u0)

    sys_params = {}         # set system params
    sys_params['b'] = 2.    # ^what he said
    print('Params: ', sys_params)

    system = hopf_mod      # declare which system to solve

    # shoot some ics and a period
    sol = shoot_root(system, u0, sys_params=sys_params, condeq=1, plot=True, improve_guess=False)
    print('ICs: ', sol.ics, '  Period: ', sol.period)

    # solve and display a phase portrait
    tl, vl = solve_to(system, 0, sol.period, sol.ics, sys_params=sys_params)
    plt.plot(vl[:, 0], vl[:, 1])
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()
