from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
from ODE_Utils3 import *
from datetime import datetime

#set up the ode - returns vdot in numpy array
def dvdt(t, vect):
    x = vect[0]
    return np.array([x])

# analytical function for comparison
def f_actual(t):
    return np.exp(t)

@logger.catch
def main():

    steps = [1 * (10**(-1*i)) for i in range(5)] # Declare list of possible step sizes
    methods = [euler_step, rk4_step] # List of methods to compare

    # analytic solution
    t_actual = np.linspace(0, 1, 1000)
    x_actual = f_actual(t_actual)

    ers = [] # list of errors for each method

    for method in methods: # loop through the methods
        plt.title('Solve_ODE with ' + method.__name__)
        plt.xlabel('t')
        plt.ylabel('x')
        plt.plot(t_actual, x_actual, label='Actual')
        # solve and plot
        solved = solve_ode(dvdt, 0, 1, [1], steps, method=method, plot=True)

        err = [] # create list of errs
        for step in steps:
            tl, vl = solved.tracings[step]
            err.append(get_abs_err_av(tl, vl, f_actual)) # add the err
        ers.append(err)

    # error plot
    for i in range(len(ers)):
        plt.plot(steps, ers[i], 'o-', label=methods[i].__name__)

    plt.grid()
    plt.title('Error Comparison')
    plt.ylabel('Error')
    plt.xlabel('Step size')
    plt.yscale('log')
    plt.xscale('log')
    plt.show()

if __name__ == "__main__":
    main()
