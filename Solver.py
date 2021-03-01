from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
from ODE_Utils import *
from datetime import datetime

#logger.add(sys.stdout, format = "{time} - {level} - {message}")

def plot_sols(plot_sets):
    for plot_set in plot_sets:
        plt.plot(plot_set[0], plot_set[1], label= plot_set[2])

    plt.plot(np.linspace(0, 1, 100), np.exp(np.linspace(0, 1, 100)), label='Actual')
    plt.legend()
    plt.yscale('log')
    plt.show()
def plot_errs(stepsize, err, method):
    plt.plot(stepsize, err, 'ro-')
    plt.yscale('log')
    plt.xscale('log')

    plt.ylabel('Average Absolute Error')
    plt.xlabel('Step Size')

    plt.title(method + ' Error vs. Step Size')

    plt.grid()
    plt.show()

def plot_err_compare(stepsize, err, method):
    plt.plot(stepsize, err, 'o-', label=method)

#set up the ode - returns vdot in numpy array
def dvdt(t, vect):
    x = vect[0]
    return np.array([x])

# analytical function for comparison
def f_actual(t):
    return np.exp(t)

steps = [1 * (10**(-1*i)) for i in range(7)] # Declare list of possible step
                                             # sizes

methods = ['Euler', 'RK4'] # List of methods to compare
solutions = [] # Initialise solution set

# We need to run this method for each solving method - Euler and RK4
@logger.catch
def main():
    for method in methods:
        # Grab a list of timestamps and solved values, init conds inside a list
        tls, sols = solve_ode(dvdt, 0, 1, [1], steps, method)

        # find the average absolute error for each stepsize and put in a list
        errs = [get_abs_err_av(tls[i], sols[i], f_actual) for i in range(len(tls))]
        # plot the errors for the stepsizes
        # plot_errs(steps, errs, method)
        plot_err_compare(steps, errs, method)

        for i in range(len(tls)):

            name = (str(method) + ' ' + str(steps[i]) + '   Error: ' +
                    str(get_abs_err_av(tls[i], sols[i], f_actual)))

            solutions.append([tls[i], sols[i], name])

    #plot_sols(solutions)
    plt.ylabel('Average Absolute Error')
    plt.xlabel('Step Size')
    plt.title('Error vs. Step Size')

    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.grid()
    plt.show()

    # SPEED COMPARE
    # now = datetime.now()
    # tl, vl = solve_to(dvdt, 0, 1, [1], 0.00009, 'Euler')
    # print('time taken: ', datetime.now() - now)
    #
    # now = datetime.now()
    # tl, vl = solve_to(dvdt, 0, 1, [1], 0.1)
    # print('time taken: ', datetime.now() - now)

if __name__ == "__main__":
    main()
