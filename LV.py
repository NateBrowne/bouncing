import numpy as np
import matplotlib.pyplot as plt
from ODE_Utils import *

def plot_sols(plot_sets):
    for plot_set in plot_sets:
        # xs = [item[0] for item in plot_set[1]]
        # plt.plot(plot_set[0], xs, label= plot_set[2])
        plt.plot(plot_set[0], plot_set[1], label= plot_set[2])

    #plt.plot(np.linspace(0, 1, 100), np.exp(np.linspace(0, 1, 100)), label='Actual')
    plt.legend()
    #plt.yscale('log')
    plt.show()
def plot_errs(stepsize, err):
    plt.plot(stepsize, err)
    plt.yscale('log')
    plt.xscale('log')
    plt.show()

#set up the ode system in one function
def dvdt(t, vect):

    a = 1
    d = 0.1
    b = 0.1

    x = vect[0]
    y = vect[1]
    dxdt = x*(1-x) - a*x*y/(d+x)
    dydt = b*y*(1- (y/x))

    return np.array([dxdt, dydt])

# Declare list of possible step sizes
steps = [0.001]
methods = ['RK4'] # List of methods to compare
solutions = []

for method in methods:
    tls, sols = solve_ode_system(dvdt, 0, 100, np.array([0.5, 0.5]), steps, method)
    # errs = [get_abs_err_av(tls[i], sols[i], f_actual) for i in range(len(tls))]
    # plot_errs(steps, errs)

    for i in range(len(tls)):
        # name = str(method) + ' ' + str(steps[i]) + '   Error: ' + str(get_abs_err_av(tls[i], sols[i], f_actual))
        name = str(method) + ' ' + str(steps[i])
        solutions.append([tls[i], sols[i], name])

plot_sols(solutions)
