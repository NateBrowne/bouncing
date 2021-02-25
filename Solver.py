import numpy as np
import matplotlib.pyplot as plt


def plot_sols(plot_sets):
    for plot_set in plot_sets:
        plt.plot(plot_set[0], plot_set[1], label= plot_set[2])

    plt.plot(np.linspace(0, 1, 100), np.exp(np.linspace(0, 1, 100)), label='Actual')
    plt.legend()
    plt.yscale('log')
    plt.show()

def plot_errs(stepsize, err):
    plt.plot(stepsize, err)
    plt.yscale('log')
    plt.xscale('log')
    plt.show()

#inputs gradient function at 2, current x, current t, and the step size
def euler_step(func, t, x, h):
    x = x + h * func(t, x)
    t = t + h
    return t, x
def rk4_step(func, t, x, h):
    k1 = h * func(t, x)
    k2 = h * func(t + h/2, x + k1/2)
    k3 = h * func(t + h/2, x + k2/2)
    k4 = h * func(t + h, x + k3)
    x += (k1 + 2*k2 + 2*k3 + k4) / 6
    t = t+h
    return t, x

#solves between two time bounds; t1 (start) and t2 (end); returns 2 lists in tuple
def solve_to(func, t1, t2, x, deltat_max, method):
    tl = [t1]
    xl = [x]
    if method == 'Euler':
        while t1 < t2:
            if t1 + deltat_max <= t2:
                t1, x = euler_step(func, t1, x, deltat_max)
                xl.append(x)
                tl.append(t1)
            else:
                deltat_max = t2 - t1
                t1, x = euler_step(func, t1, x, deltat_max)
                xl.append(x)
                tl.append(t1)

    if method == 'RK4':
        while t1 < t2:
            if t1 + deltat_max <= t2:
                t1, x = rk4_step(func, t1, x, deltat_max)
                xl.append(x)
                tl.append(t1)
            else:
                deltat_max = t2 - t1
                t1, x = rk4_step(func, t1, x, deltat_max)
                xl.append(x)
                tl.append(t1)

    return tl, xl

#gives sols from t = 0, t = 1, t = 2, ...  t = t as a list
def solve_ode(func, t1, t2, x0, stepsizes, method='RK4'):
    tls = []
    sols = []
    
    for size in stepsizes:
        tls.append(solve_to(func, t1, t2, x0, size, method)[0])
        sols.append(solve_to(func, t1, t2, x0, size, method)[1])
        
    return tls, sols

#set up the ode
def fdash(t, x):
    return x
# analytical function for comparison
def f_actual(t):
    return np.exp(t)

# finds absolute average error for the integration
def get_abs_err_av(tl, sol, func):
    errors = []
    for i in range(len(tl)):
        errors.append(abs(func(tl[i]) - sol[i]))
    return np.mean(errors)

# Declare list of possible step sizes
steps = [0.1, 0.01, 0.001, 0.0001]
methods = ['Euler', 'RK4'] # List of methods to compare
solutions = []

for method in methods:
    tls = solve_ode(fdash, 0, 1, 1, steps, method)[0]
    sols = solve_ode(fdash, 0, 1, 1, steps, method)[1]
    errs = [get_abs_err_av(tls[i], sols[i], f_actual) for i in range(len(tls))]
    plot_errs(steps, errs)

    for i in range(len(tls)):
        name = str(method) + ' ' + str(steps[i]) + '   Error: ' + str(get_abs_err_av(tls[i], sols[i], f_actual))
        solutions.append([tls[i], sols[i], name])

plot_sols(solutions)
