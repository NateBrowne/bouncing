import numpy as np
import matplotlib.pyplot as plt


def plot_sols(plot_sets):
    for plot_set in plot_sets:
        xs = [item[0] for item in plot_set[1]]
        plt.plot(plot_set[0], xs, label= plot_set[2])

    #plt.plot(np.linspace(0, 1, 100), np.exp(np.linspace(0, 1, 100)), label='Actual')
    plt.legend()
    #plt.yscale('log')
    plt.show()

def plot_errs(stepsize, err):
    plt.plot(stepsize, err)
    plt.yscale('log')
    plt.xscale('log')
    plt.show()

#inputs gradient function at 2, current vector x, y, current t, and the step size
def euler_step(func, t, vect, h):
    vect = vect + h * func(t, vect)
    t = t + h
    return t, vect
    
def rk4_step(func, t, vect, h):
    k1 = h * func(t, vect)
    k2 = h * func(t + h/2, vect + k1/2)
    k3 = h * func(t + h/2, vect + k2/2)
    k4 = h * func(t + h, vect + k3)
    vect = vect + (k1 + 2*k2 + 2*k3 + k4) / 6
    t = t+h
    return t, vect

#solves between two time bounds; t1 (start) and t2 (end); returns 2 lists in tuple
def solve_to(func, t1, t2, v, deltat_max, method):
    tl = [t1]
    vl = [v]
    if method == 'Euler':
        while t1 < t2:
            if t1 + deltat_max <= t2:
                t1, v = euler_step(func, t1, v, deltat_max)
                vl.append(v)
                tl.append(t1)
            else:
                deltat_max = t2 - t1
                t1, x = euler_step(func, t1, v, deltat_max)
                vl.append(v)
                tl.append(t1)

    if method == 'RK4':
        while t1 < t2:
            if t1 + deltat_max <= t2:
                t1, v = rk4_step(func, t1, v, deltat_max)
                vl.append(v)
                tl.append(t1)
            else:
                deltat_max = t2 - t1
                t1, v = rk4_step(func, t1, v, deltat_max)
                vl.append(v)
                tl.append(t1)

    return tl, vl

#gives sols from t = 0, t = 1, t = 2, ...  t = t as a list
def solve_ode(func, t1, t2, v0, stepsizes, method='RK4'):
    tls = []
    sols = []
    for size in stepsizes:
        tls.append(solve_to(func, t1, t2, v0, size, method)[0])
        sols.append(solve_to(func, t1, t2, v0, size, method)[1])
    return tls, sols

#set up the ode
def dvdt(t, vect):
    x = vect[0]
    y = vect[1]
    return np.array([y, -x])

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
    tls, sols = solve_ode(dvdt, 0, 50, np.array([1, 0]), steps, method)
    # errs = [get_abs_err_av(tls[i], sols[i], f_actual) for i in range(len(tls))]
    # plot_errs(steps, errs)

    for i in range(len(tls)):
        # name = str(method) + ' ' + str(steps[i]) + '   Error: ' + str(get_abs_err_av(tls[i], sols[i], f_actual))
        name = str(method) + ' ' + str(steps[i])
        solutions.append([tls[i], sols[i], name])

plot_sols(solutions)
