from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt


def plot_all(tls, sols, steps):
    for i in range(len(tls)):
        plt.plot(tls[i], sols[i], label=steps[i])

    plt.plot(np.linspace(0, 1, 100), np.exp(np.linspace(0, 1, 100)), label='Actual')
    plt.legend()
    plt.yscale('log')
    plt.show()

#inputs gradient function at 2, current x, current t, and the step size
def euler_step(func, x, t, h):
    x = x + h * func(x, t)
    t = t + h
    return x, t


#solves between two time bounds; t1 (start) and t2 (end); returns 2 lists in tuple
def solve_to(func, t1, t2, x, deltat_max):
    tl = [t1]
    xl = [x]
    while t1 < t2:
        if t1 + deltat_max <= t2:
            x, t1 = euler_step(func, x, t1, deltat_max)
            xl.append(x)
            tl.append(t1)
        else:
            deltat_max = t2 - t1
            x, t1 = euler_step(func, x, t1, deltat_max)
            xl.append(x)
            tl.append(t1)
    return tl, xl


#gives sols from t = 0, t = 1, t = 2, ...  t = t as a list
def solve_ode(func, t1, t2, x0, stepsizes):
    tls = []
    sols = []
    for size in stepsizes:
        tls.append(solve_to(func, t1, t2, x0, size)[0])
        sols.append(solve_to(func, t1, t2, x0, size)[1])
    return tls, sols

#set up the ode
def fdash(x, t):
    return x + 1

steps = [0.1, 0.01, 0.001, 0.0001]
tls, sols = solve_ode(fdash, 0, 1, 1, steps)

plot_all(tls, sols, steps)
