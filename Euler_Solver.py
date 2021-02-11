from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt


#inputs gradient function at 2, current x, current t, and the step size
def euler_step(func, x, t, h):
    x = x + h * func(x, t)
    t = t + h
    return x, t


#solves between two time, t1 (start), t2 (end) bounds
def solve_to(func, t1, t2, x, deltat_max):

    while t1 < t2:
        x, t1 = euler_step(func, x, t1, deltat_max)
    return x


#gives sols from t = 0, t = 1, t = 2, ...  t = t as a list
def solve_ode(func, x0, t, stepsizes):
    x = [['Step size'], ['Estimate']]
    for size in stepsizes:
        x[0].append(size)
        x[1].append(solve_to(func, 0, t, x0, size))
    return x

#set up the ode
def fdash(x, t):
    return x + 1

def pretty_print(list):
    for i in range(len(list[0])):
        print(list[0][i], '   ', list[1][i])

pretty_print(solve_ode(fdash, 1, 1, [0.1, 0.01, 0.001, 0.0001]))
