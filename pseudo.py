from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
from ODE_Utils2 import *
from scipy.optimize import fsolve

#set up the ode system in one function
def dvdt(t, vect, a = 1, b = .1, d = .1):

    x = vect[0]
    y = vect[1]

    dxdt = x*(1-x) - a*x*y/(d+x)
    dydt = b*y*(1- (y/x))

    return np.array([dxdt, dydt])

@logger.catch
def main():

    # Init guesses
    u_guess = np.array([.3, .3])
    t_guess = 30.

    # first and second known vals
    b0 = .1
    shot = shoot_root(dvdt, u_guess, t_guess, b=b0)
    u0 = shot.ics
    t0 = shot.period
    b1 = b0 + .01
    shot = shoot_root(dvdt, u_guess, t_guess, b=b1)
    u1 = shot.ics
    t1 = shot.period
    u0 = np.append([t0], u0)
    u1 = np.append([t1], u1)
    v0 = np.append([b0], u0)
    v1 = np.append([b1], u1)
    print('Starting known sols: ')
    print('v0: ', v0)
    print('v1: ', v1)

    # number of iteration steps
    max_steps = 10

    secant = v1 - v0
    u_guess = v1 + secant

    params = {}

    root = fsolve(arc_len, u_guess, args=(v0, v1, params))

    plt.plot(sols[0], sols[2])
    plt.ylabel('Roots')
    plt.xlabel('c')
    plt.grid()
    plt.title('Root of equation as c varies')
    plt.show()

if __name__ == '__main__':
    main()
