from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
from ODE_Utils2 import *

#set up the ode system in one function
def dvdt(t, vect, a = 1, b = .1, d = .1):

    x = vect[0]
    y = vect[1]

    dxdt = x*(1-x) - a*x*y/(d+x)
    dydt = b*y*(1- (y/x))

    return np.array([dxdt, dydt])

@logger.catch
def main():

    u_guess = np.array([.3, .3])
    t_guess = 30.

    b0 = .1
    shot = shoot_root(dvdt, u_guess, t_guess, b=b0)
    u0 = shot.ics
    t0 = shot.period

    b1 = b0 + .01
    shot = shoot_root(dvdt, u_guess, t_guess, b=b1)
    u1 = shot.ics
    t1 = shot.period

    v0 = np.append([b0, t0], u0)
    v1 = np.append([b1, t1], u1)

    print('Starting known sols: ')
    print('v0: ', v0)
    print('v1: ', v1)

    sols = pseudo_arc_cont(dvdt, v0, v1, b=.11)

    sols = np.transpose(sols)

    plt.plot(sols[0], sols[2])
    plt.ylabel('Roots')
    plt.xlabel('c')
    plt.grid()
    plt.title('Root of equation as c varies')
    plt.show()

if __name__ == '__main__':
    main()
