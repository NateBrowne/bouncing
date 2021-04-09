# I've rewritten this code so that argument handling is more versatile on the way up

import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

from ODE_Utils2 import *

# Basic ode
def dvdt(t, v):
    x = v[0] #unpack the vector
    dxdt = x
    return np.array([dxdt])

# This will be the basic form in which a user defines their function:
#   A float value t of the independent variable
#   A numpy array of all other variables
#   Parameters as optional keyword arguments
def dvdt2(t, vect, a=1., b=.1, d=.1):
    x, y = vect #unpack the vector

    dxdt = x*(1-x) - a*x*y/(d+x)
    dydt = b*y*(1- (y/x))

    return np.array([dxdt, dydt])

def dvdt3(t, vect):
    x, y = vect

    dxdt = y
    dydt = -x

    return np.array([dxdt, dydt])

@logger.catch
def main():
    ics = np.array([1.])

    solves = solve_ode(dvdt, 0., 1., ics, method=rk4_step)
    tracing = solves.tracings.get(0.01)
    print(solves.estimates)

    tl = tracing[0]
    vl = tracing[1:]

    plt.plot(tl, vl[0], label='Prey')
    #plt.plot(tl, vl[1], label='Predator')
    plt.ylabel('Population')
    plt.xlabel('Time')
    plt.grid()
    plt.title('Lotka-Volterra, b = 0.1, no shoot')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
