from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
from ODE_Utils import *

#set up the ode system in one function
def dvdt(t, vect):

    a = 1
    b = 0.26
    d = 0.1

    x = vect[0]
    y = vect[1]
    dxdt = x*(1-x) - a*x*y/(d+x)
    dydt = b*y*(1- (y/x))

    return np.array([dxdt, dydt])

@logger.catch
def main():

    tl, vl = solve_to(dvdt, 0, 400, [.3, .3], 0.001, method='RK4')
    vl = np.transpose(vl[1])

    ##### PLOT system wrt time
    # plt.plot(tl, vl[0], label='Prey')
    # plt.plot(tl, vl[1], label='Predator')
    #
    # plt.ylabel('Population')
    # plt.xlabel('Time')
    #
    # plt.title('Lotka-Volterra, b = 0.5')
    # plt.legend()

    ##### PLOT orbit
    plt.plot(vl[0], vl[1], label='Orbit')
    plt.xlabel('Predator')
    plt.ylabel('Prey')

    plt.title('Periodic orbit b = 0.1')
    plt.show()

if __name__ == '__main__':
    main()
