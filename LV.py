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

    # solve the system
    # tl, vl = solve_to(dvdt, 0, 150, [.3, .3])
    # # find start conds and period of a periodic orbit
    # orbit, warning = isolate_orbit(tl, vl[0])
    # print('real period: ', orbit.period)
    #
    # #### PLOT system wrt time
    # plt.plot(tl, vl[0], label='Prey')
    # plt.plot(tl, vl[1], label='Predator')
    # plt.ylabel('Population')
    # plt.xlabel('Time')
    # plt.grid()
    # plt.title('Lotka-Volterra, b = 0.1, no shoot')
    # plt.legend()
    # plt.show()
    #
    # #### PLOT orbit
    # plt.plot(vl[0], vl[1], label='Orbit')
    # plt.xlabel('Prey')
    # plt.ylabel('Predator')
    # plt.title('Phase portrait b = 0.1, no shoot')
    # plt.grid()
    # plt.show()

    # guess the right ICs
    u0 = np.array([.7, .2])
    t_guess = 30.

    # find the right ICs that lead to the same period orbit as earlier
    shot = shoot_root(dvdt, u0, t_guess)
    print('\nClosest ICs:')
    print('Prey:', shot.ics[0], ' Pred: ', shot.ics[1])
    print('Period:', shot.period)

    # solve the system with new ICs
    tl, vl = solve_to(dvdt, 0, shot.period, shot.ics)

    #### PLOT system wrt time
    plt.plot(tl, vl[0], label='Prey')
    plt.plot(tl, vl[1], label='Predator')
    plt.ylabel('Population')
    plt.xlabel('Time')
    plt.grid()
    plt.title('Lotka-Volterra, b = 0.1 with shot ICs')
    plt.legend()
    plt.show()

    #### PLOT orbit
    plt.plot(vl[0], vl[1], label='Orbit')
    plt.xlabel('Prey')
    plt.ylabel('Predator')
    plt.title('Phase portrait b = 0.1 with shot ICs')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()
