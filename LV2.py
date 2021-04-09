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

    ivs = np.array([.3, .3])
    # solve the system
    tl, vl = solve_to(dvdt, 0., 90., ivs, b=.3)

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
    ### PLOT orbit
    plt.plot(vl[0], vl[1], lw=3)
    plt.xlabel('Prey')
    plt.ylabel('Predator')
    plt.arrow(vl[0][1000], vl[1][1000], .005, .0022, shape='full', lw=0, length_includes_head=True, head_width=.003)
    # plt.grid()
    plt.show()

    # shot = shoot_root(dvdt, ivs, b=.1, cond=.2)
    # print(shot.ics)
    # print(shot.period)
    #
    # tl, vl = solve_to(dvdt, 0., shot.period, shot.ics, b=.1)
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
    # ### PLOT orbit
    # plt.plot(vl[0], vl[1], lw=4)
    # plt.xlabel('Prey')
    # plt.ylabel('Predator')
    # plt.arrow(vl[0][700], vl[1][700], .05, .004, shape='full', lw=0, length_includes_head=True, head_width=.03)
    # # plt.grid()
    # plt.show()


if __name__ == '__main__':
    main()
