from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
from ODE_Utils2 import *

def fx(x, c=-2.):
    return x**3 - x + c

def hopf_norm(t, vect, b=0.):
    x = vect[0]
    y = vect[1]
    dxdt = b*x - y + x*(x**2 + y**2)
    dydt = x + b*y + y*(x**2 + y**2)
    return np.array([dxdt, dydt])

def hopf_mod(t, vect, b=2.):
    x = vect[0]
    y = vect[1]
    dxdt = b*x - y + x*(x**2 + y**2) - x*((x**2 + y**2)**2)
    dydt = x + b*y + y*(x**2 + y**2) - y*((x**2 + y**2)**2)
    return np.array([dxdt, dydt])

@logger.catch
def main():

    param_vals, roots = continuation(fx, 1.5, max_steps=400, discretisation=lambda x, u0 : x, c=-2.)

    plt.plot(param_vals, roots)
    plt.ylabel('Roots')
    plt.xlabel('c')
    plt.grid()
    plt.title('Root of equation as c varies')
    plt.show()

    # param_vals, roots = continuation(hopf_norm, [1., 1.], max_steps=200, b=0.)
    # plt.plot(param_vals, roots[0])
    # plt.plot(param_vals, roots[1])
    # plt.ylabel('Roots')
    # plt.xlabel('b')
    # plt.grid()
    # plt.title('Root of equation as b varies')
    # plt.show()


if __name__ == '__main__':
    main()
