from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
from ODE_Utils import *

def fx(x, *params):
    c = params
    return x**3 - x + c

def hopf_norm(t, vect, *params):
    x = vect[0]
    y = vect[1]
    b = params
    dxdt = b*x - y + x*(x**2 + y**2)
    dydt = x + b*y + y*(x**2 + y**2)
    return np.array([dxdt, dydt])

def hopf_mod(t, vect, *params):
    x = vect[0]
    y = vect[1]
    b = params
    dxdt = b*x - y + x*(x**2 + y**2) - x*((x**2 + y**2)**2)
    dydt = x + b*y + y*(x**2 + y**2) - y*((x**2 + y**2)**2)
    return np.array([dxdt, dydt])


@logger.catch
def main():

    param_vals, roots = continuation(fx, 1.5, -2, vary_par=0, max_steps=400, discretisation=lambda x : x)
    plt.plot(param_vals, roots)
    plt.ylabel('Roots')
    plt.xlabel('c')
    plt.grid()
    plt.title('Root of equation as c varies')
    plt.show()

    # param_vals, roots = continuation(hopf_norm, [1., 1.], 0, vary_par=0, max_steps=200, discretisation=shooting)
    # plt.plot(param_vals, roots)
    # plt.ylabel('Roots')
    # plt.xlabel('b')
    # plt.grid()
    # plt.title('Root of equation as b varies')
    # plt.show()


if __name__ == '__main__':
    main()
