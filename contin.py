from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
from ODE_Utils2 import *

def fx(x, c=-2., b=1.):
    return x**3 - x + c

@logger.catch
def main():

    u0 = np.array([1.5])

    param_vals, roots = continuation(fx, u0, t_guess=2., max_steps=400, discretisation=param_discretise, c=-2., b=2.)

    plt.plot(param_vals, roots[0])
    plt.ylabel('Roots')
    plt.xlabel('c')
    plt.grid()
    plt.title('Root of equation as c varies')
    plt.show()

if __name__ == '__main__':
    main()
