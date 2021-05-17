from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from ODE_Utils2 import *

def hopf_norm(t, vect, b=2., s=-1):
    u1, u2 = vect
    du1dt = b*u1 - u2 + s*u1*(u1**2 + u2**2)
    du2dt = u1 + b*u2 + s*u2*(u1**2 + u2**2)
    return np.array([du1dt, du2dt])

def hopf_mod(t, vect, b=2.):
    u1, u2 = vect
    du1dt = b*u1 - u2 + u1*(u1**2 + u2**2) - u1*((u1**2 + u2**2)**2)
    du1dt = u1 + b*u2 + u2*(u1**2 + u2**2) - u2*((u1**2 + u2**2)**2)
    return np.array([du1dt, du1dt])

def lotka_volterra(t, vect, a=1, b=.1, d=.1):
    x, y = vect
    dxdt = x*(1-x) - a*x*y/(d+x)
    dydt = b*y*(1- (y/x))
    return np.array([dxdt, dydt])

@logger.catch
def main():

    t_guess = 30.
    u0 = [.3, .3]

    now = datetime.now()
    param_vals, roots = continuation(lotka_volterra, u0, t_guess=t_guess, step_size=.002, max_steps=200, b=.1, deltat_max=.1)
    print('Time: ', datetime.now() - now)

    plt.plot(param_vals, roots[1, :])
    plt.plot(param_vals, roots[2, :])
    plt.ylabel('Roots')
    plt.xlabel('b')
    plt.grid()
    plt.title('Root of equation as b varies')
    plt.show()

if __name__ == '__main__':
    main()
