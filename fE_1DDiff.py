# simple forward Euler solver for the 1D heat equation
#   u_t = kappa u_xx  0<x<L, 0<t<T
# with zero-temperature boundary conditions
#   u=0 at x=0,L, t>0
# and prescribed initial temperature
#   u=u_I(x) 0<=x<=L,t=0

import numpy as np
import pylab as plt
from math import pi
from scipy.sparse.linalg import spsolve
from ODE_Utils2 import *
from loguru import logger

# Set problem parameters/functions
kappa = .01 # diffusion constant
L = 1. # length of spatial domain
T = .5 # total time to solve for

def u_I(x):
    # initial temperature distribution
    y = (np.sin(pi*x/L))**1
    return y

def u_exact(x,t):
    # the exact solution
    y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)
    return y

@logger.catch
def main():
    # Set numerical parameters
    mx = 20     # number of gridpoints in space
    nt = 1000   # number of gridpoints in time
    x, u_jp1 = solve_diffusion_pde(u_I, mx, nt, kappa, L, T, direction='fe')

    # Plot the final result and exact solution
    plt.plot(x,u_jp1,'ro',label='num')
    xx = np.linspace(0,L,250)
    plt.plot(xx,u_exact(xx,T),'b-',label='exact')
    plt.xlabel('x')
    plt.ylabel('u(x,0.5)')
    plt.legend(loc='upper right')
    plt.show()

main()
