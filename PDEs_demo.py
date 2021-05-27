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
from ODE_Utils3 import *
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

    # Set up the numerical environment variables
    x = np.linspace(0, L, mx+1) # mesh points in space
    t = np.linspace(0, T, nt+1) # mesh points in time

    u_j = np.zeros(mx+1)
    # get the initial vector
    for i in range(0, mx+1):
        u_j[i] = u_I(x[i])

    u_jp1 = solve_diffusion_pde(u_j, x, t, mx, kappa, L, T, direction='fe')

    # Plot the final result and exact solution
    plt.plot(x,u_jp1,'ro',label='num')
    xx = np.linspace(0,L,250)
    plt.plot(xx,u_exact(xx,T),'b-',label='exact')
    plt.xlabel('x')
    plt.ylabel('u(x,0.5)')
    plt.legend(loc='upper right')
    plt.show()

    T_steady = steady_state(u_j, mx, nt, kappa, L, .1, step_size=.01, tol=1e-2, max_steps=200)
    print('\nSteady state at T=', T)

main()
