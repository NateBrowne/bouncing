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
L = 1. # length of spatial domain
T = .5 # total time to solve for

def u_I(x):
    # initial temperature distribution
    y = (np.sin(pi*x/L))
    return y

def u_exact(x,t):
    # the exact solution
    y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)
    return y

@logger.catch
def main():
    # Set numerical parameters
    mx = 20

    kappas, nts = pde_contin(u_I, .01, L, T, mx, step_size=.001, plot=True, direction='cn')

main()
