from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
from ODE_Utils import *

def dvdt(t, vect):
    u1, u2 = vect

    s=-1

    du1dt = b*u1 - u2 + s*u1*((u1 ** 2) + (u2 ** 2))
    du1dt = u1 + b*u2 + s*u2*((u1 ** 2) + (u2 ** 2))

    return np.array([du1dt, du2dt])

@logger.catch()
def main():
    pass

if __name__ == '__main__':
    main()
