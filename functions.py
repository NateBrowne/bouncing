import numpy as np

def dvdt(t, vect):
    x = vect[0]
    y = vect[1]
    return np.array([y, -x])

def lotka_volterra(t, vect, a=1, b=.1, d=.1):
    x, y = vect
    dxdt = x*(1-x) - a*x*y/(d+x)
    dydt = b*y*(1- (y/x))
    return np.array([dxdt, dydt])

def hopf_norm(t, vect, b=2., s=-1):
    u1, u2 = vect
    du1dt = b*u1 - u2 + s*u1*(u1**2 + u2**2)
    du2dt = u1 + b*u2 + s*u2*(u1**2 + u2**2)
    return np.array([du1dt, du2dt])

def hopf_mod(t, vect, b=2.):
    u1, u2 = vect
    du1dt = b*u1 - u2 + u1*(u1**2 + u2**2) - u1*((u1**2 + u2**2)**2)
    du2dt = u1 + b*u2 + u2*(u1**2 + u2**2) - u2*((u1**2 + u2**2)**2)
    return np.array([du1dt, du2dt])

def cubic(x, b=-.2):
    return x**3 - x + b
