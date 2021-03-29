import numpy as np
from scipy.optimize import fsolve, fmin
import warnings

warnings.filterwarnings('ignore')

#################       STEP METHODS      #####################################
def euler_step(func, t, v, h, **kwargs):
    v = v + h * func(t, v, **kwargs)
    t = t + h
    return t, v

def rk4_step(func, t, v, h, **kwargs):
    k1 = h * func(t, v, **kwargs)
    k2 = h * func(t + h/2, v + k1/2, **kwargs)
    k3 = h * func(t + h/2, v + k2/2, **kwargs)
    k4 = h * func(t + h, v + k3, **kwargs)
    v = v + (k1 + 2*k2 + 2*k3 + k4) / 6
    t = t+h
    return t, v

################        SOLVERS         #######################################
def rescale_dt(t1, t2, deltat_max):
    if deltat_max % (t2-t1) != 0:
        deltat_max = (t2-t1) / np.ceil((t2-t1) / deltat_max)
    return deltat_max, np.round((t2-t1)/deltat_max)

def solve_to(func, t1, t2, v, deltat_max_orig=0.01, method='RK4', **kwargs):
    tl = [t1]
    vl = [v]
    # rescale dt
    deltat_max = rescale_dt(t1, t2, deltat_max_orig)[0]
    if method == 'Euler':
        # This cond is still necessary to cope with the rounding of step sizes
        # if they become irrational after rescaling
        while t1 < t2:
            if t1 + deltat_max <= t2:
                t1, v = euler_step(func, t1, v, deltat_max, **kwargs)
                vl.append(v)
                tl.append(t1)
            else:
                deltat_max = t2 - t1
                t1, x = euler_step(func, t1, v, deltat_max, **kwargs)
                vl.append(v)
                tl.append(t1)

    if method == 'RK4':
        while t1 < t2:
            if t1 + deltat_max <= t2:
                t1, v = rk4_step(func, t1, v, deltat_max, **kwargs)
                vl.append(v)
                tl.append(t1)
            else:
                deltat_max = t2 - t1
                t1, v = rk4_step(func, t1, v, deltat_max, **kwargs)
                vl.append(v)
                tl.append(t1)

        vl = np.transpose(vl)
    return tl, vl

def solve_ode(func, t1, t2, v0, stepsizes, method='RK4'):
    #############################
    # DOES NOT WORK WITH IMPROVED SOLVETO - WILL ADAPT
    #############################

    """
    A function that solves an ODE between two bounds for a variety of stepsizes.

    Parameters
    ----------
    func : function
        The ODE system to solve. The ode function should take two parameters,
        the independent variable and a list of dependent variables, and return
        the right-hand side of the ODE as a numpy.array.

    t1 : float
        The start value of the independent variable

    t2 : float
        The final value of the independent variable

    v0 : numpy.array
        A numpy.array of the initial values of the dependent variables

    method : string, optional
        All integration methods will be done using the RK4 method by default,
        but can be done by Euler if the argument 'Euler' is passed.

    step : float, optional
        The stepsize to be used in the integration, defaults to 0.001

    Returns
    -------
    Returns a numpy.array containing the corrected initial values
    for the limit cycle. If the numerical root finder failed, the
    returned array is empty.
    """

    tls = []
    sols = []
    for size in stepsizes:
        tl, sol = solve_to(func, t1, t2, v0, size, method)
        tls.append(tl)
        sols.append(sol)
    return tls, sols

################     SHOOTING METHOD     ######################################
# Wrapper class for returns from the shooting method
class Shot_Sol(object):
    def __init__(self, ics, period):
        self.ics = ics
        self.period = period

def shooting(func, u0, t2=1000, xtol=1.0e-01, condeq = 0, cond='min', **kwargs):
    """
    A function that uses numerical shooting to find limit cycles of
    a specified ODE.

    Parameters
    ----------
    func : function
        The ODE system to apply shooting to. The ode function should take
        a single parameter (the state vector) and return the
        right-hand side of the ODE as a numpy.array.

    u0 : numpy.array
        An initial guess at the initial values for the limit cycle.

    t2 : float, optional
        A second time to which system will be solved to find a min/max if need be

    xtol : float, optional
        The root-finding calculation will terminate if the relative error
        between two consecutive iterates is at most xtol.

    condeq : integer, optional
        The equation in the system to look at for the phase condition. For
        example, if a user wanted to use the second equation in the system, they
        would pass a 1. Defaults to zero

    cond : string or float, optional
        Defaults to 'min' which starts the solution at it's minimum value.
        Similarly, if 'max' is passed, the selected equation solution will start
        at a maximum.

        If a float is passed, the selected equation will start at the float
        value. If outside the range, the closest of the max or min will be
        chosen instead.

    Returns
    -------
    Returns a numpy.array containing the corrected initial values
    for the limit cycle. If the numerical root finder failed, the
    returned array is empty.
    """

    try:
        func(0, u0)
    except:
        print('WARNING: initial conditions are of wrong dimension for your system.')
        exit()

    tl, vl = solve_to(func, 0, t2, u0, **kwargs)
    orbit = isolate_orbit(tl, vl[condeq])
    period = orbit.period

    if type(cond) == str:
        if cond == 'max':
            intercept = orbit.max_height
        elif cond == 'min':
            intercept = orbit.min_height
    else:
        if cond > orbit.max_height:
            intercept = orbit.max_height
        elif cond < orbit.min_height:
            intercept = orbit.min_height
        else:
            intercept = cond

    def F(u0, T):
        # Grab the last value of the solve
        tl, vl = solve_to(func, 0, T, u0, **kwargs)
        return np.array([v[-1] for v in vl])

    def G(u0):
        u0[condeq] = intercept
        return u0 - F(u0, period)

    try:
        new_vect = fsolve(G, u0, xtol=xtol)
        return Shot_Sol(np.round(new_vect, 6), period)
    except:
        print('WARNING: numerical root finder has not converged')


##################  PERIOD FINDER   ###########################################
# Wrapper class for tidying up orbit returns
class Orbit(object):
    def __init__(self, period, max_height, min_height):
        self.period = period
        self.max_height = max_height
        self.min_height = min_height

def isolate_orbit(iv, dv, peak_tol=0.001, **kwargs):
    """
    A function that uses numerical shooting to find limit cycles of
    a specified ODE.

    Parameters
    ----------
    iv : array
        The idependent variable; i.e. time domain

    dv : array
        The variable to find the period and heights of

    peak_tol : float, optional
        The function will print a warning if periodic measurements
        such as period and peak/trough heights do not converge within
        this tolerance

    Returns
    -------
    .period : float
        The period of the function measured peak-to-peak

    .max_height : float
        The value of the dependent variable at its peak

    .min_height : float
        The value of the dependent variable at a trough
    """

    # initialise list of peak values and an index to trace them to as well as heights
    peak_times = []
    peak_index = []
    peak_heights = []

    trough_times = []
    trough_index = []
    trough_heights = []

    # loop through the dep var
    for i in range(1, len(iv)-1):

        # find the peaks and add them to the peak list
        if dv[i] > dv[i-1] and dv[i] > dv[i+1]:
            peak_times.append(iv[i])
            peak_index.append(i)
            peak_heights.append(dv[i])

        # find troughs and add them to trough list
        elif dv[i] < dv[i-1] and dv[i] < dv[i+1]:
            trough_times.append(iv[i])
            trough_index.append(i)
            trough_heights.append(dv[i])

    # create wavelength values from final peaks and final troughs
    peak_wlen = peak_times[-1] - peak_times[-2]
    trough_wlen = trough_times[-1] - trough_times[-2]

    # find the float values of the period and the trough/peak heights
    period = peak_wlen
    max_height = peak_heights[-1]
    min_height = trough_heights[-1]

    peak_err = abs((peak_heights[-1] - peak_heights[-2]) / peak_heights[-1])

    if abs((peak_wlen - trough_wlen) / peak_wlen) > peak_tol:
        print('WARNING: Wavelength derived from peaks and troughs differ by more than desired error tolerance. Consider solving over a greater range of values or choose a different initial guess')
    if peak_err > peak_tol:
        print('WARNING: Final two peak heights differ by more than desired error tolerance. Consider solving over a greater range of values or choose a different initial guess')

    return Orbit(period, max_height, min_height)


##############    CONTINUATION         ########################################

def continuation(func, u0, par0, vary_par=0, step_size=0.01, max_steps=100, discretisation=shooting, solver = fsolve):

    varied_params = []

    # Constructing some params to feed into the solver
    if type(par0) == list:
        for i in range(max_steps):
            point = []
            for j in range(len(par0)):
                if j == vary_par:
                    point.append(par0[j] + i * step_size)
                else:
                    point.append(par0[j])
            varied_params.append(point)
    else:
        for i in range(max_steps):
            point = par0 + i * step_size
            varied_params.append(point)

    # Initialise list of roots
    roots = []

    # Solve system for the roots
    for val in varied_params:
        u0 = solver(discretisation(func), u0, args=val)
        roots.append(u0)

    # Initialise list of initial conditions
    if type(par0) == list:
        iv = []
        for point in varied_params:
            iv.append(point[vary_par])
    else:
        iv = varied_params

    return iv, roots

###############         ERROR FINDER    #######################################
def get_abs_err_av(tl, sol, func):
    errors = []
    for i in range(len(tl)):
        errors.append(abs(func(tl[i]) - sol[i]))
    return np.mean(errors)
