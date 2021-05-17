import numpy as np
from scipy.optimize import fsolve, fmin
import warnings
import sys
from functools import reduce

warnings.filterwarnings('ignore')

#################       STEP METHODS      #####################################
def euler_step(func, t, v, h, **params):
    v = v + h * func(t, v, **params)
    t = t + h
    return t, v

def rk4_step(func, t, v, h, **params):
    k1 = h * func(t, v, **params)
    k2 = h * func(t + h/2, v + k1/2, **params)
    k3 = h * func(t + h/2, v + k2/2, **params)
    k4 = h * func(t + h, v + k3, **params)
    v = v + (k1 + 2*k2 + 2*k3 + k4) / 6
    t = t + h
    return t, v

################    RESCALING           #######################################

def rescale_dt(t1, t2, deltat_max):
    if deltat_max % (t2-t1) != 0:
        deltat_max = (t2-t1) / np.ceil((t2-t1) / deltat_max)
    return deltat_max, np.round((t2-t1)/deltat_max)

################        SOLVERS         #######################################
def solve_to(func, t1, t2, v, deltat_max_orig=0.01, method=rk4_step, **params):
    tl = [t1]
    vl = [v]

    # rescale dt
    deltat_max = rescale_dt(t1, t2, deltat_max_orig)[0]

    # Parameters have to be fed in as keyword arguments at the end so they can
    # be correctly fed into the functions
    # print('params: ', params)

    while t1 < t2:
        if t1 + deltat_max <= t2:
            # The **params here unpacks the keyword arg dictionary into the func
            t1, v = method(func, t1, v, deltat_max, **params)
            vl.append(v)
            tl.append(t1)
        else:
            deltat_max = t2 - t1
            t1, v = method(func, t1, v, deltat_max, **params)
            vl.append(v)
            tl.append(t1)

    vl = np.transpose(vl)
    return tl, vl

# Wrapper class for returns of solve_ode function
# the estimates are the last vector values
# the tracings are the full solves
class Solved_ODE(object):
    def __init__(self, stepsizes, tls, sols):
        estimates = {}
        tracings = {}

        for i in range(len(stepsizes)):
            estimates[stepsizes[i]] = sols[i][0, -1]
            tracings[stepsizes[i]] = np.vstack((np.array(tls[i]), sols[i]))

        self.estimates = estimates
        self.tracings = tracings

def solve_ode(func, t1, t2, v0, stepsizes=[1., .1, .01], **params):

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

    kwargs : optional keyword arguments
        These keyword arguments can be any of the keyword arguments defined in
        solve_to (In the current version 2021.03.31 the only option is method
        which can take euler_step or rk4_step)
        -- and/or --
        parameter values for the system

    Returns
    -------
    .estimates : dict
        .estimates is a dictionary where the stepsize is a key that points to
        final values of the solve

    .tracings : dict
        .tracings is a dictionary where the stepsize is a key that points to a
        numpy array of solved values as t varies between the two bounds, with t
        as the first column. This way the user can plot the solving methods
        after decontructing the dictionary.
    """

    tls = []
    sols = []
    for size in stepsizes:
        tl, sol = solve_to(func, t1, t2, v0, deltat_max_orig=size, **params)
        tls.append(tl)
        sols.append(sol)

    return Solved_ODE(stepsizes, tls, sols)

################     SHOOTING METHOD     ######################################

# Wrapper class for returns from the shooting method
class Shot_Sol(object):
    def __init__(self, ics, period):
        self.ics = ics
        self.period = period

def shooting(func, u0, t2=1000, condeq = 0, cond='extrema', deltat_max=.01, **params):

    """
    A function that discretises a function to a shooting function.

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

    condeq : integer, optional
        The equation in the system to look at for the phase condition. For
        example, if a user wanted to use the second equation in the system, they
        would pass a 1. Defaults to 0

    cond : string or float, optional
        Defaults to 'max' which starts the solution at it's maximum value.
        Similarly, if 'min' is passed, the selected equation solution will start
        at a minimum.

        If a float is passed, the selected equation will start at the float
        value. If outside the range, the closest of the max or min will be
        chosen instead.

    Returns
    -------
    G : function
        The shooting discretisation function
    """

    def F(u0):
        tl, vl = solve_to(func, 0, u0[0], u0[1:], deltat_max, **params)
        return vl[:, -1]

    def phi(u0):
        return u0[condeq+1] - cond

    def int_phi(u0):
        return func(u0[0], u0[1:], **params)[condeq]

    # define G function to solve which is now n+1 dim
    def G(u0):
        if cond == 'extrema':
            return np.append([int_phi(u0)], u0[1:] - F(u0))
        else:
            return np.append([phi(u0)], u0[1:] - F(u0))

    return G

def shoot_root(func, u0, t_guess, discretisation=shooting, xtol=1.0e-10, **params):
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

    notes:
     - if t-guess is too large it will get T wrong

    Returns
    -------
    Returns a numpy.array containing the corrected initial values
    for the limit cycle. If the numerical root finder failed, the
    returned array is empty.
    """

    u0 = np.append([t_guess], u0)
    new_vect = fsolve(discretisation(func, u0, **params), u0, xtol=xtol)

    if new_vect[0] < 0:
        print('\nWARNING: Negative period found\n- Phase condition may be out of system range.\n- Consider changing initial conditions guess. \n\n(TIP: If you wish to fix a variable to a particular value it is not necessary to set it to this exact value in the initial condition guess.)')
        exit()

    return Shot_Sol(new_vect[1:], new_vect[0])

##################  PERIOD FINDER   ###########################################
# Wrapper class for tidying up orbit returns
class Orbit(object):
    def __init__(self, period, max_height, min_height):
        self.period = period
        self.max_height = max_height
        self.min_height = min_height

def isolate_orbit(iv, dv, peak_tol=0.001, **kwargs):
    """
    A function that identifies parameters of a period oscillation in a specified ODE.

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

    if len(peak_times) < 2:
        sys.exit('ERROR: no orbits can be found in this system. Consider a longer period to solve over or a smaller step size')

    # create wavelength values from final peaks and final troughs
    peak_wlen = peak_times[-1] - peak_times[-2]
    trough_wlen = trough_times[-1] - trough_times[-2]

    # find the float values of the period and the trough/peak heights
    period = peak_wlen
    max_height = peak_heights[-1]
    min_height = trough_heights[-1]

    peak_err = abs((peak_heights[-1] - peak_heights[-2]) / peak_heights[-1])

    warning = None
    if abs((peak_wlen - trough_wlen) / peak_wlen) > peak_tol:
        warning = '\nWARNING: Wavelength derived from peaks and troughs differ by more than desired error tolerance. Consider solving over a greater range of values or choose a different initial guess\nAlternatively, there may be no stable limit cycle in this system.\n'
    if peak_err > peak_tol:
        warning = '\nWARNING: Final two peak heights differ by more than desired error tolerance. Consider solving over a greater range of values or choose a different initial guess\nAlternatively, there may be no stable limit cycle in this system.\n'

    return Orbit(period, max_height, min_height), warning

def param_discretise(func, x, **params):

    # A function which constrains a function at a particular parameter. Works similar to lambda but is simpler

    # BUG TO FIX:
    # Won't work if user inputs extra args, can be handled by analysing function inputs
    def new_func(x):
        return func(x, **params)
    return new_func

def vary_params(max_steps, step_size, direction, **params):
    varied_params = []

    if params == {}:
        sys.exit('ERROR: you have not specified a parameter to vary and/or its initial value')

    keys = list(params.keys())
    to_vary = keys[0]

    for i in range(max_steps):
        current_params = {}
        current_params[to_vary] = params[to_vary] + i * step_size * direction

        if len(keys) > 1:
            for other_param in keys[1:]:
                current_params[other_param] = params[other_param]

        varied_params.append(current_params)

    return to_vary, varied_params

#############    CONTINUATION         ########################################
def continuation(func, u0, t_guess, step_size=0.01, max_steps=100, discretisation=shooting, solver=fsolve, direction=1, **params):

    # User can specify discretisation=param_discretise if they wish to keep the function the same i.e. lambda x:x

    # In the **params, as long as they are set after the param to vary, the user can specify other args such as, for shooting, deltat_max if they wish to speed up computation, though this may decrease accuracy.
    to_vary, varied_params = vary_params(max_steps, step_size, direction, **params)
    print('Parameters ranging from:')
    print(varied_params[0], ' to ', varied_params[-1])
    # Initialise list of roots
    roots = []
    iv = []
    # If shooting, we are clearly dealing with periodicity so will need a period to solve.
    if discretisation == shooting:
        u0 = np.append([t_guess], u0)

    for val in varied_params:
        new_func = discretisation(func, u0, **val)
        u0 = solver(new_func, u0, xtol=1e-10)
        iv.append(val[to_vary])
        roots.append(u0)

    return iv, np.transpose(roots)


################     DOESN'T WORK      ##################################
def pseudo_shooting(func, v1, t2=1000, condeq = 0, cond='extrema', deltat_max=.01, **params):

    params[list(params.keys())[0]] = v1[0]

    def F(v1):
        tl, vl = solve_to(func, 0, v1[1], v1[2:], deltat_max, **params)
        return vl[:, -1]

    def phi(v1):
        return v1[2:][condeq] - cond

    def int_phi(v1):
        return func(v1[1], v1[2:], **params)[condeq]

    def G(v1):
        if cond == 'extrema':
            return np.append([int_phi(v1)], v1[2:] - F(v1))
        else:
            return np.append([phi(v1)], v1[2:] - F(v1))

    return G

def pseudo_eq_disc(v_true, v_est, secant):
    def pseudo_eq(v_true):
        return (v_true - v_est) * secant
    return pseudo_eq

def pseudo_arc_cont(func, v0, v1, discretisation=pseudo_shooting, solver=fsolve, max_steps=20, **params):

    # In the **params, as long as they are set after the param to vary, the user can specify other args such as, for shooting, deltat_max if they wish to speed up computation, though this may decrease accuracy.

    # Initialise list of roots
    sols = []
    to_vary = list(params.keys())[0]
    val = params[to_vary]

    step = 0
    while step < max_steps:

        secant = v1 - v0 # generate secant
        val = val + secant[0] # increment param
        params[to_vary] = val # ^what he said
        pred = v1 + secant

        new_func = discretisation(func, pred, **params) # shooting discretise
        v_true = solver(new_func, v1, xtol=1e-10)
        print(v_true)

        sols.append(v_true)
        v0 = v1 # move v0 along
        v1 = v_true # move v1 along
        step += 1 # increase step

    return sols

###############         ERROR FINDER    #######################################
def get_abs_err_av(tl, sol, func):
    errors = []
    for i in range(len(tl)):
        errors.append(abs(func(tl[i]) - sol[i]))
    return np.mean(errors)
