import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy import integrate
import warnings
import sys
from datetime import datetime

warnings.filterwarnings('ignore')

################    RESCALING           #######################################
class Rescale(object):
    # Wrapper class for returns of rescale_dt function
    def __init__(self, t1, t2, deltat):
        self.step_size = deltat
        self.no_steps = np.round((t2-t1)/deltat)

def rescale_dt(t1, t2, deltat=0.01, **kwargs):
    """
    A function that rescales a step size to fit more uniformly within integration bounds.

    Parameters
    ----------
    t1 : float
        The start time bound of the integration

    t2 : float
        The end time bound of the integration

    deltat : float, default 0.01
        The maximum size of the corrected time step

    kwargs : optional keyword arguments
        This ensures all invalid keyword args are ignored in this function

    Returns
    -------
    .step_size: float
        The new re-scaled size of the step

    .no_steps: float
        The number of steps in the new solve

    """
    if deltat % (t2-t1) != 0:
        deltat = (t2-t1) / np.ceil((t2-t1) / deltat)

    return Rescale(t1, t2, deltat)
#################       STEP METHODS      #####################################
def euler_step(func, t, v, deltat=0.01, sys_params={}, **kwargs):
    """
    A function that performs a singular Euler Step on an ODE.

    Parameters
    ----------
    func : function
        The ODE system to solve one step of.

    t : float
        The current time value of the solve

    v : numpy array
        The current state vector

    deltat : float, default 0.01
        The step size of the Euler Method

    sys_params : keyword float arguments, optional, default empty
        Used to specify certain parameters to apply to the function used in the Euler Step. System defaults used if none given.

    kwargs : optional keyword arguments to be ignored

    Returns
    -------
    t : numpy array
        The next time value of the solve incremented by h

    v : numpy array
        The next state vector after one Euler Step
    """
    try:
        v = v + deltat * func(t, v, **sys_params)
    except:
        print('EULER STEP error: step could not be completed. Check system dimesions and system parameters are correct.')
        exit()
    t = t + deltat
    return t, v

def rk4_step(func, t, v, deltat=0.01, sys_params={}, **kwargs):
    """
    A function that performs a singular 4th Order Runge Kutta Step on an ODE.

    Parameters
    ----------
    func : function
        The ODE system to solve one step of.

    t : float
        The current time value of the solve

    v : numpy array
        The current state vector

    deltat : float, default 0.01
        The step size of the RK4 Method

    sys_params : keyword float arguments, optional, default empty
        Used to specify certain parameters to apply to the function used in the RK4 Step. System defaults used if none given.

    kwargs : optional keyword arguments to be ignored

    Returns
    -------
    t : numpy array
        The next time value of the solve incremented by h

    v : numpy array
        The next state vector after one RK4 Step
    """
    try:
        k1 = deltat * func(t, v, **sys_params)
    except:
        print('RK4 STEP error: step could not be completed. Check system dimesions and system parameters are correct.')
        exit()
    k2 = deltat * func(t + deltat/2, v + k1/2, **sys_params)
    k3 = deltat * func(t + deltat/2, v + k2/2, **sys_params)
    k4 = deltat * func(t + deltat, v + k3, **sys_params)
    v = v + (k1 + 2*k2 + 2*k3 + k4) / 6
    t = t + deltat
    return t, v
################        SOLVERS         #######################################
def solve_to(func, t1, t2, v0, method=rk4_step, plot=False, **params):

    """
    A function that solves a function by a given method between two bounds given a set of initial conditions

    Parameters
    ----------

    func : function
        The ODE system to solve between the bounds.

    t1 : float
        The start time bound of the integration

    t2 : float
        The end time bound of the integration

    v0 : numpy array
        Initial condition state vector

    method : function, default rk4_step
        The numerical integration routine to use in the solve

    plot : boolean, optional
        Decides whether or not the solves will be plotted with respect to time after solving, default False

    params : optional keyword args
        Arguments that can be optionally changed for different parts of the routine. For example, the deltat could be adjusted, or non-default system variables could be inputted

    Returns
    -------
    tl: numpy array
        The time values through the solve

    vl: numpy array
        The state vectors as they change through time
    """
    tl = [t1]
    vl = [v0]

    # rescale dt
    params['deltat'] = rescale_dt(t1, t2, **params).step_size

    # Parameters have to be fed in as keyword arguments at the end so they can
    # be correctly fed into the functions

    while t1 < t2: # while we have not finished integrating
        if t1 + params['deltat'] <= t2:
            # The **params here unpacks the keyword arg dictionary into the func
            t1, v0 = method(func, t1, v0, **params)
            vl.append(v0)
            tl.append(t1)
        else:
            params['deltat'] = t2 - t1
            t1, v0 = method(func, t1, v0, **params)
            vl.append(v0)
            tl.append(t1)

    if plot == True:
        plt.plot(tl, vl)
        plt.ylabel('Vars')
        plt.xlabel('t')
        plt.grid()
        plt.show()

    return np.array(tl), np.array(vl)

class Solved_ODE(object):
    # Wrapper class for returns of solve_ode function
    # the estimates are the last vector values
    # the tracings are the full solves
    def __init__(self, stepsizes, tls, sols):
        estimates = {}
        tracings = {}

        for i in range(len(stepsizes)):
            estimates[stepsizes[i]] = sols[i][-1, :]
            tracings[stepsizes[i]] = [tls[i], sols[i]]

        self.estimates = estimates
        self.tracings = tracings

def solve_ode(func, t1, t2, v0, stepsizes=(1., .1, .01), plot=False, **params):

    """
    A function that solves an ODE between two bounds for a variety of stepsizes.

    Parameters
    ----------
    func : function
        The ODE system to solve. The ode function should take two parameters,
        the independent variable and a list of dependent variables, and return
        the right-hand side of the ODE as a numpy.array.

    t1 : float
        The start time of integration

    t2 : float
        The start time of integration

    v0 : numpy.array
        A numpy.array of the initial values of the dependent variables

    stepsizes : tuple, optional
        The stepsizes to be used in the integrations, default (1., .1, .01)

    plot : boolean, optional
        Decides whether or not the solves will be plotted with respect to time after solving, default False

    params : optional keyword arguments
        These keyword arguments can be any of the keyword arguments defined in
        solve_to such as method, deltat or parameter values for the system

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

    # ignore any other stepsize args found in **params or there will be repeated arguments
    if 'deltat' in list(params.keys()):
        params['deltat'].pop()

    tls = []
    vls = []
    for size in stepsizes:
        tl, vl = solve_to(func, t1, t2, v0, deltat=size, **params)
        tls.append(tl)
        vls.append(vl)

    solved = Solved_ODE(stepsizes, tls, vls)

    if plot == True:
        for i in list(solved.tracings.keys()):
            tl, vl = solved.tracings[i]
            plt.plot(tl, vl, label=i)
        plt.legend()
        plt.show()

    return solved
##################  PERIOD FINDER   ###########################################
class Orbit(object):
    # Wrapper class for tidying up orbit returns
    def __init__(self, period, max_height, min_height):
        self.period = period
        self.max_height = max_height
        self.min_height = min_height

def isolate_orbit(iv, dv, peak_tol=1e-2, warnings=False):
    """
    A function that identifies parameters of a period oscillation in a specified ODE.

    Parameters
    ----------
    iv : 1D numpy array
        The idependent variable; i.e. time domain

    dv : 1D numpy array
        The variable to find the period and heights of

    peak_tol : float, optional
        The function will print a warning if periodic measurements
        such as period and peak/trough heights do not converge within
        this tolerance. Default 1e-2

    warnings : boolean, optional
        Set to true if user wishes to see warnings with regards to peak heights, default False

    Returns
    -------
    .period : float
        The period of the function measured peak-to-peak

    .max_height : float
        The value of dv at its peak

    .min_height : float
        The value of dv at a trough
    """

    # initialise list of peak values and an index to trace them to as well as heights
    peak_times, peak_index, peak_heights = [], [], []
    trough_times, trough_index, trough_heights = [], [], []

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

    warning = 'No warning'
    if abs((peak_wlen - trough_wlen) / peak_wlen) > peak_tol:
        warning = '\nISOLATE ORBIT WARNING: Wavelength derived from peaks and troughs differ by more than desired error tolerance. Consider solving over a greater range of values or choose a different initial guess\nAlternatively, there may be no stable limit cycle in this system.\n'
    if peak_err > peak_tol:
        warning = '\nISOLATE ORBIT WARNING: Final two peak heights differ by more than desired error tolerance. Consider solving over a greater range of values or choose a different initial guess\nAlternatively, there may be no stable limit cycle in this system.\n'

    if warnings == True:
        print(warning)

    return Orbit(period, max_height, min_height)
################     SHOOTING METHOD     ######################################
class Shot_Sol(object):
    # Wrapper class for returns from the shooting method
    def __init__(self, ics, period):
        self.ics = ics
        self.period = period

def shooting(func, u0t, condeq=0, cond='extrema', **params):

    """
    A function that discretises a function to a shooting root-finding problem.

    Parameters
    ----------
    func : function
        The ODE system to apply shooting to.

    u0t : numpy.array
        An initial guess at the initial values for the limit cycle and a guess at its period in the form [u0, ..., un, T] for an n dimension system.

    condeq : integer, optional
        The equation in the system to use for the phase condition. For
        example, if a user wanted to use the second equation in the system, they
        would pass condeq=1. Defaults to 0

    cond : string or float, optional
        Phase condition value. Defaults to 'extrema' which starts the solution at it's closest extremum.

    params : optional keyword arguments, may contain a system parameters vector or a stepsize to use in the solve_to to create F

    Returns
    -------
    G : function
        The shooting discretisation function dependent on u0t
    """

    # needed for int phi
    if 'sys_params' in list(params.keys()):
        sys_params = params['sys_params']
    else:
        sys_params = {}

    # defines F function
    def F(u0t):
        u0 = u0t[:-1]
        t = u0t[-1]
        tl, vl = solve_to(func, 0, t, u0, **params)
        return vl[-1]

    # phase condition
    def phi(u0t):
        u0 = u0t[:-1]
        return u0[condeq] - cond

    # integral phase condition
    def int_phi(u0t):
        u0 = u0t[:-1]
        t = u0t[-1]
        grad = func(t, u0, **sys_params)[condeq]
        return grad

    # defines G function to solve which is now n+1 dim
    def G(u0t):
        u0 = u0t[:-1]
        if cond == 'extrema':
            return np.append([int_phi(u0t)], u0 - F(u0t))
        else:
            return np.append([phi(u0t)], u0 - F(u0t))
    return G

def shoot_root(func, u0, solver=fsolve, xtol=1e-20, plot=False, period_find_len=500, improve_guess=False, **params):
    """
    A function that uses numerical shooting to find limit cycles of
    a specified ODE.

    Parameters
    ----------
    func : function
        The ODE system to apply shooting to.

    u0 : numpy.array
        A guess at the initial values for the limit cycle.

    solver : function, optional
        The solving routine to use to solve the shooting problem. Default is scipy's fsolve

    xtol : float, optional
        The root-finding calculation will terminate if the relative error
        between two consecutive iterates is at most xtol. Defaults to 1e-20

    plot : boolean, optional
        Decides whether or not the limit cycle across its period will be plotted with respect to time after solving, default False

    period_find_len : integer or float, optional
        When determining an initial period guess, this value is used to solve across, default is 500

    improve_guess : boolean, optional
        After the system is initially solved to find the period, if improve_guess=True is passed, the initial guess passed into shooting will be chosen from the final value of the solve

    params : keyword args, optional
        contains other information that may impact processes in the solver

    Returns
    -------
    .ics : numpy array
        The initial conditions found by shooting

    .period : float
        The period found by shooting
    """

    tl, vl = solve_to(func, 0, period_find_len, u0, **params) # solve for t-guess

    if improve_guess == True: # grab better ics if requested
        u0 = vl[-1]
        print('We have a better guess. We\'ll start with ', u0)

    orbit = isolate_orbit(tl, vl[:,0], peak_tol=1e-3) # find t_guess
    t_guess = orbit.period

    if u0[0] > orbit.max_height or u0[0] < orbit.min_height:
        print('\nWARNING: Initial guess may be out of system limit cycle range and could lead to incorrect shooting. The first equation of the system appears to oscillate between ', np.round(orbit.min_height, 3), ' and ', np.round(orbit.max_height, 3), '.\n')

    u0t = np.append(u0, [t_guess])

    discret = shooting(func, u0t, **params)
    shot = solver(discret, u0t, xtol=xtol)

    # print('\nConvergence check: ', np.round(discret(shot), 12))

    if False in np.isclose(discret(shot), np.array([.0, .0, .0]), atol=1e-12):
        print('WARNING: Numerical root finder has not convereged. Try using different initial conditions.\nAlternatively, you may have entered an invalid phase condition or there are no limit cycles to be found here.')
        exit()

    sol = Shot_Sol(shot[:-1], shot[-1])

    if sol.period < 0:
        print('WARNING: Negative period found. Try shooting from a different initial guess.')
        exit()

    if np.isclose(sol.period, orbit.period, rtol=.05) == False:
        print('WARNING: Periods found by period finder and shooting differ by more than 5 percent. Consider a different initial condition guess or simply pass improve_guess=True.')
        #exit()

    if False not in np.isclose(sol.ics, np.zeros_like(sol.ics), atol=1e-15):
        print('WARNING: the calculated initial conditions seem quite close to zero. Perhaps set a phase condition or set improve_guess=False for this system.')

    if plot == True:
        tl, vl = solve_to(func, 0, sol.period, sol.ics, plot=True, **params)

    return sol
#################    NUMERICAL CONTINUATION     ###############################
def param_discretise(func, x, **params):

    # A function which constrains a function at a particular parameter. Works similar to lambda but is simpler

    # BUG TO FIX:
    # Won't work if user inputs extra args, can be handled by analysing function inputs

    # we need this to remove other args
    if 'sys_params' in list(params.keys()):
        sys_params = params['sys_params']
    else:
        sys_params = {}

    def new_func(x):
        return func(x, **sys_params)
    return new_func

def continuation(func, u0, par0, vary_par, end_par, step=.1, max_steps=100, discretisation=shooting, solver=fsolve, xtol=1e-12, first_shoot_deltat=.01, contin_deltat=.1, root_change_tol=1e-15, plot=False, print_progress=False, **params):

    """
    A function that performs natural parameter root-finding continuation.

    Parameters
    ----------
    func : function
        The ODE system to apply continuation to.

    u0 : numpy.array
        A guess at the initial values for the first solution.

    par0 : float
        Initial value of the param to vary.

    vary_par : string
        The name of the paramter to vary

    end_par : float
        Continuation will terminate if the varied parameter reaches this value

    step : float, optional
        The step to change the parameter by on each iteration (can be negative), default .1

    max steps : integer, optional
        The maximum steps by which to increment the parameter, default 100

    discretisation : function, optional
        The function which creates the root-finding problem from the input function

    solver : function, optional
        The solving routine to use to solve the shooting problem. Default is scipy's fsolve

    xtol : float, optional
        The root-finding calculation will terminate if the relative error
        between two consecutive iterates is at most xtol. Defaults to 1e-8

    plot : boolean, optional
        Decides whether or not the limit cycle across its period will be plotted with respect to time after solving, default False

    print_progress : boolean, optional
        When set to true, the function will show the steps being taken, default False

    first_shoot_deltat : float, optional
        The deltat to be used in the first root-find. May need to be small to ensure correct init conds are used. Default .01

    contin_deltat : float, optional
        The deltat to be used in the continuation solves. May be bigger than first_shoot_deltat to decrease computation. Default .1

    root_change_tol : float, optional
        The maximum absolute error used to check if the shooting solve has converged. Default 1e-15

    Returns
    -------
    par_vals : numpy array
        An array of parameter values across the continuation

    ics : numpy array
        An array of initial conditions for each parameter value
    """

    # initialise lists to return
    par_vals = []
    ics = []

    dps = len(str(step)) # workaround for step bug

    # create system params dict if not already exists
    if 'sys_params' not in list(params.keys()):
        params['sys_params'] = {}

    # grab an initial guess for shooting if func is periodic:
    if discretisation == shooting:
        # the shoot root here is needed as it is more capable at debugging than the later method for the first root--- arguably more important therefore
        sol  = shoot_root(func, u0, deltat=first_shoot_deltat, **params)
        u0t = np.append(sol.ics, [sol.period])
        start = 1
        # add the first solutions to the lists
        par_vals.append(params['sys_params'][vary_par])
        ics.append(np.append(sol.ics, [sol.period]))
        params['sys_params'][vary_par] += step
    else:
        u0t = u0
        start = 0

    start_time = datetime.now()
    # looping through
    for i in range(start, max_steps):
        # discretise for this parameter value
        discret = discretisation(func, u0t, deltat=contin_deltat, **params)
        saved_u0t = u0t
        # solve the discretisation
        sol = solver(discret, u0t, xtol=xtol)
        # add the solutions to the lists
        par_vals.append(params['sys_params'][vary_par])
        ics.append(sol)

        # check the new sol isn't the same as the last one; if it is, we stop continuating
        if False not in np.isclose(saved_u0t, sol, atol=root_change_tol):
            print('\nRoot no longer changing at param value ', params['sys_params'][vary_par])
            break
        else:
            u0t = sol # let the new initial guess be the current solution
            # increment the value of the parameter by one step
            params['sys_params'][vary_par] = np.round(params['sys_params'][vary_par] + step, dps)
            # print a debug screen if they want
            if print_progress == True:
                print('sol found: ', sol)
                print('New param: ', params['sys_params'][vary_par])
            else:
                # or a progress bar thing
                print(str(datetime.now()-start_time)[2:11], '    Step:', i+1, '/', max_steps, '    Paramater value:', np.round(params['sys_params'][vary_par], 5), end='\r')

    # we prefer numpy arrays to lists really
    par_vals = np.array(par_vals)
    ics = np.array(ics)

    # plot if required
    if plot == True:
        if discretisation == shooting:
            plt.plot(par_vals, ics[:, :-1])
        else:
            plt.plot(par_vals, ics)
        plt.xlabel('Parameter')
        plt.ylabel('Roots')
        plt.grid()
        plt.show()

    return par_vals, ics

def arclen_discret(func, u0tp, u0tp_guess, secant, vary_par, discretisation=shooting, **params):

    """
    A function that discretises a function to a pseudo-arclength root-finding problem.

    Parameters
    ----------
    func : function
        The ODE system to apply shooting to.

    u0tp : numpy.array
        A guess at the initial values for the limit cycle and a guess at its period and closest valid paramter in the form [u0, ..., un, T, param] for an n dimension system. This will change

    u0tp_guess : numpy.array
        An guess at the initial values for the limit cycle and a guess at its period and closest valid paramter in the form [u0, ..., un, T, param] for an n dimension system. This will NOT change

    secant : integer
        The difference between the last two known solutions

    vary_par : string
        The name of the parameter to vary

    discretisation : function, optional
        The function that discretises the system, default is shooting

    params : optional keyword arguments, may contain a system parameters vector or a stepsize to use in the solve_to to create F

    Returns
    -------
    root_find_func : function
        The root-finding function dependent on u0tp
    """

    def root_find_func(u0tp):
        params['sys_params'][vary_par] = u0tp[-1]
        disc = discretisation(func, u0tp[:-1], **params)
        return np.append(disc(u0tp[:-1]), [np.dot((u0tp - u0tp_guess), secant)])

    return root_find_func

def pseudo_arc_contin(func, u0, par0, vary_par, end_par, step=.1, max_steps=100, discretisation=shooting, solver=fsolve, xtol=1e-12, first_shoot_deltat=.01, contin_deltat=.1, check_shoot=False, plot=False, print_progress=False, **params):

    """
    A function that performs natural parameter root-finding continuation.

    Parameters
    ----------
    func : function
        The ODE system to apply continuation to.

    u0 : numpy.array
        A guess at the initial values for the first solution.

    par0 : float
        Initial value of the param to vary.

    vary_par : string
        The name of the paramter to vary

    end_par : float
        Continuation will terminate if the varied parameter reaches this value

    check_shoot : boolean, optional
        If the user wishes to check shooting is being applied correctly, set to true for a plot of the first limit cycle

    step : float, optional
        The step to change the parameter by on each iteration (can be negative), default .1

    max steps : integer, optional
        The maximum steps by which to increment the parameter, default 100

    discretisation : function, optional
        The function which creates the root-finding problem from the input function

    solver : function, optional
        The solving routine to use to solve the shooting problem. Default is scipy's fsolve

    xtol : float, optional
        The root-finding calculation will terminate if the relative error
        between two consecutive iterates is at most xtol. Defaults to 1e-12

    plot : boolean, optional
        Decides whether or not the limit cycle across its period will be plotted with respect to time after solving, default False

    print_progress : boolean, optional
        When set to true, the function will show the steps being taken, default False

    first_shoot_deltat : float, optional
        The deltat to be used in the first root-find. May need to be small to ensure correct init conds are used. Default .01

    contin_deltat : float, optional
        The deltat to be used in the continuation solves. May be bigger than first_shoot_deltat to decrease computation. Default .1

    root_change_tol : float, optional
        The maximum absolute error used to check if the shooting solve has converged. Default 1e-15

    Returns
    -------
    par_vals : numpy array
        An array of parameter values across the continuation

    ics : numpy array
        An array of initial conditions for each parameter value
    """

    # initialise lists to return
    points = []

    if discretisation == shooting:
        # the shoot root here is needed as it is more capable at debugging than the later method for the first root--- arguably more important therefore to get a good initial guess
        print('Shooting first root...', end='\r')
        sol  = shoot_root(func, u0, deltat=first_shoot_deltat, plot=check_shoot, **params)
        u0t = np.append(sol.ics, [sol.period]) # reformat to u0t
    else:
        u0t = u0 # period not necessary for non-shooting

    ######   GET FIRST TWO KNOWN SOLUTIONS   ##########################
    root_func = discretisation(func, u0t, deltat=contin_deltat, **params)
    res1 = solver(root_func, u0t, xtol=1e-20)
    res1 = np.append(res1, [params['sys_params'][vary_par]])
    points.append(res1)

    params['sys_params'][vary_par] += step # increment parameter once

    root_func = discretisation(func, res1[:-1], deltat=contin_deltat, **params)
    u0t = solver(root_func, u0t, xtol=1e-20)
    u0tp = np.append(u0t, [params['sys_params'][vary_par]])
    points.append(u0tp)

    # progress bar stuff
    if print_progress == True:
        print('First known solution: ', res1)
        print('\nSecond known solution:   ', u0t)

    ###################################################################

    # grab secant and next guess
    secant = u0tp - res1
    u0tp_guess = u0tp + secant

    # progress bar stuff
    if print_progress == True:
        print('First Secant:            ', secant)
        print('So next guess:           ', u0tp_guess)

    params['sys_params'][vary_par] = u0tp_guess[-1] # change parameter to next guess

    start_time = datetime.now() # Measure the time

    for i in range(max_steps):

        # check we haven't hit the second boundary
        if (params['sys_params'][vary_par] < end_par and step > 0) or (params['sys_params'][vary_par] > end_par and step < 0):

            # create the new root-finding problem
            root_func = arclen_discret(func, u0tp, u0tp_guess, secant, vary_par, discretisation=discretisation, deltat=contin_deltat, **params)
            # solve it
            new = solver(root_func, u0tp_guess, xtol=xtol)
            points.append(new) # add to points list
            secant = new - u0tp # find new secant
            next_guess = new + secant # create next guess

            # progress bar stuff
            if print_progress == True:
                print('Sol found:               ', new)
                print('Therefore the secant is: ', secant)
                print('So the next guess is:    ', next_guess)

            u0tp = new # the next guess is what we just found
            u0tp_guess = next_guess # set next guess
            params['sys_params'][vary_par] = u0tp_guess[-1] # increment parameter

            if print_progress == False:
                # or a progress bar thing
                print(str(datetime.now()-start_time)[2:11], '    Step:', i+1, '/', max_steps, '    Paramater value:', np.round(params['sys_params'][vary_par], 5), end='\r')

    points = np.array(points) # we prefer np arrays
    par_vals, ics = points[:, -1], points[:, :-1] # split into par vals and ics

    if plot == True: # plot if user wants to
        if discretisation == shooting:
            plt.plot(par_vals, ics[:, :-1])
        else:
            plt.plot(par_vals, ics)
        plt.xlabel('Param')
        plt.ylabel('Root')
        plt.show()

    return par_vals, ics
###############         ERROR FINDER    #######################################
def get_abs_err_av(tl, sol, func, sys_params={}, **kwargs):

    """
    A function that finds the absolute error of a solve versus the analytic solution.

    Parameters
    ----------
    tl : numpy array
        The solve's time vector

    sol : function
        The state vectors through time after solving

    sys_params : dict, optional
        Parameters to apply to the analytic function, default empty

    kwargs : optional keyword arguments
        Extra keyword args to be ignored

    Returns
    -------
    abs_err : float
        The absolute error of the solve
    """

    errors = []
    for i in range(len(tl)):
        errors.append(abs(func(tl[i]) - sol[i]))
    abs_err = np.mean(errors)
    return abs_err
#####################       PDES         ######################################
def tridiagonal(n, lmbda, direction='cn', **kwargs):

    """
    A function that creates a sparse tridiagonal matrix for solving a partial differential equation.

    Parameters
    ----------
    n : positive integer
        The size of the tridiagonal will be nxn

    lmbda : float
        The lambda value to be used to construct the tridiagonal matrix

    direction : string, optional
        The scheme to use to create the tridiagonal matrix, default cn:
        'cn' - Crank Nicholson
        'fe' - Forward Euler
        'be' - Backward Euler

    kwargs : optional keyword arguments
        Extra keyword args to be ignored

    Returns
    -------
    Ae : matrix
        Tridiagonal matrix euler

    -----   or, for Crank Nicholson:   -----

    A : matrix
        LHS Tridiagonal matrix cn

    B : matrix
        RHS Tridiagonal matrix cn
    """

    offset = [-1,0,1] # needed to create tridiagonals

    if direction == 'cn':
        # create the two patterns
        k1 = np.array([-.5*lmbda*np.ones(n-1), (1+lmbda)*np.ones(n), -.5*lmbda*np.ones(n-1)])
        k2 = np.array([.5*lmbda*np.ones(n-1), (1-lmbda)*np.ones(n), .5*lmbda*np.ones(n-1)])

        # construct matrices
        Acn = csr_matrix(diags(k1,offset).toarray()) # have to be csr for spsolve
        Bcn = csr_matrix(diags(k2,offset).toarray())
        return Acn, Bcn
    else:
        if direction == 'fe':
            # create the forward Euler pattern
            k = np.array([lmbda*np.ones(n-1), (1-2*lmbda)*np.ones(n), lmbda*np.ones(n-1)])

        elif direction == 'be':
            # create the backward Euler pattern
            k = np.array([-1*lmbda*np.ones(n-1), (1+2*lmbda)*np.ones(n), -1*lmbda*np.ones(n-1)])

        # construct matrix
        Ae = diags(k, offset).toarray()
        return Ae

def var_tridiagonal(n, kappa, x, deltat, deltax, direction='be', **kwargs):

    """
    A function that creates a sparse tridiagonal matrix with a varying diffusion coefficient for solving a partial differential equation.

    Parameters
    ----------
    n : positive integer
        The size of the tridiagonal will be nxn

    kappa : function
        function for kappa w.r.t time

    x : integer
        mesh points in space

    deltat : float
        Step size in time domain

    deltax : float
        Step size in space domain

    direction : string, optional
        The scheme to use to create the tridiagonal matrix, default fe:
        'fe' - Forward Euler
        'be' - Backward Euler

    kwargs : optional keyword arguments
        Extra keyword args to be ignored

    Returns
    -------
    Ae : matrix
        Tridiagonal matrix euler
    """

    lmbda = deltat / (deltax**2) # calculate lambda

    # switch sign if backward euler
    if direction == 'be':
        lmbda = -1 * lmbda

    Ae = [] # create list of vectors

    for i in range(n):
        new_row = np.zeros(n)
        one = kappa(x[i] - deltax/2)
        two = kappa(x[i] + deltax/2)
        if i == 0:
            new_row[0] = 1 - lmbda * (one + two)
            new_row[1] = lmbda * two
        elif i == n-1:
            new_row[-2] = lmbda * one
            new_row[-1] = 1 - lmbda * (one + two)
        else:
            new_row[i-1] = lmbda * one
            new_row[i] = 1 - lmbda * (one + two)
            new_row[i+1] = lmbda * two
        Ae.append(new_row)

    return np.array(Ae)

def solve_diffusion_pde(u_j, x, t, mx, kappa, L, T, ext=None, bounds=(0,0), neu_bounds=(0,0), direction='cn', rhs_fn=None, **kwargs):

    """
    A function that solves a function by a given method between two bounds given a set of initial conditions

    Parameters
    ----------

    u_j : numpy array
        Numpy array that describes the initial heat distribution

    x : numpy array
        spacial domain linspace

    t : numpy array
        time domain linspace

    mx : integer
        number of points in spacial domain

    kappa : float or function
        diffusion coefficient or function that describes variable diffusion coefficient

    L : float
        length of spatial domain

    T : float
        total time to solve for

    ext : string, optional
            extension of pde method, default is None

    bounds : tuple of size (2, 1), floats, optional
        the boundary values of the solve in order (t_0, t_end), default (0, 0)

    neu_bounds : tuple of size (2, 1), floats, optional
        the boundary values of the solve in order (t_0, t_end), default (0, 0)

    Returns
    -------

    u_jp1: numpy array
        The next state vector

    """


    deltax = x[1] - x[0] # gridspacing in x
    deltat = t[1] - t[0] # gridspacing in t

    lmbda = kappa*deltat/(deltax**2)    # mesh fourier number
    extra = np.zeros(mx+1)

    if ext == 'NHD': # non-homo-dirichlet
        A = tridiagonal(mx+1, lmbda, direction)

        r_vec = np.zeros(u_j.size)
        r_vec[0] = bounds[0]
        r_vec[-1] = bounds[1]

        extra = lmbda * r_vec

    elif ext == 'NEU': # Neuman
        A = tridiagonal(mx+1, lmbda, direction)
        A[0, 1] = A[0, 1] * 2
        A[-1, -2] = A[-1, -2] * 2

        u_j[0] = bounds[0]
        u_j[-1] = bounds[1]

        neu = np.zeros(mx+1)
        neu[0] = neu_bounds[0]
        neu[-1] = neu_bounds[1]

        extra = 2 * lmbda * deltax * neu

    elif ext == 'periodic': # Periodic boundary condition
        A = tridiagonal(mx+1, lmbda, direction)
        A[0, -1] = A[-1, 0] = A[0, 1]
        u_j[0] = bounds[0]
        u_j[-1] = bounds[1]

    elif ext == 'var_coef': # varied coefficient
        A = var_tridiagonal(mx+1, kappa, x, deltat, deltax)
    else:
        A = tridiagonal(mx+1, lmbda, direction)

    if rhs_fn != None:
        rhs = np.zeros(mx+1)
        for i in range(0, mx+1):
            rhs[i] = rhs_fn(x[i])
        extra = deltat * rhs

    if direction == 'cn':
        # redefine if necessary to get two matrices
        A, B = tridiagonal(mx+1, lmbda, direction)
        # then use the spsolve
        return spsolve(A, B @ u_j + extra)
    else:
        return A @ u_j + extra

def steady_state(u_j, mx, nt, kappa, L, T_start, step_size=.01, tol=1e-2, max_steps=200, **kwargs):

    step = 0
    T = T_start

    while step < max_steps:
        print('step: ', step+1, 'T: ', T, end='\r')

        x = np.linspace(0, L, mx+1) # mesh points in space
        t = np.linspace(0, T, nt+1) # mesh points in time

        u_jp1 = solve_diffusion_pde(u_j, x, t, mx, nt, kappa, L, T, **kwargs)

        if False in np.isclose(u_j, u_jp1, rtol=tol):
            T += step_size
            u_j = u_jp1
            step += 1
        else:
            break

    return T
