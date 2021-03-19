import numpy as np
from scipy.optimize import fsolve

#################       STEP METHODS      #####################################
# inputs gradient function at 2, current vector x, y, current t, and the step
# size
def euler_step(func, t, v, h):
    v = v + h * func(t, v)
    t = t + h
    return t, v
def rk4_step(func, t, v, h):
    k1 = h * func(t, v)
    k2 = h * func(t + h/2, v + k1/2)
    k3 = h * func(t + h/2, v + k2/2)
    k4 = h * func(t + h, v + k3)
    v = v + (k1 + 2*k2 + 2*k3 + k4) / 6
    t = t+h
    return t, v


################        SOLVERS         #######################################
# rescales dt to finish closer to the bound
def rescale_dt(t1, t2, deltat_max):
    if deltat_max % (t2-t1) != 0:
        deltat_max = (t2-t1) / np.ceil((t2-t1) / deltat_max)
    return deltat_max, np.round((t2-t1)/deltat_max)

# solves between two time bounds; t1 (start) and t2 (end); returns 2 lists in
# tuple
def solve_to(func, t1, t2, v, deltat_max_orig, method='RK4'):
    tl = [t1]
    vl = [v]
    # rescale dt
    deltat_max = rescale_dt(t1, t2, deltat_max_orig)[0]
    if method == 'Euler':
        # This cond is still necessary to cope with the rounding of step sizes
        # if they become irrational after rescaling
        while t1 < t2:
            if t1 + deltat_max <= t2:
                t1, v = euler_step(func, t1, v, deltat_max)
                vl.append(v)
                tl.append(t1)
            else:
                deltat_max = t2 - t1
                t1, x = euler_step(func, t1, v, deltat_max)
                vl.append(v)
                tl.append(t1)

    if method == 'RK4':
        while t1 < t2:
            if t1 + deltat_max <= t2:
                t1, v = rk4_step(func, t1, v, deltat_max)
                vl.append(v)
                tl.append(t1)
            else:
                deltat_max = t2 - t1
                t1, v = rk4_step(func, t1, v, deltat_max)
                vl.append(v)
                tl.append(t1)
    #print('Final sol by ', method, ' with stepsize ', deltat_max_orig, ': ', vl[-1][0])
    return tl, vl

# gives sols from t = 0, t = 1, t = 2, ...  t = t as a list
# depending on a list of stepsizes fed in
# works for 1-d ODE but also n-dimension ODEs
# method defaults to RK4, func should return all first order derivatives
def solve_ode(func, t1, t2, v0, stepsizes, method='RK4'):
    tls = []
    sols = []
    for size in stepsizes:
        tl, sol = solve_to(func, t1, t2, v0, size, method)
        tls.append(tl)
        sols.append(sol)
    return tls, sols


# Given a differential system, a period of the limit cycle, and a guess at some
# initial conditions, the function will return some initial conditions that
# lead the system straight into a limit cycle with the required period

# User may also select a change tolerance between iters to finish the root
# finding process
def shoot_ics(func, period, guess, xtol=1.0e-02):

    def F(t, guess):
        # Grab the last value of the solve
        return solve_to(func, 0, t, guess, 0.001)[1][-1]

    def G(guess):
        return guess - F(period, guess)

    ics = fsolve(G, guess, xtol=xtol)
    return ics

##################  PERIOD FINDER   #################################
def isolate_orbit(iv, dvs):
    # grab the first dep variable
    fdv = dvs[0]

    # initialise list of peak values and an index to trace them to
    peak_times = []
    index = []

    # initialise the list of start conds
    start_conds = []

    # loop through the dep var
    for i in range(1, len(iv)-1):

        # find the peaks and add them to the peak list
        if fdv[i] > fdv[i-1] and fdv[i] > fdv[i+1]:
            peak_times.append(iv[i])
            index.append(i)

    # Find the dep var values at the last index
    for dv in dvs:
        start_conds.append(dv[index[-1]])

    return peak_times[-1] - peak_times[-2], start_conds

###############         ERROR FINDER    #######################################
# finds average absolute error for an integrated solution
def get_abs_err_av(tl, sol, func):
    errors = []
    for i in range(len(tl)):
        errors.append(abs(func(tl[i]) - sol[i]))
    return np.mean(errors)
