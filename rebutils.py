import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
from scipy.stats.distributions import t
import math

def fitNLSR(beta_guess, x, y_meas, calcY, use_rel_error):
    # set the weights
    if use_rel_error:
        weight = y_meas
    else:
        weight = np.ones(len(y_meas))
    
    # estimate the parameters
    beta, beta_cov, info, mesg, ier = curve_fit(calcY, x, y_meas, p0=beta_guess, sigma=weight,
        full_output=True)

    # calculate r_squared
    y_pred = info['fvec']
    # I originally had the following line; don't know where it came from, but it
    # seems to be wornk
    #y_pred = info['fvec']*y_meas + y_meas
    y_mean = np.mean(y_meas)
    ss_res = np.sum((y_meas - y_pred)**2)
    ss_tot = np.sum((y_meas - y_mean)**2)
    r_squared = 1 - ss_res/ss_tot

    # calculate 95% confidence interval
    beta_ci = np.zeros((len(beta),2))
    alpha = 0.05
    dof = max(0, len(y_meas) - len(beta))
    t_val = t.ppf(1.0 - alpha/2., dof)
    for i, p in enumerate(beta):
        beta_ci[i,0] = beta[i] - beta_cov[i,i]**0.5*t_val
        beta_ci[i,1] = beta[i] + beta_cov[i,i]**0.5*t_val
    return beta, beta_ci, r_squared

def fit_Arrhenius_expression_to_data(k, T, R):
    # T must be in absolute units and R must be in terms of the same temperature
    # units and whatever energy units are desired.

    # get the number of data
    n_data = len(k)

    # create a matrix with x (-1/R/T) in the first column and 1's in the second
    x = np.transpose(np.array([-1/R/T, np.ones(n_data)]))

    # create an array with ln(k)
    y_meas = np.log(k)

    # calculate the parameters
    t1 = np.transpose(x)
    x1 = np.linalg.inv(np.matmul(t1,x))
    x2 = np.matmul(x1,t1)
    beta = np.matmul(x2,y_meas)

    # calculate the model-predicted responses
    y_pred = np.matmul(x,beta)

    # calculate r_squared
    y_mean = np.mean(y_meas)
    ss_res = np.sum((y_meas - y_pred)**2)
    ss_tot = np.sum((y_meas - y_mean)**2)
    r_squared = 1 - ss_res/ss_tot

    # calculate the 95% confidence intervals
    beta_ci = np.zeros((len(beta),2))
    eps = y_meas - y_pred
    var_eps = 1/(n_data - 2)*(np.matmul(np.transpose(eps),eps))
    covar_beta = var_eps*np.linalg.inv(np.matmul(t1,x))
    t_critical = t.ppf(0.975,n_data - 2)
    for i, p in enumerate(beta):
        beta_ci[i,0] = beta[i] - covar_beta[i,i]**0.5*t_critical
        beta_ci[i,1] = beta[i] + covar_beta[i,i]**0.5*t_critical

    # convert from ln(k0) to k0
    beta[1] = math.e**(beta[1])
    beta_ci[1,0] = math.e**(beta_ci[1,0])
    beta_ci[1,1] = math.e**(beta_ci[1,1])
    return beta, beta_ci, r_squared

def solveIVODEs(ind0, dep0, f_var, f_val, ODE_fcn, odes_are_stiff=False):
    # define an event in case the final value of a dependent variable is known
    def event(ind, dep):
        return dep[f_var - 1] - f_val
    event.terminal = True
    
    if f_var == 0:
        # do not use the event
        ind = (ind0, f_val)
        if odes_are_stiff:
            res = solve_ivp(ODE_fcn, ind, dep0, method='LSODA')
        else:
            res = solve_ivp(ODE_fcn, ind, dep0, method='RK45')
    else:
        # use the event
        solved = False
        count = 0
        ind_f = ind0 + 1.0
        while (count < 10 and not solved):
            ind = (ind0, ind_f)
            count += 1
            if odes_are_stiff:
                res = solve_ivp(ODE_fcn, ind, dep0, method='LSODA', events=event)
            else:
                res = solve_ivp(ODE_fcn, ind, dep0, method='RK45',events=event)
            if res.t[-1] == ind_f: # ind_f was not large enough
                ind_f = (ind0 + 1.0)*10**count
            elif res.t[-1] < 0.1*ind_f: # ind_f was too large
                ind_f = (ind0 + 1.0)/10**count
            else:
                solved = True
    return res
    
