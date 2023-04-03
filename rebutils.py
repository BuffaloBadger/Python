import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
from scipy.stats.distributions import t
import statsmodels.api as sm
import math

def fitNLSR(beta_guess, x, y_meas, calcY, use_rel_error):
    # set the weights
    if use_rel_error:
        weight = y_meas
    else:
        weight = np.ones(len(y_meas))
    
    # estimate the parameters
    beta, beta_cov, info, mesg, ier = curve_fit(calcY, x, y_meas, p0=beta_guess,
            sigma=weight, method = 'trf', full_output=True)

    # calculate r_squared
    y_pred = info['fvec'] + y_meas
    y_mean = np.mean(y_meas)
    ss_res = np.sum(np.square(y_meas - y_pred))
    ss_tot = np.sum(np.square(y_meas - y_mean))
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

def fitLinSR(x, y, model_has_intercept):
    # define x
    if model_has_intercept:
        x=sm.add_constant(x)
    
    # defing the model and fit it to the data
    model = sm.OLS(y,x)
    res = model.fit()

    # return the results
    return res.params, res.conf_int(alpha=0.05), res.rsquared

def fit_Arrhenius_model(k,T,R):
    # T must be in absolute units and R must be in terms of the same temperature
    # units and whatever energy units are desired.

    # define x and y
    x = -1/R/T
    x = sm.add_constant(x)
    y = np.log(k)

    # define the model and fit it to the data
    model = sm.OLS(y,x)
    res = model.fit()

    # get the raw results
    beta = res.params
    r_squared = res.rsquared
    beta_ci = res.conf_int(alpha=0.05)

    # process the results
    beta[0] = np.exp(beta[0])
    beta_ci[0,0] = np.exp(beta_ci[0,0])
    beta_ci[0,1] = np.exp(beta_ci[0,1])

    # return the results
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
    
