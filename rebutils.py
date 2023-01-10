import numpy as np
from scipy.optimize import curve_fit
from scipy.stats.distributions import t

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
    y_pred = info['fvec']*y_meas + y_meas
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