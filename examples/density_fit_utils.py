import time
import numpy as np
import emcee
from multiprocessing import Pool
from scipy.integrate import odeint
from scipy.interpolate import interp1d


def log_prior_iso(theta):
    """
    The natural logarithm of the prior probability.
    It sets prior to 1 (log prior to 0) if params are in range, and zero (-inf) otherwise.
    Args: theta (tuple): a sample containing individual parameter values
    """
    r0, rho0 = theta
    log_prior = -np.inf
    if 0.001 < r0 < 10 and 3 < rho0 < 10 : log_prior = 0.0

    return log_prior


def log_prior_nfw(theta):
    """
    The natural logarithm of the prior probability.
    It sets prior to 1 (log prior to 0) if params are in range, and zero (-inf) otherwise.
    Args: theta (tuple): a sample containing individual parameter values
    """
    r0, rho0 = theta
    log_prior = -np.inf
    if 0.001 < r0 < 10 and 3 < rho0 < 10 : log_prior = 0.0

    return log_prior


def log_posterior_iso(theta, x, y, yerr):
    """
    The natural logarithm of the joint posterior.
    Args:
        theta (tuple): a sample containing individual parameter values
        x (array): values over which the data/model is defined
        y (array): the set of data
        yerr (array): the standard deviation of the data points
    """
    lp = log_prior_iso(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_iso(theta, x, y, yerr)


def log_posterior_nfw(theta, x, y, yerr):
    """
    The natural logarithm of the joint posterior.
    Args:
        theta (tuple): a sample containing individual parameter values
        x (array): values over which the data/model is defined
        y (array): the set of data
        yerr (array): the standard deviation of the data points
    """
    lp = log_prior_nfw(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_nfw(theta, x, y, yerr)


def log_likelihood_iso(theta, x, y, yerr):
    """
    The natural logarithm of the joint likelihood.
    Args:
        theta (tuple): a sample containing individual parameter values
        x (array): values over which the data/model is defined
        y (array): the set of data
        yerr (array): the standard deviation of the data points
    """
    r0, rho0 = theta
    model = fit_isothermal_model(x, r0, rho0)
    sigma2 = yerr**2
    log_l = -0.5 * np.sum((y - model) ** 2 / sigma2)
    return log_l


def log_likelihood_nfw(theta, x, y, yerr):
    """
    The natural logarithm of the joint likelihood.
    Args:
        theta (tuple): a sample containing individual parameter values
        x (array): values over which the data/model is defined
        y (array): the set of data
        yerr (array): the standard deviation of the data points
    """
    r0, rho0 = theta
    model = fit_nfw_model(x, r0, rho0)
    sigma2 = yerr**2
    log_l = -0.5 * np.sum((y - model) ** 2 / sigma2)
    return log_l


def diff_isothermal_equation(f,x,n):
    """
    Differential equation that describes the isothermal profile
    """
    y, z = f
    dfdx = [z,-(n+2)*(1./x)*z-n*(n+1)*(1./x**2)-(1./x**n)*np.exp(y)]
    return dfdx


def fit_isothermal_model(xdata, a, b):
    xrange = np.arange(-3, 3, 0.01)
    xrange = 10**xrange
    xrange = xrange / a
    y0 = [0, 0]
    n = 0

    sol = odeint(diff_isothermal_equation, y0, xrange, args=(n,))
    yrange = np.exp(sol[:, 0])
    yrange = np.log10(yrange)
    finterpolate = interp1d(xrange, yrange)
    x = xdata / a

    max_x = 1e6
    if len(x) > 1: max_x = np.max(x)
    if a<=0 or max_x>1e5 : return 0

    ydata = finterpolate(x)
    f = b + ydata
    return f


def fit_nfw_model(xdata, a, b):
    xrange = np.arange(-3, 3, 0.01)
    xrange = 10**xrange
    xrange = xrange / a

    sol = 1 / ( (xrange) * (1+xrange)**2 )
    yrange = np.log10(sol)

    finterpolate = interp1d(xrange, yrange)
    x = xdata / a

    max_x = 1e6
    if len(x) > 1: max_x = np.max(x)
    else: max_x = x
    if a<=0 or max_x>1e5 : return 0

    ydata = finterpolate(x)
    f = b + ydata
    return f


def run_mcmc_iso(x, y, yerr, soln):

    pos = soln.x + 1e-4 * np.random.randn(32, 2)
    nwalkers, ndim = pos.shape

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior_iso, args=(x, y, yerr), pool=pool)
        start = time.time()
        sampler.run_mcmc(pos, 5000, progress=True)
        end = time.time()
        multi_time = end - start
        print("Multiprocessing took {0:.1f} minutes".format(multi_time / 60))

    samples = sampler.get_chain(discard=100, thin=15, flat=True)
    r0 = np.median(samples[:, 0])
    rho0 = np.median(samples[:, 1])

    print("Mean autocorrelation time: {0:.3f} steps".format(np.mean(sampler.get_autocorr_time(quiet=True))))
    return sampler, r0, rho0


def run_mcmc_nfw(x, y, yerr, soln):

    pos = soln.x + 1e-4 * np.random.randn(32, 2)
    nwalkers, ndim = pos.shape

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior_nfw, args=(x, y, yerr), pool=pool)
        start = time.time()
        sampler.run_mcmc(pos, 5000, progress=True)
        end = time.time()
        multi_time = end - start
        print("Multiprocessing took {0:.1f} minutes".format(multi_time / 60))

    samples = sampler.get_chain(discard=100, thin=15, flat=True)
    r0 = np.median(samples[:, 0])
    rho0 = np.median(samples[:, 1])

    print("Mean autocorrelation time: {0:.3f} steps".format(np.mean(sampler.get_autocorr_time(quiet=True))))
    return sampler, r0, rho0