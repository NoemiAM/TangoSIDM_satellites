import time
import numpy as np
import emcee
from multiprocessing import Pool
from scipy.integrate import odeint
from scipy.interpolate import interp1d


def log_prior_core_nfw(theta):
    """
    The natural logarithm of the prior probability.
    It sets prior to 1 (log prior to 0) if params are in range, and zero (-inf) otherwise.
    Args: theta (tuple): a sample containing individual parameter values
    """
    log10_M200, rc, n = theta
    log_prior = -np.inf
    if 7 < log10_M200 < 12 and 0.01 < rc < 10 and 0 < n < 1 : log_prior = 0.0
    return log_prior


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


def log_posterior_core_nfw(theta, x, y, yerr):
    """
    The natural logarithm of the joint posterior.
    Args:
        theta (tuple): a sample containing individual parameter values
        x (array): values over which the data/model is defined
        y (array): the set of data
        yerr (array): the standard deviation of the data points
    """
    lp = log_prior_core_nfw(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_core_nfw(theta, x, y, yerr)


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


def log_likelihood_core_nfw(theta, x, y, yerr):
    """
    The natural logarithm of the joint likelihood.
    Args:
        theta (tuple): a sample containing individual parameter values
        x (array): values over which the data/model is defined
        y (array): the set of data
        yerr (array): the standard deviation of the data points
    """
    log10_M200, rc, n = theta
    model = fit_core_nfw_model(x, log10_M200, rc, n)
    sigma2 = yerr**2
    log_l = -0.5 * np.sum((y - model) ** 2 / sigma2)
    return log_l


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


def c_M_relation(log10_M0):
    """
    Concentration-mass relation from Correa et al. (2015).
    This relation is most suitable for Planck cosmology.
    """
    z = 0
    # Best-fit params:
    alpha = 1.7543 - 0.2766 * (1. + z) + 0.02039 * (1. + z) ** 2
    beta = 0.2753 + 0.0035 * (1. + z) - 0.3038 * (1. + z) ** 0.0269
    gamma = -0.01537 + 0.02102 * (1. + z) ** (-0.1475)

    log_10_c200 = alpha + beta * log10_M0 * (1. + gamma * log10_M0 ** 2)
    c200 = 10 ** log_10_c200
    return c200


def calculate_R200(log10_M200):
    """
    General definition of R200 for z=0
    """
    z = 0
    Om = 0.309
    Ol = 1. - Om
    rhocrit0 = 2.775e11 * 0.6777 ** 2 / (1e3) ** 3  # Msun/kpc^3
    rho_crit = rhocrit0 * (Om * (1. + z) ** 3 + Ol)
    R200 = 10**log10_M200 / (4. * np.pi * 200 * rho_crit / 3.)
    R200 = R200 ** (1. / 3.)  # kpc
    return R200


def fit_core_nfw_model(xdata, log10_M200, rc, n):
    """
    CoreNFW profile introduced by Read et al. (2019).
    The free parameters correspond to the total halo mass, M200,
    defined as for an NFW profile. The core size radius, rc,
    and the logarithmic slope n
    """
    
    c = c_M_relation(log10_M200)
    R200 = calculate_R200(log10_M200)
    gc = 1./ (np.log(1. +c) - c/(1.+c))
    rhocrit = 2.775e11 * 0.6777 ** 2 / (1e3) ** 3  # Msun/kpc^3
    rhos = 200 * rhocrit * c**3 * gc / 3.
    rs =  R200 / c

    xrange = np.arange(-3, 3, 0.01)
    xrange = 10**xrange
    xrange = xrange / rs
    
    rho_nfw = rhos / (xrange * (1. + xrange)**2)
    M_nfw = 10**log10_M200 * gc * (np.log(1. + xrange) - xrange / (1. + xrange))
    f = (np.tanh(xrange * rs / rc))

    sol = f**n * rho_nfw
    sol +=  n * f**(n-1.) * (1.-f**2) * M_nfw / (4. * np.pi * (xrange * rs)**2 * rc)
    yrange = np.log10(sol)

    finterpolate = interp1d(xrange, yrange)
    x = xdata / rs

    max_x = 1e6
    if len(x) > 1: max_x = np.max(x)
    else: max_x = x
    if rs<=0 or max_x>1e5 : return 0

    ydata = finterpolate(x)
    f = ydata
    return f


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


def run_mcmc_core_nfw(x, y, yerr, soln):

    pos = soln.x + 1e-4 * np.random.randn(32, 3)
    nwalkers, ndim = pos.shape

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior_core_nfw, args=(x, y, yerr), pool=pool)
        start = time.time()
        sampler.run_mcmc(pos, 5000, progress=True)
        end = time.time()
        multi_time = end - start
        # print("Multiprocessing took {0:.1f} minutes".format(multi_time / 60))

    samples = sampler.get_chain(discard=100, thin=15, flat=True)
    log10_M200 = np.median(samples[:, 0])
    rc = np.median(samples[:, 1])
    n = np.median(samples[:, 2])
    sigma_log10_M200 = np.std(samples[:, 0])
    sigma_rc = np.std(samples[:, 1])
    sigma_n = np.std(samples[:, 2])

    # print("Mean autocorrelation time: {0:.3f} steps".format(np.mean(sampler.get_autocorr_time(quiet=True))))
    return sampler, log10_M200, rc, n, sigma_log10_M200, sigma_rc, sigma_n


def run_mcmc_iso(x, y, yerr, soln):

    pos = soln.x + 1e-4 * np.random.randn(32, 2)
    nwalkers, ndim = pos.shape

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior_iso, args=(x, y, yerr), pool=pool)
        start = time.time()
        sampler.run_mcmc(pos, 5000, progress=True)
        end = time.time()
        multi_time = end - start
        # print("Multiprocessing took {0:.1f} minutes".format(multi_time / 60))

    samples = sampler.get_chain(discard=100, thin=15, flat=True)
    r0 = np.median(samples[:, 0])
    rho0 = np.median(samples[:, 1])
    sigma_r0 = np.std(samples[:, 0])
    sigma_rho0 = np.std(samples[:, 1])

    # print("Mean autocorrelation time: {0:.3f} steps".format(np.mean(sampler.get_autocorr_time(quiet=True))))
    return sampler, r0, rho0, sigma_r0, sigma_rho0


def run_mcmc_nfw(x, y, yerr, soln):

    pos = soln.x + 1e-4 * np.random.randn(32, 2)
    nwalkers, ndim = pos.shape

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior_nfw, args=(x, y, yerr), pool=pool)
        start = time.time()
        sampler.run_mcmc(pos, 5000, progress=True)
        end = time.time()
        multi_time = end - start
        # print("Multiprocessing took {0:.1f} minutes".format(multi_time / 60))

    samples = sampler.get_chain(discard=100, thin=15, flat=True)
    r0 = np.median(samples[:, 0])
    rho0 = np.median(samples[:, 1])
    sigma_r0 = np.std(samples[:, 0])
    sigma_rho0 = np.std(samples[:, 1])

    # print("Mean autocorrelation time: {0:.3f} steps".format(np.mean(sampler.get_autocorr_time(quiet=True))))
    return sampler, r0, rho0, sigma_r0, sigma_rho0