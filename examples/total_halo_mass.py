import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from pylab import *
import matplotlib.pyplot as plt
import os
import emcee
from scipy.optimize import minimize
from multiprocessing import Pool
import corner


def log_model(x, q, m):
    # Model: Power-law function
    # f_theta = 10^q r^m
    # log10 f_theta
    f = q + m * x
    return f

def log_prior(theta):
    """
    The natural logarithm of the prior probability.
    It sets prior to 1 (log prior to 0) if params are in range, and zero (-inf) otherwise.
    Args: theta (tuple): a sample containing individual parameter values
    """
    q, m = theta
    log_prior = -np.inf
    if -15 < q < 15 and -15 < m < 15 : log_prior = 0.0
    return log_prior

def log_likelihood(theta, x, y, xerr, yerr):
    """
    The natural logarithm of the joint likelihood.
    Args:
        theta (tuple): a sample containing individual parameter values
        x (array): values over which the data/model is defined
        y (array): the set of data
        xerr (array): the standard deviation of the data points
        yerr (array): the standard deviation of the data points
    """
    q, m = theta
    model = log_model(x, q, m)
    sigma2 = yerr**2 + (m * 10**model) **2 * (xerr / 10**x) **2
    ll = (10**y - 10**model) ** 2 / sigma2
    ll += np.log( 2. * np.pi * sigma2 )
    log_l = -0.5 * np.sum(ll)
    return log_l

def log_posterior(theta, x, y, xerr, yerr):
    """
    The natural logarithm of the joint posterior.
    Args:
        theta (tuple): a sample containing individual parameter values
        x (array): values over which the data/model is defined
        y (array): the set of data
        xerr (array): the standard deviation of the data points
        yerr (array): the standard deviation of the data points
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, xerr, yerr)

def run_mcmc(x, y, xerr, yerr, soln):

    pos = soln.x + 1e-4 * np.random.randn(32, 2)
    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(x, y, xerr, yerr))
    start = time.time()
    sampler.run_mcmc(pos, 5000, progress=True)
    end = time.time()
    multi_time = end - start
    print("Multiprocessing took {0:.1f} minutes".format(multi_time / 60))

    samples = sampler.get_chain(discard=100, thin=15, flat=True)
    q = np.median(samples[:, 0])
    m = np.median(samples[:, 1])
    qerrl = np.percentile(samples[:, 0], 16)
    qerrh = np.percentile(samples[:, 0], 84)
    merrl = np.percentile(samples[:, 1], 16)
    merrh = np.percentile(samples[:, 1], 84)
    print(q, q-qerrl, qerrh-q)
    print(m, m-merrl, merrh-m)

    print("Mean autocorrelation time: {0:.3f} steps".format(np.mean(sampler.get_autocorr_time(quiet=True))))
    return samples


def calcR200(r, r_s, M_s):
    rhocrit = 2.775e11 * 0.6777 ** 2 / (1e3) ** 3  # Msun/kpc^3
    c = r / r_s
    Yc = np.log(1. + c) - c / (1. + c)
    Y1 = np.log(2.) - 0.5
    M200_1 = (4. * np.pi / 3.) * 200. * rhocrit * r ** 3  # Msun
    M200_2 = M_s * Yc / Y1
    f = M200_1 - M200_2
    return f

def calc_Ms(r_s, rho_s):
    r = np.arange(-2, 2, 0.001)
    r = 10 ** r
    deltar = r[1:] - r[:-1]
    deltar = np.append(deltar, deltar[-1])
    rho = NFW_profile(r, rho_s, r_s)
    mass = 4. * np.pi * np.cumsum(rho * r ** 2 * deltar)
    interpolate = interp1d(r, mass)
    M_s = interpolate(r_s)
    return M_s

def calculate_rho_s(Vmax, Rmax):
    G = 6.674e-8 #unit [cm/g (cm/s)^2]
    Msun_in_g = 1.98847e33 #g
    kpc_in_cm = 3.086e21 #cm
    G /= (1e5)**2 # [cm/g (km/s)^2)]
    G *= Msun_in_g # [cm/Msun (km/s)^2)]
    G /= kpc_in_cm # [kpc/Msun (km/s)^2]
    rho_s = 1.721 * Vmax**2 / (G * Rmax**2)
    return rho_s

def calculate_r_s(Vmax, Rmax):
    r_s = 0.462 * Rmax
    return r_s

def calculate_error_r_s(Vmax, Rmax, delta_Vmax, delta_Rmax):
    r_s = calculate_r_s(Vmax, Rmax)
    e_r_s = r_s * delta_Rmax / Rmax
    return e_r_s

def calculate_error_rho_s(Vmax, Rmax, delta_Vmax, delta_Rmax):
    rho_s = calculate_rho_s(Vmax, Rmax)
    e_rho_s = (delta_Vmax / Vmax)**2 + (delta_Rmax / Rmax)**2
    e_rho_s = np.sqrt(e_rho_s) * rho_s
    return e_rho_s

def NFW_profile(r, rho_s, r_s):
    rho = np.log10(rho_s)
    rho -= np.log10( (r/r_s) * ((r/r_s) + 1.0)**2 )
    return 10**rho

def calculate_error_NFW_rho(r, Vmax, Rmax, delta_Vmax, delta_Rmax):
    r_s = calculate_r_s(Vmax, Rmax)
    rho_s = calculate_rho_s(Vmax, Rmax)
    e_rho_s = calculate_error_rho_s(Vmax, Rmax, delta_Vmax, delta_Rmax)
    e_r_s = calculate_error_r_s(Vmax, Rmax, delta_Vmax, delta_Rmax)

    rho = NFW_profile(r, rho_s, r_s)
    error_rho = (e_rho_s / rho_s)**2
    error_rho += ( (e_r_s / r_s) * ((3. + r_s / r) / (1. + r_s / r)) )**2
    error_rho *= rho**2
    error_rho = np.sqrt(error_rho)
    return error_rho


def calculate_M200(R200):
    rhocrit = 2.775e11 * 0.6777 ** 2 / (1e3) ** 3  # Msun/kpc^3
    M200 = 4. * np.pi * 200. * rhocrit * R200**3 / 3.
    return M200

def mass(rho, r):
    deltar = r[1:]-r[:-1]
    deltar = np.append(deltar,deltar[-1])
    Mass = 4. * np.pi * rho * deltar * r ** 2
    Mass = np.cumsum(Mass)
    return Mass

def rho_mean(mass, r):
    rho_m = 3. * mass / (4. * np.pi * r**3)
    return rho_m

def calculate_R200(rho_mean, r):
    rhocrit = 2.775e11 * 0.6777 ** 2 / (1e3) ** 3  # Msun/kpc^3
    function_interpolate = interp1d(rho_mean, r)

    if 200 * rhocrit > np.max(rho_mean):
        R200 = 1. # Dummy value
    else:
        R200 = function_interpolate(200 * rhocrit)
    return R200


def run_best_fit(r_p, rho_150pc, color):
    # Prepare arrays ..
    x = np.log10(r_p[0, :])
    xerr = r_p[1, :] ** 2 + r_p[2, :] ** 2
    xerr = np.sqrt(xerr)

    y = np.log10(rho_150pc[0, :] / 1e7)
    yerr = rho_150pc[1, :] ** 2 + rho_150pc[2, :] ** 2
    yerr = np.sqrt(yerr) / 1e7

    # Make initial fit ..
    np.random.seed(42)
    nl = lambda *args: -log_likelihood(*args)
    initial = np.array([2, -0.5])
    soln = minimize(nl, initial, args=(x, y, xerr, yerr))
    q, m = soln.x
    print('=======')
    print(q, m)

    samples = run_mcmc(x, y, xerr, yerr, soln)

    q = np.median(samples[:, 0])
    m = np.median(samples[:, 1])

    xrange = np.arange(np.log10(10), np.log10(500), 0.2)
    num_range = len(xrange)
    yrange = np.ones((2,num_range))
    for i in range(num_range):
        yrange[0, i] = np.percentile(1e7 * 10 ** log_model(xrange[i], samples[:, 0], samples[:, 1]), 16)
        yrange[1, i] = np.percentile(1e7 * 10 ** log_model(xrange[i], samples[:, 0], samples[:, 1]), 84)


    # Plot ..
    xrange = np.arange(np.log10(10), np.log10(500), 0.2)
    plt.fill_between(10 ** xrange, yrange[0,:], yrange[1,:], color=color, alpha=0.1)
    plt.plot(10 ** xrange, 1e7 * 10 ** log_model(xrange, q, m), '--', lw=1, color=color)

    return samples


GAIA_PATH = "../data/Gaia/"
Kaplinghat_data = pd.read_csv(GAIA_PATH+"Kaplinghat_2019.csv").to_numpy()

dSph = Kaplinghat_data[:,0]
dSph = dSph[::2]

Vmax = Kaplinghat_data[:,2]
le_Vmax = Kaplinghat_data[:,2] + Kaplinghat_data[:,3]
he_Vmax = Kaplinghat_data[:,2] + Kaplinghat_data[:,4]
NFW_Vmax = 10**Vmax[::2] # Only NFW fit data; [km/s] units
le_NFW_Vmax = NFW_Vmax - 10**le_Vmax[::2] # Only NFW fit data; [km/s] units
he_NFW_Vmax = 10**he_Vmax[::2] - NFW_Vmax # Only NFW fit data; [km/s] units

Rmax = Kaplinghat_data[:,5]
le_Rmax = Kaplinghat_data[:,6] + Kaplinghat_data[:,5]
he_Rmax = Kaplinghat_data[:,7] + Kaplinghat_data[:,5]
NFW_Rmax = 10**Rmax[::2] # Only NFW fit data; [kpc] units
le_NFW_Rmax = NFW_Rmax - 10**le_Rmax[::2]  # Only NFW fit data; [kpc] units
he_NFW_Rmax = 10**he_Rmax[::2] - NFW_Rmax  # Only NFW fit data; [kpc] units
he_NFW_Rmax /= 2 # Reducing uncertainty in Rmax by factor of 2
le_NFW_Rmax /= 2 # Reducing uncertainty in Rmax by factor of 2

num_satellites = len(NFW_Rmax)
M200 = np.zeros((3,num_satellites))
R200 = np.zeros((3,num_satellites))

rho_s = np.zeros(num_satellites)
r_s = np.zeros(num_satellites)
r_range = np.arange(-4, 4, 0.01) # Some radial range
r_range = 10**r_range # to kpc

for i in range(num_satellites):
    rho_s[i] = calculate_rho_s(NFW_Vmax[i], NFW_Rmax[i])
    r_s[i] = calculate_r_s(NFW_Vmax[i], NFW_Rmax[i])
    rho = NFW_profile(r_range, rho_s[i], r_s[i])
    mass_within_r = mass(rho, r_range)
    rho_within_r = rho_mean(mass_within_r, r_range)
    R200[0, i] = calculate_R200(rho_within_r, r_range)
    M200[0, i] = calculate_M200(R200[0, i])

    ## Dealing with error propagration..
    low_error_rho = rho - calculate_error_NFW_rho(r_range, NFW_Vmax[i], NFW_Rmax[i], le_NFW_Vmax[i], le_NFW_Rmax[i])
    mass_within_r = mass(low_error_rho, r_range)
    rho_within_r = rho_mean(mass_within_r, r_range)
    R200[1, i] = calculate_R200(rho_within_r, r_range)
    M200[1, i] = M200[0, i] - calculate_M200(R200[1, i])

    high_error_rho = rho + calculate_error_NFW_rho(r_range, NFW_Vmax[i], NFW_Rmax[i], he_NFW_Vmax[i], he_NFW_Rmax[i])
    mass_within_r = mass(high_error_rho, r_range)
    rho_within_r = rho_mean(mass_within_r, r_range)
    R200[2, i] = calculate_R200(rho_within_r, r_range)
    M200[2, i] = calculate_M200(R200[2, i]) - M200[0, i]

    # Alternative method for NFW:
    # M_s = calc_Ms(r_s[i], rho_s[i])
    # R200[i] = fsolve(calcR200, 100., args=(r_s[i], M_s))
    # M200[i] = calculate_M200(R200b[i])


# We remove CVnI
num_satellites -= 1
dSph = dSph[:-1]

r_p_Isolated = np.zeros((3,num_satellites)) # Reading pericenter assuming isolated MW
r_p_LMC = np.zeros((3,num_satellites))      # Reading pericenter assuming MW perturbed by LMC
rho_150pc_K19 = np.zeros((3,num_satellites)) # Reading Kaplinghat+ 2019 determination under NFW


for i, dSphi in enumerate(dSph):
    filename = "Cardona_2023_"+dSphi+".csv"
    if not os.path.exists(GAIA_PATH + filename): continue
    data = pd.read_csv(GAIA_PATH + filename).to_numpy()
    r_p_Isolated[0,i] = data[2,0] # Pericenter [kpc]
    r_p_Isolated[1,i] = data[2,1] # (Error) Pericenter [kpc]
    r_p_Isolated[2,i] = data[2,2] # (Error) Pericenter [kpc]
    r_p_LMC[0,i] = data[3,0] # Pericenter [kpc]
    r_p_LMC[1,i] = data[3,1] # Pericenter [kpc]
    r_p_LMC[2,i] = data[3,2] # Pericenter [kpc]
    rho_150pc_K19[0,i] = data[8,0] * 1e7 # Central density [Msun/kpc^3]
    rho_150pc_K19[1,i] = data[8,1] * 1e7 # Central density [Msun/kpc^3]
    rho_150pc_K19[2,i] = data[8,2] * 1e7 # Central density [Msun/kpc^3]




#################
# Plot parameters
params = {
    "font.size": 10,
    "font.family": "Times",
    "text.usetex": True,
    "figure.figsize": (5, 2.5),
    "figure.subplot.left": 0.1,
    "figure.subplot.right": 0.95,
    "figure.subplot.bottom": 0.16,
    "figure.subplot.top": 0.95,
    "figure.subplot.wspace": 0.3,
    "figure.subplot.hspace": 0.3,
    "lines.markersize": 2,
    "lines.linewidth": 1.5,
}
plt.rcParams.update(params)

plt.figure()
ax = plt.subplot(1, 2, 1)
plt.grid(linestyle='-', linewidth=0.3)
plt.errorbar(r_p_Isolated[0,:], rho_150pc_K19[0,:], xerr=r_p_Isolated[1:,:], yerr=rho_150pc_K19[1:,:],
         marker='o', markersize=3.5, markeredgecolor="none", ls='none', lw=0.5, c='tab:blue',label='Isolated MW')

plt.errorbar(r_p_LMC[0,:], rho_150pc_K19[0,:], xerr=r_p_LMC[1:,:], yerr=rho_150pc_K19[1:,:],
         marker='v', markersize=3.5, markeredgecolor="none", ls='none', lw=0.5, c='crimson',label='MW + LMC')

samples_isolated = run_best_fit(r_p_Isolated, rho_150pc_K19, 'tab:blue')
samples_LMC = run_best_fit(r_p_LMC, rho_150pc_K19, 'crimson')

plt.legend(loc=[0.01, 0.01], labelspacing=0.05,
           handlelength=0.7, handletextpad=0.1,
           frameon=True, fontsize=11, ncol=1)

plt.axis([10, 200, 1e7, 1e9])
plt.yscale('log')
plt.xscale('log')
plt.ylabel(r"$\rho_{150\mathrm{pc}}$ [M$_{\odot}$ kpc$^{-3}$]")
plt.xlabel("$r_{p}$ [kpc]")
ax.tick_params(direction='in', axis='both', which='both', pad=4.5)

####
ax = plt.subplot(1, 2, 2)
plt.grid(linestyle='-', linewidth=0.3)

plt.errorbar(M200[0,:-1], rho_150pc_K19[0,:], yerr=rho_150pc_K19[1:,:], xerr=M200[1:,:-1],
             marker='o', markersize=3.5, markeredgecolor="none", ls='none', lw=0.5, c='tab:blue')


plt.axis([5e7, 1e10, 1e7, 1e9])
plt.yscale('log')
plt.xscale('log')
plt.ylabel(r"$\rho_{150\mathrm{pc}}$ [M$_{\odot}$ kpc$^{-3}$]")
plt.xlabel("$M_{200}$ [M$_{\odot}$]")
ax.tick_params(direction='in', axis='both', which='both', pad=4.5)

filename = "Observational_data.png"
plt.savefig(filename, dpi=300)
plt.close()


###

# labels = ['q', 'm']
# fig = corner.corner(samples_isolated,
#                     labels=labels,
#                     # truths=[q, m],
#                     quantiles=[0.16, 0.5, 0.84],
#                     show_titles=True,
#                     title_kwargs={"fontsize": 12})
# plt.savefig('corner_plot_rp_Isolated', dpi=300)
#
# fig = corner.corner(samples_LMC,
#                     labels=labels,
#                     # truths=[q, m],
#                     quantiles=[0.16, 0.5, 0.84],
#                     show_titles=True,
#                     title_kwargs={"fontsize": 12})
# plt.savefig('corner_plot_rp_LMC', dpi=300)