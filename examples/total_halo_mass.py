import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from pylab import *
import matplotlib.pyplot as plt
import os

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

def NFW_profile(r, rho_s, r_s):
    rho = np.log10(rho_s)
    rho -= np.log10( (r/r_s) * ((r/r_s) + 1.0)**2 )
    return 10**rho

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
    R200 = function_interpolate(200 * rhocrit)
    return R200

GAIA_PATH = "../data/Gaia/"
Kaplinghat_data = pd.read_csv(GAIA_PATH+"Kaplinghat_2019.csv").to_numpy()

Vmax = Kaplinghat_data[:,2]
NFW_Vmax = 10**Vmax[::2] # Only NFW fit data; [km/s] units
Rmax = Kaplinghat_data[:,5]
NFW_Rmax = 10**Rmax[::2] # Only NFW fit data; [kpc] units

num_satellites = len(NFW_Rmax)
M200 = np.zeros(num_satellites)
R200 = np.zeros(num_satellites)

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
    R200[i] = calculate_R200(rho_within_r, r_range)
    M200[i] = calculate_M200(R200[i])

    # Alternative method for NFW:
    # M_s = calc_Ms(r_s[i], rho_s[i])
    # R200[i] = fsolve(calcR200, 100., args=(r_s[i], M_s))
    # M200[i] = calculate_M200(R200b[i])

dSph = Kaplinghat_data[:,0]
dSph = dSph[::2]

r_p_Isolated = np.zeros(num_satellites) # Reading pericenter assuming isolated MW
r_p_LMC = np.zeros(num_satellites)      # Reading pericenter assuming MW perturbed by LMC
rho_150pc_K19 = np.zeros(num_satellites) # Reading Kaplinghat+ 2019 determination under NFW

for i, dSphi in enumerate(dSph):
    filename = "Cardona_2023_"+dSphi+".csv"
    if not os.path.exists(GAIA_PATH + filename): continue
    data = pd.read_csv(GAIA_PATH + filename).to_numpy()
    r_p_Isolated[i] = data[2,0]
    r_p_LMC[i] = data[3,0]
    rho_150pc_K19[i] = data[8,0] * 1e7 #Msun/kpc^3

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
plt.plot(r_p_Isolated, rho_150pc_K19, 'o', color='tab:blue',label='Isolated MW')
plt.plot(r_p_LMC, rho_150pc_K19, 'x',ms=5, color='crimson',label='MW + LMC')

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
plt.plot(M200, rho_150pc_K19, 'o', color='tab:blue')

plt.axis([5e7, 1e10, 1e7, 1e9])
plt.yscale('log')
plt.xscale('log')
plt.ylabel(r"$\rho_{150\mathrm{pc}}$ [M$_{\odot}$ kpc$^{-3}$]")
plt.xlabel("$M_{200}$ [M$_{\odot}$]")
ax.tick_params(direction='in', axis='both', which='both', pad=4.5)

filename = "Observational_data.png"
plt.savefig(filename, dpi=300)
plt.close()
