import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from pylab import *
import matplotlib.pyplot as plt
import os

GAIA_PATH = "../data/Gaia/"
Kaplinghat_data = pd.read_csv(GAIA_PATH+"Kaplinghat_2019.csv").to_numpy()
dSph = Kaplinghat_data[:,0]
dSph = dSph[::2]
num_satellites = len(dSph)
r_p_Isolated = np.zeros(num_satellites)  # Reading pericenter assuming isolated MW
r_p_LMC = np.zeros(num_satellites)       # Reading pericenter assuming MW perturbed by LMC
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
    "font.size": 11,
    "font.family": "Times",
    "text.usetex": True,
    "figure.figsize": (3.2, 3),
    "figure.subplot.left": 0.16,
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
ax = plt.subplot(1, 1, 1)
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

filename = "Observational_data_rho_150pc_r_p.png"
plt.savefig(filename, dpi=300)
plt.close()