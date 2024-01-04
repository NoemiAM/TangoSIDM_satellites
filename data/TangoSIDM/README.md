# TangoSIDM

The data contains information regarding Milky-Way type halos and their satellites in the TangoSIDM simulations. TangoSIDM is a suite of cosmological simulations of structure formation in a $\Lambda$ self-interacting dark matter ($\Lambda$SIDM) universe presented in [Correa et al. 2022](https://arxiv.org/abs/2206.11298).

In this work, we focus on six simulations of size $25\ \mathrm{Mpc^3}$ that follow the evolution of $752^3$ dark matter particles. The simulations include different dark matter model (for more details see [Correa et al. 2022](https://arxiv.org/abs/2206.11298)):
- `SigmaConstant00.hdf5` -- constant scattering cross section of $0\ \mathrm{cm^2\ \per\ g}$, a cold dark matter model (CDM).
- `SigmaConstant01.hdf5` -- constant scattering cross section of $1\ \mathrm{cm^2\ \per\ g}$.
- `SigmaConstant10.hdf5` -- constant scattering cross section of $10\ \mathrm{cm^2\ \per\ g}$.
- `SigmaVelDep20Anisotropic.hdf5` -- elastic and anisotropic collisions with scattering cross-section per unit mass of $\sigma_T / m_\chi = 20\ \mathrm{cm^2\ \per\ g}$ at $10\ \mathrm{km\ \per\ s}$.
- `SigmaVelDep60Anisotropic.hdf5` -- elastic and anisotropic collisions with scattering cross-section per unit mass of $\sigma_T / m_\chi = 60\ \mathrm{cm^2\ \per\ g}$ at $10\ \mathrm{km\ \per\ s}$.
- `SigmaVelDep100Anisotropic.hdf5` -- elastic and anisotropic collisions with scattering cross-section per unit mass of $\sigma_T / m_\chi = 100\ \mathrm{cm^2\ \per\ g}$ at $10\ \mathrm{km\ \per\ s}$.

## Data structure

To learn more about the content of the file and how they are structured use the notebook [`examples/ 0 - TangoSIDM_data.ipynb`](https://github.com/NoemiAM/TangoSIDM_satellites/blob/main/examples/0%20-%20TangoSIDM_data.ipynb).

