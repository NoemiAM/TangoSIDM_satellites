Table A1 from Kaplinghat et al. (2019), MNRAS, 490, 1, 231. 
Each row indicates the name of the dSph galaxy and the model assumed in the fitting (NFW or cISO).
The first three columns correspond to log10(Vmax) in units of [km/s] with the 50th (-16th,+84th)
percentiles. The following three columns correspond to log10(Rmax) in units of [kpc], with the 50th (-16th,+84th).

Vmax and Rmax can be used to calculate the initial free parameters of the NFW and cISO profiles as follows:

NFW:
rho_s = 1.721 Vmax^2 / (G Rmax^2)
r_s = 0.462 Rmax

cISO:
rho_0 = 2.556 Vmax^2 / (G Rmax^2)
sigma_0 = 0.63 Vmax
