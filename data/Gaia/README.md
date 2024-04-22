## Observational data


### Cardona_2023_XXX.csv 

Data compiled from Table 1 from [Cardona-Barrero et al. 2023](https://arxiv.org/abs/2304.06611). Each file Cardona_2023_XXX.csv corresponds to a given spheroidal galaxy (Carina, Draco, Fornax, LeoI, LeoII, Sculptor, Sextans, UMi). 

The first six rows correspond to the pericenter (kpc) determined by 
1. [Fritz et al. 2018](https://arxiv.org/abs/1805.00908) (for a MW galaxy of $8*10^11\ M_\odot$ and based on Gaia DR2),
2. [Battaglia et al. 2022](https://arxiv.org/abs/2106.08819) (for an Isolated MW galaxy of $8.8\ 10^{11}\ M_\odot$), 
3. [Battaglia et al. 2022](https://arxiv.org/abs/2106.08819) (for an Isolated MW galaxy of $1.6\ 10^{12}\ M_\odot$), 
4. [Battaglia et al. 2022](https://arxiv.org/abs/2106.08819) (for an MW galaxy of $8.8\ 10^{11}\ M_\odot$ perturbed by a $1.5\ 10^{11}\ M_\odot$ LMC), 
5. [Pace et al. 2022](https://arxiv.org/abs/2205.05699) (for a MW galaxy of $1.3\ 10^{12}\ M_\odot$ and based on Gaia DR3),
6. [Pace et al. 2022](https://arxiv.org/abs/2205.05699) (for a MW galaxy of $1.3\ 10^{12}\ M_\odot$ perturbed by a $1.38\ 10^{11}\ M_\odot$ LMC, and based on Gaia DR3).

The following four rows correspond to the central density at $150 \rm{pc}$ ($10^7\ M_\odot\ kpc^{−3}$) given by
1. [Read et al. 2019](https://arxiv.org/abs/1808.06634),
2. [Hayashi et al. 2020](https://arxiv.org/abs/2007.13780),
3. [Kaplinghat et al. 2019](https://arxiv.org/abs/1904.04939) (assuming NFW profile),
4. [Kaplinghat et al. 2019](https://arxiv.org/abs/1904.04939) (assuming cored-density profile).

The various columns correspond to the 50th, (-)16th, (+)84th percentiles.


### Kaplinghat_2019.csv 

Data compiled from Table 1 from [Kaplinghat et al. 2019](https://arxiv.org/abs/1904.04939). Each row indicates the name of the dSph galaxy and the model assumed in the fitting (NFW or cISO). The first three columns correspond to log10(Vmax) in units of [km/s] with the 50th (-16th,+84th) percentiles. The following three columns correspond to log10(Rmax) in units of [kpc], with the 50th (-16th,+84th).

Vmax and Rmax can be used to calculate the initial free parameters of the NFW and cISO profiles as follows:

NFW:
rho_s = 1.721 Vmax^2 / (G Rmax^2)
r_s = 0.462 Rmax

cISO:
rho_0 = 2.556 Vmax^2 / (G Rmax^2)
sigma_0 = 0.63 Vmax


### Read_2019.csv 
Data compiled from Table 1 from [Read et al. 2019](https://arxiv.org/abs/1808.06634). Each row indicates the name of the dSph galaxy. The three columns correspond to $M_{200}$ in units of $[M_\odot]$ with the 50th (-16th,+84th) percentiles.


### Hayashi_2019.csv 
Data from [Hayashi et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023ApJ...953..185H/abstract). The central densities were taken from Table A1, whereas the stellar-to-halo mass ratios were taken from Fig. 2. In their work, Hayashi et al. computed the stellar mass-halo mass ratios of the dwarf galaxies by employing the self-consistent abundance matching model by Moster et al. (2013), and adopting the stellar masses of most dSphs taken from the literature (Table 1). Additionally, for the ultra-faint dwarfs with no stellar mass estimation, they calculated them based on their luminosities by assuming a stellar mass-to-light ratio of 1.6 Msun/Lsun, which is the median value for dSphs measured by Woo et al. (2008). Errors in the table correspond to 16th and 84th percentiles.


### McConnachie_2012.csv 
Stellar masses from dwarf satellites and ultra-faint dwarfs taken from the updated table from [McCoonnachie et al. (2012)](https://www.astro.uvic.ca/~alan/Nearby_Dwarf_Database.html). Table 4. Exceptions are Antlia 2, Crater 2, and Eridanus 2. Stellar masses for Antlia 2 were taken from [Torrealba et al. 2019](https://academic.oup.com/mnras/article/488/2/2743/5514354), for Crater 2 from [Torrealba et al. 2016](https://ui.adsabs.harvard.edu/abs/2016MNRAS.459.2370T/abstract), and for Eridanus 2 from [Gallart et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021ApJ...909..192G/abstract).
