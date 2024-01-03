import numpy as np
from scipy.interpolate import interp1d


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


def calculate_r_s(Rmax):
    r_s = 0.462 * Rmax
    return r_s


def calculate_error_r_s(Rmax, delta_Rmax):
    r_s = calculate_r_s(Rmax)
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
    r_s = calculate_r_s(Rmax)
    rho_s = calculate_rho_s(Vmax, Rmax)
    e_rho_s = calculate_error_rho_s(Vmax, Rmax, delta_Vmax, delta_Rmax)
    e_r_s = calculate_error_r_s(Rmax, delta_Rmax)

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