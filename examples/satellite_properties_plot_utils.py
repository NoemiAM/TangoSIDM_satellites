# Plotting utilities for satellite properties plots
import matplotlib
import matplotlib.pyplot as plt
import h5py
import numpy as np
import scipy
from density_profile_fit_utils import fit_isothermal_model, fit_nfw_model, fit_core_nfw_model

######################
### Data utilities ###
######################

DATA_PATH = "../data/TangoSIDM/"

IDs = {
    "SigmaConstant00": "CDM",
    "SigmaConstant01": "SigmaConstant01",
    "SigmaConstant10": "SigmaConstant10",
    "SigmaVelDep20Anisotropic": "SigmaVel20",
    "SigmaVelDep60Anisotropic": "SigmaVel60", 
    "SigmaVelDep100Anisotropic": "SigmaVel100",
}

######################
### Plotting style ###
######################
plt.style.use("pltstyle.mplstyle")

mycmap = matplotlib.cm.RdYlBu
myblue = mycmap(0.9)
myred =  mycmap(0.1)


##################################
### Plot median with error bar ###
##################################
def plot_median_relation(ax, bins, x, y, errorbars=True, color='grey'):
    num_bins = len(bins)
    indx = np.digitize(x, bins)
    p_bins_medians = np.array([np.median(x[indx == idx]) for idx in np.arange(num_bins) if len(x[indx==idx])>5])
    r_bins_medians = np.array([np.median(y[indx == idx]) for idx in np.arange(num_bins) if len(x[indx==idx])>5])
    ax.plot(p_bins_medians, r_bins_medians, lw=2.5, color='white')
    ax.plot(p_bins_medians, r_bins_medians, color=color, lw=2)
    
    if errorbars:
        r_bins_16 = np.array([np.percentile(y[indx == idx], 16) for idx in np.arange(num_bins) if len(x[indx==idx])>5])
        r_bins_84 = np.array([np.percentile(y[indx == idx], 84) for idx in np.arange(num_bins) if len(x[indx==idx])>5]) 
        ax.plot(p_bins_medians, r_bins_16, '--', color=color)
        ax.plot(p_bins_medians, r_bins_84, '--', color=color)
        ax.fill_between(
                        p_bins_medians,
                        r_bins_16,
                        r_bins_84,
                        color=color,
                        alpha=0.15,
                    )


##############################
### Get data correlations ###
##############################
def get_correlations(x_array, y_array, id_name):
            pearson = scipy.stats.pearsonr(x_array, y_array)
            spearman = scipy.stats.spearmanr(x_array, y_array)
            print(id_name)
            print("Pearson's r:", pearson)
            print("Spearman's rho:", spearman)
            print("\n")
            return pearson, spearman
        
        
###########################
### Colorbar parameters ###
###########################

COLORBAR_DICT = {
    "accretion" : r'$z_\mathrm{infall}$',
    "mass_0" : r'$\log_{10}\mathrm{M_{bound}}$ $\mathrm{[M_\odot]}$',
    "mass_peak" : r'$\log_{10}\mathrm{M_{peak}}$ $\mathrm{[M_\odot]}$',
    "v_peak" : r'$\mathrm{V_{peak}}$ $\mathrm{[km\ s^{-1}]}$',
    "v_max" : r'$\mathrm{V_{max}(z=0)}$ $\mathrm{[km\ s^{-1}]}$',
    "r_max" : r'$\mathrm{R_{max}(z=0)}$ $\mathrm{[kpc]}$',
    "pericenter" : r'$r_{p}\ [\mathrm{kpc}]$'
}


def colorbar_args(colorbar_param):
    if colorbar_param == 'accretion':
        vmin, vmax = 0, 2.6
        cmap = mycmap 
        norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=((vmax+vmin)/2), vmax=vmax)

    elif colorbar_param == 'mass_0':
        vmin, vmax = 9, 12
        cmap = matplotlib.colors.ListedColormap([mycmap(0.85), mycmap(0.15), 'olivedrab'])
        norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=((vmax+vmin)/2), vmax=vmax)

    elif colorbar_param == 'mass_peak':
        vmin, vmax = 9, 12
        cmap = matplotlib.colors.ListedColormap([mycmap(0.85), mycmap(0.15), 'olivedrab'])
        norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=((vmax+vmin)/2), vmax=vmax)
    
    elif colorbar_param == 'v_peak':
        vmin, vmax = 15, 50
        cmap = matplotlib.cm.YlGnBu
        norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=((vmax+vmin)/2), vmax=vmax)
        
    elif colorbar_param == 'v_max':
        vmin, vmax = 15, 70
        cmap = matplotlib.cm.YlGnBu
        norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=((vmax+vmin)/2), vmax=vmax)
        
    elif colorbar_param == 'r_max':
        vmin, vmax = 0.1, 20
        cmap = matplotlib.cm.YlGnBu
        norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=5, vmax=vmax)
     
    elif colorbar_param == 'pericenter':
        vmin, vmax = 1, 200
        cmap = matplotlib.cm.viridis_r
        norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
        
    else:
        raise KeyError("Wrong key for colorbar_param.")
        
    return cmap, norm


##################################################################
### Density at 150pc versus pericenter radius plotting routine ###
##################################################################
def plot_density_150pc(colorbar_param, profile='NFW', print_correlation=False, filename:str=None):
    
    print(f'Plotting {profile} density at 150pc versus pericenter distance with {colorbar_param} colorbar!')

    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(13, 8), dpi=200, facecolor='white')
    axs = axs.flatten()
    cmap, norm = colorbar_args(colorbar_param)
    
    for i, (id, id_name) in enumerate(IDs.items()):
        file = h5py.File(DATA_PATH+f"{id}.hdf5", "r")
        x_array, y_array, c_array = [], [], []
        
        # Position labels
        axs[i].text(0.1, 0.1,  fr'$\texttt{{{id_name}}}$', color='black', transform=axs[i].transAxes,
            bbox=dict(facecolor='silver', edgecolor='none', alpha=0.4, boxstyle='round, pad=0.2')) #, horizontalalignment='right')
                    
        for idx in file.keys():
            if file[f'{idx}'].attrs.get('subhalo_of') is not None:
                subhalo_idx = idx

                if np.log10(file[str(subhalo_idx)]['tree_data']['bound_mass_dm'][0]) > 9: # MINIMUM satellite mass = 10^9
                    data_subhalo = file[f'{subhalo_idx}']
                    pericenter = data_subhalo['tree_data']['pericenter'][1]
                    z_accr_type_idx, accretion = data_subhalo['tree_data']['accretion']
                    
                    if profile == "NFW":
                        r0, rho0, _, _ = data_subhalo['halo_data']['nfw_fit']
                        density_fit = fit_nfw_model(np.array([0.15]), r0, rho0)
                        density_fit = 10**density_fit
                    elif profile == "core-NFW":
                        log10_M200, rc, n, _, _, _ = data_subhalo['halo_data']['core_nfw_fit']
                        density_fit = fit_core_nfw_model(np.array([0.15]*2), log10_M200, rc, n)[0]
                        density_fit = 10**density_fit
                    elif profile == "ISO":
                        r0, rho0, _, _ = data_subhalo['halo_data']['iso_fit']
                        density_fit = fit_isothermal_model(np.array([0.15]*2), r0, rho0)[0]
                        density_fit = 10**density_fit
                    else:
                        print("Wrong profile key! Choose between 'NFW', 'core-NFW', 'ISO'.")

                    if colorbar_param == 'accretion':
                        c = accretion
                    elif colorbar_param == 'mass_0':
                        mass_0 = data_subhalo['tree_data']['bound_mass_dm'][0] 
                        c = np.log10(mass_0)
                    elif colorbar_param == 'mass_peak':
                        mass_peak = data_subhalo['tree_data']['bound_mass_dm'][int(z_accr_type_idx)]
                        c = np.log10(mass_peak)
                    elif colorbar_param == 'v_peak':
                        c = data_subhalo['tree_data']['Vmax'][int(z_accr_type_idx)]
                    elif colorbar_param == 'v_max':
                        c = data_subhalo['tree_data']['Vmax'][0]
                    elif colorbar_param == 'r_max':
                        if 'Rmax' in data_subhalo['tree_data'].keys(): c = data_subhalo['tree_data']['Rmax'][...]
                        else: continue
                    
                    p = pericenter[0] if pericenter.shape==(1,) else pericenter
                    if p>6 and p<1.5e3: # cleanup
                        im = axs[i].scatter(x=pericenter, y=density_fit, marker='o',linewidths=0, norm=norm, c=c, cmap=cmap, alpha= 0.8)
                        x_array.append(p)
                        y_array.append(density_fit)
                        c_array.append(c)
                  
        x_array = np.array(x_array)
        y_array = np.array(y_array)
        c_array = np.array(c_array)
        
        if print_correlation: 
            get_correlations(x_array, y_array, id_name)
        
        bins = np.arange(1, 3, 0.3)
        bins = 10**bins
        if colorbar_param in ["mass_0", "mass_peak"]:
            mask_9 = c_array<10
            mask_10 = (c_array>=10)*(c_array<11)
            mask_11 = c_array>=11
                
            for jm, (mask, color) in enumerate(zip([mask_9, mask_10], [myblue, myred])):
                x_mask = x_array[mask]
                y_mask = y_array[mask]
                plot_median_relation(axs[i], bins, x_mask, y_mask, color=color)
            
        else:
            plot_median_relation(axs[i], bins, x_array, y_array, color='grey')

        file.close() 
                      
    # axis stuff
    for i, ax in enumerate(axs):
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(10, 500)
        if profile == "NFW":
            ax.set_ylim(1e7, 1e9)
        elif profile == "core-NFW":
            ax.set_ylim(5e6, 1e9)
        elif profile == "ISO":
            ax.set_ylim(1e6, 3e8)
            
    for axi in [3, 4, 5]:
        axs[axi].set_xlabel(r'$r_{{p}}\ [\mathrm{kpc}]$')
    for axi in [0, 3]:
        axs[axi].set_ylabel(r'$\rho(150\ \mathrm{pc})\ [\mathrm{M}_\odot \ \mathrm{kpc}^{-3}]$')
                
    # colorbar stuff
    cbar = fig.colorbar(im, ax=axs.ravel().tolist(), label=COLORBAR_DICT[colorbar_param], aspect=40, fraction=0.02, pad=0.03)
    if colorbar_param == "accretion":
        cbar.ax.set_yticks([0, 1, 2]) 
    elif colorbar_param in ["mass_0", "mass_peak"]:
        cbar.ax.set_yticks([9, 10, 11, 12]) 

    # plt.subplots_adjust(hspace=0.2, wspace=0.2, right=.86)
    plt.subplots_adjust(hspace=0.1, wspace=0.1, right=.86)
    if filename is not None:
        fig.savefig(f"figures/{profile}_{filename}.png", dpi=300, transparent=True)
    plt.show()
    
 
 
##############################################################################################
### Density at 150pc versus velocity  (V_max(z=0), V_peak, V(r_fiducial)) plotting routine ###
##############################################################################################
def plot_density_150pc_velocity(velocity:str="V_max", profile='NFW', print_correlation=False, filename:str=None):
    
    print(f'Plotting {profile} density at 150pc versus velocity {velocity}!')

    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(13, 8), dpi=200, facecolor='white')
    axs = axs.flatten()
    
    for i, (id, id_name) in enumerate(IDs.items()):
        file = h5py.File(DATA_PATH+f"{id}.hdf5", "r")
        x_array, y_array, c_array = [], [], []
        
        axs[i].text(0.1, 0.85,  fr'$\texttt{{{id_name}}}$', color='black', transform=axs[i].transAxes,
            bbox=dict(facecolor='silver', edgecolor='none', alpha=0.4, boxstyle='round, pad=0.2')) #, horizontalalignment='right')
                 
        for idx in file.keys():
            if file[f'{idx}'].attrs.get('subhalo_of') is not None:
                subhalo_idx = idx

                if np.log10(file[str(subhalo_idx)]['tree_data']['bound_mass_dm'][0]) > 9: # MINIMUM satellite mass = 10^9
                    data_subhalo = file[f'{subhalo_idx}']
                    pericenter = data_subhalo['tree_data']['pericenter'][1]
                    z_accr_type_idx, accretion = data_subhalo['tree_data']['accretion']
                    
                    if velocity == "v_peak" : vel = data_subhalo['tree_data']['Vmax'][int(z_accr_type_idx)]
                    elif velocity == "v_max": vel = data_subhalo['tree_data']['Vmax'][0]
                    else: print("Wrong velocity key! Choose between 'v_max', 'v_peak'.")
                    
                    if profile == "NFW":
                        r0, rho0, _, _ = data_subhalo['halo_data']['nfw_fit']
                        density_fit = fit_nfw_model(np.array([0.15]), r0, rho0)
                        density_fit = 10**density_fit
                    elif profile == "core-NFW":
                        log10_M200, rc, n, _, _, _ = data_subhalo['halo_data']['core_nfw_fit']
                        density_fit = fit_core_nfw_model(np.array([0.15]*2), log10_M200, rc, n)[0]
                        density_fit = 10**density_fit
                    elif profile == "ISO":
                        r0, rho0, _, _ = data_subhalo['halo_data']['iso_fit']
                        density_fit = fit_isothermal_model(np.array([0.15]*2), r0, rho0)[0]
                        density_fit = 10**density_fit
                    else:
                        print("Wrong profile key! Choose between 'NFW', 'core-NFW', 'ISO'.")
                        
                    if velocity == 'v_max':
                        colorbar_param = 'v_peak'
                        c = data_subhalo['tree_data']['Vmax'][int(z_accr_type_idx)]
                        cmap, norm = colorbar_args('v_peak')
                    elif velocity == 'v_peak':
                        colorbar_param = 'v_max'
                        c = data_subhalo['tree_data']['Vmax'][0]
                        cmap, norm = colorbar_args('v_max')
                    
                    im = axs[i].scatter(x=vel, y=density_fit, marker='o',linewidths=0, norm=norm, c=c, cmap=cmap, alpha=0.8)
                    x_array.append(vel)
                    y_array.append(density_fit)
                    c_array.append(c)
                  
        x_array = np.array(x_array)
        y_array = np.array(y_array)
        c_array = np.array(c_array)
        
        if print_correlation: 
            get_correlations(x_array, y_array, id_name)
        
        bins = np.arange(10, 70, 5)
        plot_median_relation(axs[i], bins, x_array, y_array, color='grey')

        file.close() 
                      
    # axis stuff
    for i, ax in enumerate(axs):
        ax.set_yscale('log')
        ax.set_xlim(10, 70)
        ax.set_xticks([10, 30, 50, 70])
        if profile == "NFW":
            ax.set_ylim(1e7, 1e9)
        elif profile == "core-NFW":
            ax.set_ylim(5e6, 1e9)
        elif profile == "ISO":
            ax.set_ylim(1e6, 3e8)
            
    for axi in [3, 4, 5]:
        if velocity == "v_peak" : xlabel = fr'$\mathrm{{V_{{peak}}}}$ $\mathrm{{[km\ s^{{-1}}]}}$'
        elif velocity == "v_max": xlabel= fr'$\mathrm{{V_{{max}}(z=0)}}$ $\mathrm{{[km\ s^{{-1}}]}}$'
        axs[axi].set_xlabel(xlabel)
    for axi in [0, 3]:
        axs[axi].set_ylabel(r'$\rho(150\ \mathrm{pc})\ [\mathrm{M}_\odot \ \mathrm{kpc}^{-3}]$')

    # colorbar stuff
    if velocity in ["v_max", "v_peak"]:
        cbar = fig.colorbar(im, ax=axs.ravel().tolist(), label=COLORBAR_DICT[colorbar_param], aspect=40, fraction=0.02, pad=0.03)
        
    plt.subplots_adjust(hspace=0.1, wspace=0.1, right=.86)
    if filename is not None:
        fig.savefig(f"figures/{profile}_{filename}.png", dpi=300, transparent=True)
    plt.show()
    
   
##############################################################################
### Max circular velocity  plotting routine ###
##############################################################################
def plot_vmax(colorbar_param, print_correlation=False, filename:str=None):
    """Vmax rp"""
    
    print(f'Max circular velocity  with {colorbar_param} colorbar!')

    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(13, 8), dpi=200, facecolor='white')
    axs = axs.flatten()
    cmap, norm = colorbar_args(colorbar_param)

    for i, (id, id_name) in enumerate(IDs.items()):
        file = h5py.File(DATA_PATH+f"{id}.hdf5", "r")
        
        axs[i].text(0.1, 0.85,  fr'$\texttt{{{id_name}}}$', color='black', transform=axs[i].transAxes,
            bbox=dict(facecolor='silver', edgecolor='none', alpha=0.4, boxstyle='round, pad=0.2')) #, horizontalalignment='right')
       
        x_array, y_array, c_array = [], [], []
        for idx in file.keys():
            if file[f'{idx}'].attrs.get('subhalo_of') is not None:
                subhalo_idx = idx

                if np.log10(file[str(subhalo_idx)]['tree_data']['bound_mass_dm'][0]) > 9: # MINIMUM satellite mass = 10^9
                    data_subhalo = file[f'{subhalo_idx}']
                    pericenter = data_subhalo['tree_data']['pericenter'][1]
                    z_accr_type_idx, accretion = data_subhalo['tree_data']['accretion']
                    vmax = data_subhalo['tree_data']['Vmax'][0]

                    if colorbar_param == 'accretion':
                        c = accretion
                    elif colorbar_param == 'mass_0':
                        mass_0 = data_subhalo['tree_data']['bound_mass_dm'][0] 
                        c = np.log10(mass_0)
                    elif colorbar_param == 'mass_peak':
                        mass_peak = data_subhalo['tree_data']['bound_mass_dm'][int(z_accr_type_idx)]
                        c = np.log10(mass_peak)
                    
                    p = pericenter[0] if pericenter.shape==(1,) else pericenter
                    if p>6 and p<1.5e3:
                        im = axs[i].scatter(x=pericenter, y=vmax, marker='o', linewidths=0, norm=norm, c=c, cmap=cmap, alpha=0.8)
                        x_array.append(p)
                        c_array.append(c)
                        y_array.append(vmax)
                  
        x_array = np.array(x_array)
        y_array = np.array(y_array)
        c_array = np.array(c_array)
            
        if print_correlation:
            get_correlations(x_array, y_array, id_name)
            
        bins = np.arange(1, 3, 0.3)
        bins = 10**bins
        if colorbar_param in ["mass_0", "mass_peak"]:
            mask_9 = c_array<10
            mask_10 = (c_array>=10)*(c_array<11)
            mask_11 = c_array>=11
                
            for jm, (mask, color) in enumerate(zip([mask_9, mask_10], [myblue, myred])):
                p_mask = x_array[mask]
                r_mask = y_array[mask]
                plot_median_relation(axs[i], bins, p_mask, r_mask, color=color)
            
        else:
            if id_name == "CDM":
                p_CDM = x_array
                r_CDM = y_array
                plot_median_relation(axs[i], bins, x_array, y_array, color='k')
            else:
                plot_median_relation(axs[i], bins, p_CDM, r_CDM, errorbars=False, color='k')
                plot_median_relation(axs[i], bins, x_array, y_array, color='grey')

        file.close() 
           
    # axis stuff
    for ax in axs:
        ax.set_xscale('log')
        ax.set_xlim(10, 500)
        ax.set_ylim(10, 80)
        ax.set_yticks([20, 40, 60, 80])

    for axi in [3, 4, 5]:
        axs[axi].set_xlabel(r'$r_{{p}}\ [\mathrm{kpc}]$')
    for axi in [0, 3]:
        axs[axi].set_ylabel(r'$\mathrm{V_{max}(z=0)}$ $\mathrm{[km\ s^{-1}]}$')

    # colorbar stuff
    cbar = fig.colorbar(im, ax=axs.ravel().tolist(), label=COLORBAR_DICT[colorbar_param], aspect=40, fraction=0.02, pad=0.03)
    if colorbar_param == "accretion":
        cbar.ax.set_yticks([0, 1, 2]) 
    elif colorbar_param in ["mass_0", "mass_peak"]:
        cbar.ax.set_yticks([9, 10, 11, 12]) 

    plt.subplots_adjust(hspace=0.1, wspace=0.1, right=.86)
    if filename is not None:
        fig.savefig(f"figures/{filename}.png", dpi=300, transparent=True)
    plt.show()
    
##############################################################################
### Peak velocity  plotting routine ###
##############################################################################
def plot_vpeak(colorbar_param, print_correlation=False, filename:str=None):
    """Vpeak rp"""
    
    print(f'Peak velocity  with {colorbar_param} colorbar!')

    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(13, 8), dpi=200, facecolor='white')
    axs = axs.flatten()
    cmap, norm = colorbar_args(colorbar_param)

    for i, (id, id_name) in enumerate(IDs.items()):
        file = h5py.File(DATA_PATH+f"{id}.hdf5", "r")
        axs[i].text(0.1, 0.85,  fr'$\texttt{{{id_name}}}$', color='black', transform=axs[i].transAxes,
            bbox=dict(facecolor='silver', edgecolor='none', alpha=0.4, boxstyle='round, pad=0.2')) #, horizontalalignment='right')
       
        x_array, y_array, c_array = [], [], []
        for idx in file.keys():
            if file[f'{idx}'].attrs.get('subhalo_of') is not None:
                subhalo_idx = idx

                if np.log10(file[str(subhalo_idx)]['tree_data']['bound_mass_dm'][0]) > 9: # MINIMUM satellite mass = 10^9
                    data_subhalo = file[f'{subhalo_idx}']
                    pericenter = data_subhalo['tree_data']['pericenter'][1]
                    z_accr_type_idx, accretion = data_subhalo['tree_data']['accretion']
                    vpeak = data_subhalo['tree_data']['Vmax'][int(z_accr_type_idx)]

                    if colorbar_param == 'accretion':
                        c = accretion
                    elif colorbar_param == 'mass_0':
                        mass_0 = data_subhalo['tree_data']['bound_mass_dm'][0] 
                        c = np.log10(mass_0)
                    elif colorbar_param == 'mass_peak':
                        mass_peak = data_subhalo['tree_data']['bound_mass_dm'][int(z_accr_type_idx)]
                        c = np.log10(mass_peak)
                    
                    p = pericenter[0] if pericenter.shape==(1,) else pericenter
                    if p>6 and p<1.5e3:
                        im = axs[i].scatter(x=pericenter, y=vpeak, marker='o', linewidths=0, norm=norm, c=c, cmap=cmap, alpha=0.8)
                        x_array.append(p)
                        c_array.append(c)
                        y_array.append(vpeak)
                  
        x_array = np.array(x_array)
        y_array = np.array(y_array)
        c_array = np.array(c_array)
            
        if print_correlation:
            get_correlations(x_array, y_array, id_name)
            
        bins = np.arange(1, 3, 0.3)
        bins = 10**bins
        if colorbar_param in ["mass_0", "mass_peak"]:
            mask_9 = c_array<10
            mask_10 = (c_array>=10)*(c_array<11)
            mask_11 = c_array>=11
                
            for jm, (mask, color) in enumerate(zip([mask_9, mask_10], [myblue, myred])):
                p_mask = x_array[mask]
                r_mask = y_array[mask]
                plot_median_relation(axs[i], bins, p_mask, r_mask, color=color)
            
        else:
            if id_name == "CDM":
                p_CDM = x_array
                r_CDM = y_array
                plot_median_relation(axs[i], bins, x_array, y_array, color='k')
            else:
                plot_median_relation(axs[i], bins, p_CDM, r_CDM, errorbars=False, color='k')
                plot_median_relation(axs[i], bins, x_array, y_array, color='grey')

        file.close() 
           
    # axis stuff
    for ax in axs:
        ax.set_xscale('log')
        ax.set_xlim(10, 500)
        ax.set_ylim(10, 80)
        ax.set_yticks([20, 40, 60, 80])

    for axi in [3, 4, 5]:
        axs[axi].set_xlabel(r'$r_{{p}}\ [\mathrm{kpc}]$')
    for axi in [0, 3]:
        axs[axi].set_ylabel(r'$\mathrm{V_{peak}}$ $\mathrm{[km\ s^{-1}]}$')
    # colorbar stuff
    cbar = fig.colorbar(im, ax=axs.ravel().tolist(), label=COLORBAR_DICT[colorbar_param], aspect=40, fraction=0.02, pad=0.03)
    if colorbar_param == "accretion":
        cbar.ax.set_yticks([0, 1, 2]) 
    elif colorbar_param in ["mass_0", "mass_peak"]:
        cbar.ax.set_yticks([9, 10, 11, 12]) 

    plt.subplots_adjust(hspace=0.1, wspace=0.1, right=.86)
    if filename is not None:
        fig.savefig(f"figures/{filename}.png", dpi=300, transparent=True)
    plt.show()
    

##############################################################################
### Max circular velocity / max circular velocity at peak plotting routine ###
##############################################################################
def plot_vmax_over_vpeak(colorbar_param, print_correlation=False, filename:str=None):
    """E.g. Fig.2 in Robles and Bullock 2020"""
    
    print(f'Max circular velocity / max circular velocity at peak with {colorbar_param} colorbar!')

    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(13, 8), dpi=200, facecolor='white')
    axs = axs.flatten()
    cmap, norm = colorbar_args(colorbar_param)

    for i, (id, id_name) in enumerate(IDs.items()):
        file = h5py.File(DATA_PATH+f"{id}.hdf5", "r")
        axs[i].text(400, 0.2, fr'$\texttt{{{id_name}}}$', color='black', 
            bbox=dict(facecolor='silver', edgecolor='none', alpha=0.4, boxstyle='round, pad=0.2'), horizontalalignment='right')

        x_array, y_array, c_array = [], [], []
        for idx in file.keys():
            if file[f'{idx}'].attrs.get('subhalo_of') is not None:
                subhalo_idx = idx
                data_subhalo = file[f'{subhalo_idx}']
                z_accr_type_idx, accretion = data_subhalo['tree_data']['accretion']

                # if np.log10(file[str(subhalo_idx)]['tree_data']['bound_mass_dm'][0]) > 9: # MINIMUM satellite mass = 10^9
                if np.log10(file[str(subhalo_idx)]['tree_data']['bound_mass_dm'][0]) > 9 and np.log10(file[str(subhalo_idx)]['tree_data']['bound_mass_dm'][int(z_accr_type_idx)]) > 9: # MINIMUM satellite mass = 10^9
                    data_subhalo = file[f'{subhalo_idx}']
                    pericenter = data_subhalo['tree_data']['pericenter'][1]
                    z_accr_type_idx, accretion = data_subhalo['tree_data']['accretion']
                    vmax = data_subhalo['tree_data']['Vmax'][0]
                    vpeak = data_subhalo['tree_data']['Vmax'][int(z_accr_type_idx)]

                    if colorbar_param == 'accretion':
                        c = accretion
                    elif colorbar_param == 'mass_0':
                        mass_0 = data_subhalo['tree_data']['bound_mass_dm'][0] 
                        c = np.log10(mass_0)
                    elif colorbar_param == 'mass_peak':
                        cmap = matplotlib.cm.YlGnBu_r
                        norm = matplotlib.colors.TwoSlopeNorm(vmin=9, vcenter=9.5, vmax=12)        
                        mass_peak = data_subhalo['tree_data']['bound_mass_dm'][int(z_accr_type_idx)]
                        c = np.log10(mass_peak)
                    
                    p = pericenter[0] if pericenter.shape==(1,) else pericenter
                    if p>6 and p<1e3:
                        im = axs[i].scatter(x=pericenter, y=vmax/vpeak, marker='o', linewidths=0, norm=norm, c=c, cmap=cmap, alpha=0.8)
                        x_array.append(p)
                        c_array.append(c)
                        y_array.append(vmax/vpeak)
                  
        x_array = np.array(x_array)
        y_array = np.array(y_array)
        c_array = np.array(c_array)
        if print_correlation:
            get_correlations(x_array, y_array, id_name)
            
        bins = np.arange(1, 3, 0.3)
        bins = 10**bins
        
        plot_median_relation(axs[i], bins, x_array, y_array, color='grey')

        # if colorbar_param in ["mass_0", "mass_peak"]:
        #     mask_9 = c_array<10
        #     mask_10 = (c_array>=10)*(c_array<11)
        #     mask_11 = c_array>=11
                
        #     for jm, (mask, color) in enumerate(zip([mask_9, mask_10], [myblue, myred])):
        #         p_mask = x_array[mask]
        #         r_mask = y_array[mask]
        #         plot_median_relation(axs[i], bins, p_mask, r_mask, color=color)
            
        # else:
        #     if id_name == "CDM":
        #         p_CDM = x_array
        #         r_CDM = y_array
        #         plot_median_relation(axs[i], bins, x_array, y_array, color='k')
        #     else:
        #         plot_median_relation(axs[i], bins, p_CDM, r_CDM, errorbars=False, color='k')
        #         plot_median_relation(axs[i], bins, x_array, y_array, color='grey')

        file.close() 
           
    # axis stuff
    for ax in axs:
        ax.set_xscale('log')
        ax.set_xlim(10, 500)
        ax.set_ylim(0, 1.5)

    for axi in [3, 4, 5]:
        axs[axi].set_xlabel(r'$r_{{p}}\ [\mathrm{kpc}]$')
    for axi in [0, 3]:
        axs[axi].set_ylabel(fr'$\mathrm{{V_{{max}}}}(z=0)/\mathrm{{V_{{peak}}}}$')

    # colorbar stuff
    cbar = fig.colorbar(im, ax=axs.ravel().tolist(), label=COLORBAR_DICT[colorbar_param], aspect=40, fraction=0.02, pad=0.03)
    if colorbar_param == "accretion":
        cbar.ax.set_yticks([0, 1, 2]) 
    elif colorbar_param in ["mass_0", "mass_peak"]:
        cbar.ax.set_yticks([9, 10, 11, 12]) 

    plt.subplots_adjust(hspace=0.1, wspace=0.1, right=.86)
    if filename is not None:
        fig.savefig(f"figures/{filename}.png", dpi=300, transparent=True)
    plt.show()
 
 
 
#############################################################################
### VERTICAL - Density at 150pc versus pericenter radius plotting routine ###
#############################################################################

def plot_density_150pc_vertical(colorbar_param:str, profile='NFW', filename:str=None, max10:bool=False):

    print(f'Plotting {profile} density at 150pc versus pericenter distance with {colorbar_param} colorbar!')

    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(5, 12), dpi=200, facecolor='white')
    axs = axs.flatten()

    for j, (id, id_name) in enumerate(IDs.items()):
        if id_name in ["CDM", "SigmaVel100"]:
            if id_name == "CDM": i =0
            elif id_name == "SigmaVel100": i =1
                
            p_array, r_array, c_array = [], [], []

            # Position labels
            if profile == "ISO":
                axs[i].text(13, 1.5e8,  fr'$\texttt{{{id_name}}}$', color='black', 
                    bbox=dict(facecolor='silver', edgecolor='none', alpha=0.4, boxstyle='round, pad=0.2')) #, horizontalalignment='right')
            else:
                axs[i].text(13, 6e8,  fr'$\texttt{{{id_name}}}$', color='black', 
                        bbox=dict(facecolor='silver', edgecolor='none', alpha=0.4, boxstyle='round, pad=0.2')) #, horizontalalignment='right')
     
            file = h5py.File(DATA_PATH+f"{id}.hdf5", "r")
            for idx in file.keys():
                if file[f'{idx}'].attrs.get('main_halo_of') is not None:
                    halo_idx = idx
                    data_halo = file[f'{halo_idx}']
                    subhalos_idxs = file[f'{halo_idx}'].attrs.get('main_halo_of')

                    if max10: r = min(len(subhalos_idxs), 10)
                    else: r = len(subhalos_idxs)
                    
                    for subhalo in range(r):
                        subhalo_idx = subhalos_idxs[subhalo]
                        data_subhalo = file[f'{subhalo_idx}']

                        if np.log10(data_subhalo['tree_data']['bound_mass_dm'][0]) > 9: # MINIMUM satellite mass = 10^9
                            pericenter = data_subhalo['tree_data']['pericenter'][1]
                            z_accr_type_idx, accretion = data_subhalo['tree_data']['accretion']

                            if profile == "NFW":
                                r0, rho0, _, _ = data_subhalo['halo_data']['nfw_fit']
                                density_fit = fit_nfw_model(np.array([0.15]), r0, rho0)
                                density_fit = 10**density_fit
                            elif profile == "core-NFW":
                                log10_M200, rc, n, _, _, _ = data_subhalo['halo_data']['core_nfw_fit']
                                density_fit = fit_core_nfw_model(np.array([0.15]*2), log10_M200, rc, n)[0]
                                density_fit = 10**density_fit
                            elif profile == "ISO":
                                r0, rho0, _, _ = data_subhalo['halo_data']['iso_fit']
                                density_fit = fit_isothermal_model(np.array([0.15]*2), r0, rho0)[0]
                                density_fit = 10**density_fit
                            else:
                                print("Wrong profile key! Choose between 'NFW', 'core-NFW', 'ISO'.")


                            if colorbar_param == 'accretion':
                                c = accretion
                                cmap, norm = colorbar_args(colorbar_param)
                            elif colorbar_param == 'mass_0':
                                mass_0 = data_subhalo['tree_data']['bound_mass_dm'][0] 
                                c = np.log10(mass_0)
                                mycmap = matplotlib.cm.RdYlBu
                                myblue = mycmap(0.9)
                                myred =  mycmap(0.1)
                                vmin, vmax = 9, 12
                                cmap = matplotlib.colors.ListedColormap(['olivedrab', myblue, 'peru', myred, 'darkorchid', 'midnightblue'])
                                norm = matplotlib.colors.TwoSlopeNorm(vmin=9, vcenter=9.6, vmax=12)
                                # cmap =  matplotlib.colors.ListedColormap(matplotlib.cm.Blues(np.linspace(0.2, 1, 6)))
                                #cmap, norm = colorbar_args(colorbar_param)
                            elif colorbar_param == 'mass_peak':
                                mass_peak = data_subhalo['tree_data']['bound_mass_dm'][int(z_accr_type_idx)]
                                c = np.log10(mass_peak)
                                mycmap = matplotlib.cm.RdYlBu
                                myblue = mycmap(0.9)
                                myred =  mycmap(0.1)
                                vmin, vmax = 9, 12
                                cmap = matplotlib.colors.ListedColormap(['olivedrab', myblue, 'peru', myred, 'darkorchid', 'midnightblue'])
                                norm = matplotlib.colors.TwoSlopeNorm(vmin=9, vcenter=9.6, vmax=12)
                                # cmap, norm = colorbar_args(colorbar_param)
                            elif colorbar_param == 'mass_parent':
                                mass_parent = data_halo['tree_data']['bound_mass_dm'][0] 
                                c = np.log10(mass_parent)
                                vmin, vmax = np.log10(6e11), np.log10(2e12)
                                cmap =  matplotlib.colors.ListedColormap(matplotlib.cm.Blues(np.linspace(0.2, 1, 5)))
                                norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=((vmax+vmin)/2), vmax=vmax)

                            p = pericenter[0] if pericenter.shape==(1,) else pericenter
                            if p>6 and p<1.5e3:
                                im = axs[i].scatter(x=pericenter, y=density_fit, marker='o', linewidths=0,  
                                                    c=c, cmap=cmap, norm=norm, alpha=0.8)
                                p_array.append(p)
                                r_array.append(density_fit)
                                c_array.append(c)

            p_array = np.array(p_array)
            r_array = np.array(r_array)
            c_array = np.array(c_array)
            #Let's add median trends..
            p_bins = np.arange(1, 3, 0.3)
            p_bins = 10**p_bins
            plot_median_relation(axs[i], p_bins, p_array, r_array, color='grey')
            
        file.close() 

    # axis stuff
    for i, ax in enumerate(axs):
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(10, 500)
        ax.set_ylabel(r'$\rho(150\ \mathrm{pc})\ [\mathrm{M}_\odot \ \mathrm{kpc}^{-3}]$')

    axs[1].set_xlabel(r'$r_{{p}}\ [\mathrm{kpc}]$')
    for ax in axs:
        if profile == "NFW":
            ax.set_ylim(1e7, 1e9)
        elif profile == "core-NFW":
            ax.set_ylim(5e6, 1e9)
        elif profile == "ISO":
            ax.set_ylim(1e6, 3e8)
    
    # colorbar stuff
    if colorbar_param == "accretion":
        cbar = fig.colorbar(im, ax=axs.ravel().tolist(), label=COLORBAR_DICT[colorbar_param], location='top')
        cbar.ax.set_xticks([0, 1, 2]) 
        cbar.set_label(COLORBAR_DICT[colorbar_param], labelpad=7)
    elif colorbar_param in ["mass_0", "mass_peak"]:
        cbar = fig.colorbar(im, ax=axs.ravel().tolist(), label=COLORBAR_DICT[colorbar_param], location='top')
        cbar.ax.set_xticks([9, 9.2, 9.4, 9.6, 10.4, 11.2, 12]) 
        cbar.set_label(COLORBAR_DICT[colorbar_param], labelpad=7)
    elif colorbar_param == "mass_parent":
        cbar = fig.colorbar(im, ax=axs.ravel().tolist(), label=fr'$\log_{{10}}\mathrm{{M_{{200c, central}}}}$ $\mathrm{{[M_\odot]}}$',
                        location='top')
        cbar.set_label(COLORBAR_DICT[colorbar_param], labelpad=7)
        
    plt.subplots_adjust(hspace=0.075, top=0.75)
    if filename is not None:
        fig.savefig(f"figures/{profile}_{filename}.png", dpi=300, transparent=True)
    plt.show()
