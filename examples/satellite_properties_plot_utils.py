# Plotting utilities for satellite properties plots
import matplotlib
import matplotlib.pyplot as plt
import h5py
import numpy as np
import scipy
from density_fit_utils import fit_isothermal_model, fit_nfw_model

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
def plot_median_relation(ax, bins, x, y, errorbars=True, color='lightslategrey'):
    num_bins = len(bins)
    indx = np.digitize(x, bins)
    p_bins_medians = np.array([np.median(x[indx == idx]) for idx in np.arange(num_bins) if len(x[indx==idx])>5])
    r_bins_medians = np.array([np.median(y[indx == idx]) for idx in np.arange(num_bins) if len(x[indx==idx])>5])
    ax.plot(p_bins_medians, r_bins_medians, lw=3, color='white')
    ax.plot(p_bins_medians, r_bins_medians, color=color)
    
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
    "accretion" : r'$z_\mathrm{accretion}$',
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
        cmap = matplotlib.colors.ListedColormap([myblue, myred, 'olivedrab'])
        norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=((vmax+vmin)/2), vmax=vmax)

    elif colorbar_param == 'mass_peak':
        vmin, vmax = 9, 12
        cmap = matplotlib.colors.ListedColormap([myblue, myred, 'olivedrab'])
        norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=((vmax+vmin)/2), vmax=vmax)
    
    elif colorbar_param == 'v_peak':
        vmin, vmax = 15, 50
        cmap = matplotlib.cm.YlGnBu
        norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=((vmax+vmin)/2), vmax=vmax)
        
    elif colorbar_param == 'v_max':
        vmin, vmax = 15, 50
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
def plot_density_150pc(colorbar_param, print_correlation=False, filename:str=None):
    
    print(f'Plotting density at 150pc versus pericenter distance with {colorbar_param} colorbar!')

    fig, axs = plt.subplots(2, 3, sharex=True, sharey=False, figsize=(13, 8), dpi=200, facecolor='white')
    axs = axs.flatten()
    cmap, norm = colorbar_args(colorbar_param)
    
    for i, (id, id_name) in enumerate(IDs.items()):
        file = h5py.File(DATA_PATH+f"{id}.hdf5", "r")
        x_array, y_array, c_array = [], [], []
        
        if i ==0:
            axs[i].text(1e3, 1e9,  fr'$\texttt{{{id_name}}}$', color='black', 
                bbox=dict(facecolor='silver', edgecolor='none', alpha=0.4, boxstyle='round, pad=0.2'), horizontalalignment='right')
        else:
            axs[i].text(1e3, 1.5e8,  fr'$\texttt{{{id_name}}}$', color='black', 
                bbox=dict(facecolor='silver', edgecolor='none', alpha=0.4, boxstyle='round, pad=0.2'), horizontalalignment='right')

        for idx in file.keys():
            if file[f'{idx}'].attrs.get('subhalo_of') is not None:
                subhalo_idx = idx

                if np.log10(file[str(subhalo_idx)]['tree_data']['bound_mass_dm'][0]) > 9: # MINIMUM satellite mass = 10^9
                    data_subhalo = file[f'{subhalo_idx}']
                    pericenter = data_subhalo['tree_data']['pericenter'][1]
                    z_accr_type_idx, accretion = data_subhalo['tree_data']['accretion']
                    
                    if id_name == "CDM":
                        r0, rho0 = data_subhalo['halo_data']['nfw_fit']
                        density_fit = fit_nfw_model(np.array([0.15]), r0, rho0)
                        density_fit = 10**density_fit
                    else:
                        r0, rho0 = data_subhalo['halo_data']['iso_fit']
                        density_fit = fit_isothermal_model(np.array([0.15]*2), r0, rho0)[0]
                        density_fit = 10**density_fit

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
                    if p>6 and p<1.5e3:
                        im = axs[i].scatter(x=pericenter, y=density_fit, marker='+',linewidths=1.1, norm=norm, c=c, cmap=cmap, alpha= 0.7 if colorbar_param in ["mass_0", "mass_peak"] else 0.9)
                        x_array.append(p)
                        y_array.append(density_fit)
                        c_array.append(c)
                  
        x_array = np.array(x_array)
        y_array = np.array(y_array)
        c_array = np.array(c_array)
        
        if print_correlation: 
            get_correlations(x_array, y_array)
        
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
            plot_median_relation(axs[i], bins, x_array, y_array, color='lightslategrey')

        file.close() 
                      
    # axis stuff
    for i, ax in enumerate(axs):
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(6e0, 1.5e3)
        if i ==0:
            ax.set_ylim(1e7, 2e9)
        else:
            ax.set_ylim(1e6, 3e8)
            
    for axi in [3, 4, 5]:
        axs[axi].set_xlabel(r'$r_{{p}}\ [\mathrm{kpc}]$')
    for axi in [0, 3]:
        axs[axi].set_ylabel(r'$\rho(150\ \mathrm{pc})\ [\mathrm{M}_\odot \ \mathrm{kpc}^3]$')
                
    # colorbar stuff
    cbar = fig.colorbar(im, ax=axs.ravel().tolist(), label=COLORBAR_DICT[colorbar_param], aspect=40, fraction=0.02, pad=0.04)
    if colorbar_param == "accretion":
        cbar.ax.set_yticks([0, 1, 2]) 
    elif colorbar_param in ["mass_0", "mass_peak"]:
        cbar.ax.set_yticks([9, 10, 11, 12]) 

    plt.subplots_adjust(hspace=0.2, wspace=0.2, right=.86)
    if filename is not None:
        fig.savefig(f"figures/{filename}.png",dpi=300)
    plt.show()
    
 
 
##############################################################################################
### Density at 150pc versus velocity  (V_max(z=0), V_peak, V(r_fiducial)) plotting routine ###
##############################################################################################
def plot_density_150pc_velocity(velocity:str="V_max", print_correlation=False, filename:str=None):
    
    print(f'Plotting density at 150pc versus velocity {velocity}!')

    fig, axs = plt.subplots(2, 3, sharex=True, sharey=False, figsize=(13, 8), dpi=200, facecolor='white')
    axs = axs.flatten()
    
    for i, (id, id_name) in enumerate(IDs.items()):
        file = h5py.File(DATA_PATH+f"{id}.hdf5", "r")
        x_array, y_array = [], []
        
        if i ==0:
            axs[i].text(16, 1e9,  fr'$\texttt{{{id_name}}}$', color='black', 
                bbox=dict(facecolor='silver', edgecolor='none', alpha=0.4, boxstyle='round, pad=0.2'), horizontalalignment='left')
        else:
            axs[i].text(16, 1.5e8,  fr'$\texttt{{{id_name}}}$', color='black', 
                bbox=dict(facecolor='silver', edgecolor='none', alpha=0.4, boxstyle='round, pad=0.2'), horizontalalignment='left')

        for idx in file.keys():
            if file[f'{idx}'].attrs.get('subhalo_of') is not None:
                subhalo_idx = idx

                if np.log10(file[str(subhalo_idx)]['tree_data']['bound_mass_dm'][0]) > 9: # MINIMUM satellite mass = 10^9
                    data_subhalo = file[f'{subhalo_idx}']
                    pericenter = data_subhalo['tree_data']['pericenter'][1]
                    z_accr_type_idx, accretion = data_subhalo['tree_data']['accretion']
                    
                    if velocity == "v_peak" : vel = data_subhalo['tree_data']['Vmax'][int(z_accr_type_idx)]
                    elif velocity == "v_max": vel = data_subhalo['tree_data']['Vmax'][0]
                    elif velocity == "v_fid": vel = data_subhalo['halo_data']['rotation_at_fiducial_radius'][0]
                    else: print("Wrong velocity key! Choose between 'v_max', 'v_peak', or 'v_fid'.")
                    
                    if id_name == "CDM":
                        r0, rho0 = data_subhalo['halo_data']['nfw_fit']
                        density_fit = fit_nfw_model(np.array([0.15]), r0, rho0)
                        density_fit = 10**density_fit
                    else:
                        r0, rho0 = data_subhalo['halo_data']['iso_fit']
                        density_fit = fit_isothermal_model(np.array([0.15]*2), r0, rho0)[0]
                        density_fit = 10**density_fit
                    
                    im = axs[i].scatter(x=vel, y=density_fit, marker='o', alpha=0.7, c='olivedrab', linewidths=0)
                    x_array.append(vel)
                    y_array.append(density_fit)
                  
        x_array = np.array(x_array)
        y_array = np.array(y_array)
        
        if print_correlation: 
            get_correlations(x_array, y_array)
        
        bins = np.arange(10, 80, 5)
        plot_median_relation(axs[i], bins, x_array, y_array, color='lightslategrey')

        file.close() 
                      
    # axis stuff
    for i, ax in enumerate(axs):
        ax.set_yscale('log')
        ax.set_xlim(10, 80)
        if i ==0:
            ax.set_ylim(1e7, 2e9)
        else:
            ax.set_ylim(1e6, 3e8)
            
    for axi in [3, 4, 5]:
        if velocity == "v_peak" : xlabel = fr'$\mathrm{{V_{{peak}}}}$ $\mathrm{{[km\ s^{{-1}}]}}$'
        elif velocity == "v_max": xlabel= fr'$\mathrm{{V_{{max}}(z=0)}}$ $\mathrm{{[km\ s^{{-1}}]}}$'
        elif velocity == "v_fid": xlabel= fr'$\mathrm{{V_{{circ}}(r_{{fid}})}}$ $\mathrm{{[km\ s^{{-1}}]}}$'
        axs[axi].set_xlabel(xlabel)
    for axi in [0, 3]:
        axs[axi].set_ylabel(r'$\rho(150\ \mathrm{pc})\ [\mathrm{M}_\odot \ \mathrm{kpc}^3]$')

    plt.subplots_adjust(hspace=0.2, wspace=0.2, right=.86)
    if filename is not None:
        fig.savefig(f"figures/{filename}.png",dpi=300)
    plt.show()
    
   
##############################################################################
### Max circular velocity / max circular velocity at peak plotting routine ###
##############################################################################
def plot_vmax(colorbar_param, print_correlation=False, filename:str=None):
    """Vmax rp"""
    
    print(f'Max circular velocity  with {colorbar_param} colorbar!')

    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(13, 8), dpi=200, facecolor='white')
    axs = axs.flatten()
    cmap, norm = colorbar_args(colorbar_param)

    for i, (id, id_name) in enumerate(IDs.items()):
        file = h5py.File(DATA_PATH+f"{id}.hdf5", "r")
        axs[i].text(10, 14, fr'$\texttt{{{id_name}}}$', color='black', 
            bbox=dict(facecolor='silver', edgecolor='none', alpha=0.4, boxstyle='round, pad=0.2'))

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
                        im = axs[i].scatter(x=pericenter, y=vmax, marker='+', linewidths=1, norm=norm, c=c, cmap=cmap, alpha= 0.7 if colorbar_param in ["mass_0", "mass_peak"] else 0.9)
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
                plot_median_relation(axs[i], bins, x_array, y_array, color='lightslategrey')

        file.close() 
           
    # axis stuff
    for ax in axs:
        ax.set_xscale('log')
        ax.set_xlim(6e0, 1.5e3)
        ax.set_ylim(10, 80)
        ax.set_yticks([20, 40, 60, 80])

    for axi in [3, 4, 5]:
        axs[axi].set_xlabel(r'$r_{{p}}\ [\mathrm{kpc}]$')
    for axi in [0, 3]:
        axs[axi].set_ylabel(fr'$\mathrm{{V_{{max}}}}(z=0)$')

    # colorbar stuff
    cbar = fig.colorbar(im, ax=axs.ravel().tolist(), label=COLORBAR_DICT[colorbar_param], aspect=40, fraction=0.02, pad=0.04)
    if colorbar_param == "accretion":
        cbar.ax.set_yticks([0, 1, 2]) 
    elif colorbar_param in ["mass_0", "mass_peak"]:
        cbar.ax.set_yticks([9, 10, 11, 12]) 

    plt.subplots_adjust(hspace=0.1, wspace=0.1, right=.86)
    if filename is not None:
        fig.savefig(f"figures/{filename}.png",dpi=300)
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
        axs[i].text(10, 0.2, fr'$\texttt{{{id_name}}}$', color='black', 
            bbox=dict(facecolor='silver', edgecolor='none', alpha=0.4, boxstyle='round, pad=0.2'))

        x_array, y_array, c_array = [], [], []
        for idx in file.keys():
            if file[f'{idx}'].attrs.get('subhalo_of') is not None:
                subhalo_idx = idx

                if np.log10(file[str(subhalo_idx)]['tree_data']['bound_mass_dm'][0]) > 9: # MINIMUM satellite mass = 10^9
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
                        mass_peak = data_subhalo['tree_data']['bound_mass_dm'][int(z_accr_type_idx)]
                        c = np.log10(mass_peak)
                    
                    p = pericenter[0] if pericenter.shape==(1,) else pericenter
                    if p>6 and p<1.5e3:
                        im = axs[i].scatter(x=pericenter, y=vmax/vpeak, marker='+', linewidths=1, norm=norm, c=c, cmap=cmap, alpha= 0.7 if colorbar_param in ["mass_0", "mass_peak"] else 0.9)
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
                plot_median_relation(axs[i], bins, x_array, y_array, color='lightslategrey')

        file.close() 
           
    # axis stuff
    for ax in axs:
        ax.set_xscale('log')
        ax.set_xlim(6e0, 1.5e3)
        ax.set_ylim(0, 2)
        # ax.set_yticks([20, 40, 60, 80])

    for axi in [3, 4, 5]:
        axs[axi].set_xlabel(r'$r_{{p}}\ [\mathrm{kpc}]$')
    for axi in [0, 3]:
        axs[axi].set_ylabel(fr'$\mathrm{{V_{{max}}}}(z=0)/\mathrm{{V_{{peak}}}}$')

    # colorbar stuff
    cbar = fig.colorbar(im, ax=axs.ravel().tolist(), label=COLORBAR_DICT[colorbar_param], aspect=40, fraction=0.02, pad=0.04)
    if colorbar_param == "accretion":
        cbar.ax.set_yticks([0, 1, 2]) 
    elif colorbar_param in ["mass_0", "mass_peak"]:
        cbar.ax.set_yticks([9, 10, 11, 12]) 

    plt.subplots_adjust(hspace=0.1, wspace=0.1, right=.86)
    if filename is not None:
        fig.savefig(f"figures/{filename}.png",dpi=300)
    plt.show()
    
    
############################################
### Vmax vs Rmax at z=0 plotting routine ###
############################################
def plot_vmax_vs_rmax(colorbar_param, print_correlation=False, filename:str=None):
    """E.g. Fig.1 in Robles and Bullock 2020"""
    
    print(f'Vmax vs Rmax {colorbar_param} colorbar!')

    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(13, 8), dpi=200, facecolor='white')
    axs = axs.flatten()
    cmap, norm = colorbar_args(colorbar_param)

    for i, (id, id_name) in enumerate(IDs.items()):
        file = h5py.File(DATA_PATH+f"{id}.hdf5", "r")
        axs[i].text(14, 18, fr'$\texttt{{{id_name}}}$', color='black', horizontalalignment='right',
            bbox=dict(facecolor='silver', edgecolor='none', alpha=0.4, boxstyle='round, pad=0.2'))

        x_array, y_array, c_array = [], [], []
        for idx in file.keys():
            if file[f'{idx}'].attrs.get('subhalo_of') is not None:
                subhalo_idx = idx

                if np.log10(file[str(subhalo_idx)]['tree_data']['bound_mass_dm'][0]) > 9: # MINIMUM satellite mass = 10^9
                    data_subhalo = file[f'{subhalo_idx}']
                    z_accr_type_idx, accretion = data_subhalo['tree_data']['accretion']
                    if 'Rmax' in data_subhalo['tree_data'].keys(): 
                        x = data_subhalo['tree_data']['Rmax'][...]
                    else: continue
                    y = data_subhalo['tree_data']['Vmax'][0]

                    if colorbar_param == 'accretion':
                        c = accretion
                    elif colorbar_param == 'mass_0':
                        mass_0 = data_subhalo['tree_data']['bound_mass_dm'][0] 
                        c = np.log10(mass_0)
                    elif colorbar_param == 'mass_peak':
                        mass_peak = data_subhalo['tree_data']['bound_mass_dm'][int(z_accr_type_idx)]
                        c = np.log10(mass_peak)
                    elif colorbar_param == 'pericenter':
                        pericenter = data_subhalo['tree_data']['pericenter'][1]
                        c = np.log10(pericenter[0] if pericenter.shape==(1,) else pericenter)
            
                    im = axs[i].scatter(x=x, y=y, marker='+', linewidths=1, norm=norm, c=c, cmap=cmap, alpha= 0.7 if colorbar_param in ["mass_0", "mass_peak"] else 0.9)
                    x_array.append(x)
                    y_array.append(y)
                    c_array.append(c)
                  
        x_array = np.array(x_array)
        y_array = np.array(y_array)
        c_array = np.array(c_array)
            
        if print_correlation:
            get_correlations(x_array, y_array, id_name)
            
        bins = np.arange(1, 15, 3)
        if colorbar_param in ["mass_0", "mass_peak"]:
            mask_9 = c_array<10
            mask_10 = (c_array>=10)*(c_array<11)
            mask_11 = c_array>=11
            
            for jm, (mask, color) in enumerate(zip([mask_9, mask_10], [myblue, myred])):
                x_mask = x_array[mask]
                y_mask = y_array[mask]
                plot_median_relation(axs[i], bins, x_mask, y_mask, color=color)
            
        else:
            if id_name == "CDM":
                x_CDM = x_array
                y_CDM = y_array
                plot_median_relation(axs[i], bins, x_array, y_array, color='k')
            else:
                plot_median_relation(axs[i], bins, x_CDM, y_CDM, errorbars=False, color='k')
                plot_median_relation(axs[i], bins, x_array, y_array, color='lightslategrey')

        file.close() 
           
    # axis stuff
    for ax in axs:
        ax.set_xlim(1, 15)
        ax.set_ylim(15, 65)
        ax.set_yticks([20, 40, 60])

    for axi in [3, 4, 5]:
        axs[axi].set_xlabel(r'$\mathrm{r_{max}\ [kpc]}$')
    for axi in [0, 3]:
        axs[axi].set_ylabel(r'$\mathrm{V_{max}}(z=0) [\mathrm{\ km s^{-1}}]$')

    # colorbar stuff
    cbar = fig.colorbar(im, ax=axs.ravel().tolist(), label=COLORBAR_DICT[colorbar_param], aspect=40, fraction=0.02, pad=0.04)
    if colorbar_param == "accretion":
        cbar.ax.set_yticks([0, 1, 2]) 
    elif colorbar_param in ["mass_0", "mass_peak"]:
        cbar.ax.set_yticks([9, 10, 11, 12]) 

    plt.subplots_adjust(hspace=0.1, wspace=0.1, right=.86)
    if filename is not None:
        fig.savefig(f"figures/{filename}.png",dpi=300)
    plt.show()
 
 
#############################################################
### Circular velocity at fiducial radius plotting routine ###
#############################################################
def plot_circular_velocity_fiducial_radius(colorbar_param, print_correlation=False, filename:str=None):
    
    print(f'Plotting circular velocity at fiducial radius versus pericenter distance with {colorbar_param} colorbar!')

    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(13, 8), dpi=200, facecolor='white')
    axs = axs.flatten()
    cmap, norm = colorbar_args(colorbar_param)

    for i, (id, id_name) in enumerate(IDs.items()):
        file = h5py.File(DATA_PATH+f"{id}.hdf5", "r")
        axs[i].text(10, 78, fr'$\texttt{{{id_name}}}$', color='black', 
            bbox=dict(facecolor='silver', edgecolor='none', alpha=0.4, boxstyle='round, pad=0.2'))

        x_array, y_array, c_array = [], [], []
        for idx in file.keys():
            if file[f'{idx}'].attrs.get('subhalo_of') is not None:
                subhalo_idx = idx

                if np.log10(file[str(subhalo_idx)]['tree_data']['bound_mass_dm'][0]) > 9: # MINIMUM satellite mass = 10^9
                    data_subhalo = file[f'{subhalo_idx}']
                    pericenter = data_subhalo['tree_data']['pericenter'][1]
                    rotation_at_fiducial_radius = data_subhalo['halo_data']['rotation_at_fiducial_radius'][:]
                    z_accr_type_idx, accretion = data_subhalo['tree_data']['accretion']

                    if colorbar_param == 'accretion':
                        c = accretion
                    elif colorbar_param == 'mass_0':
                        mass_0 = data_subhalo['tree_data']['bound_mass_dm'][0] 
                        c = np.log10(mass_0)
                    elif colorbar_param == 'mass_peak':
                        mass_peak = data_subhalo['tree_data']['bound_mass_dm'][int(z_accr_type_idx)]
                        c = np.log10(mass_peak)
                    
                    p = pericenter[0] if pericenter.shape==(1,) else pericenter
                    if p>6 and p<1.5e3 and rotation_at_fiducial_radius[0]>3 and rotation_at_fiducial_radius[0]<100:
                        im = axs[i].scatter(x=pericenter, y=rotation_at_fiducial_radius, marker='+', linewidths=1, norm=norm, c=c, cmap=cmap, alpha= 0.7 if colorbar_param in ["mass_0", "mass_peak"] else 0.9)
                        x_array.append(p)
                        c_array.append(c)
                        y_array.append(rotation_at_fiducial_radius[0])
                  
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
            if id_name == "CDM":
                x_CDM = x_array
                y_CDM = y_array
                plot_median_relation(axs[i], bins, x_array, y_array, color='k')
            else:
                plot_median_relation(axs[i], bins, x_CDM, y_CDM, errorbars=False, color='k')
                plot_median_relation(axs[i], bins, x_array, y_array, color='lightslategrey')

        file.close() 
           
    # axis stuff
    for ax in axs:
        ax.set_xscale('log')
        ax.set_xlim(6e0, 1.5e3)
        ax.set_ylim(5, 90)
        ax.set_yticks([20, 40, 60, 80])

    for axi in [3, 4, 5]:
        axs[axi].set_xlabel(r'$r_{{p}}\ [\mathrm{kpc}]$')
    for axi in [0, 3]:
        axs[axi].set_ylabel(fr'$\mathrm{{V_{{circ}}}}(r_\mathrm{{fid}})\ [\mathrm{{km/s}}]$')

    # colorbar stuff
    cbar = fig.colorbar(im, ax=axs.ravel().tolist(), label=COLORBAR_DICT[colorbar_param], aspect=40, fraction=0.02, pad=0.04)
    if colorbar_param == "accretion":
        cbar.ax.set_yticks([0, 1, 2])  
    elif colorbar_param in ["mass_0", "mass_peak"]:
        cbar.ax.set_yticks([9, 10, 11, 12]) 

    plt.subplots_adjust(hspace=0.1, wspace=0.1, right=.86)
    if filename is not None:
        fig.savefig(f"figures/{filename}.png",dpi=300)
    plt.show()
    
    
###################################################
### Circular velocity at 2 kpc plotting routine ###
###################################################
def plot_circular_velocity_2kpc(colorbar_param, print_correlation=False, filename:str=None):
    
    print(f'Plotting circular velocity at 2kpc versus pericenter distance with {colorbar_param} colorbar!')

    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(13, 8), dpi=200, facecolor='white')
    axs = axs.flatten()
    cmap, norm = colorbar_args(colorbar_param)

    for i, (id, id_name) in enumerate(IDs.items()):
        file = h5py.File(DATA_PATH+f"{id}.hdf5", "r")
        axs[i].text(10, 78, fr'$\texttt{{{id_name}}}$', color='black', 
            bbox=dict(facecolor='silver', edgecolor='none', alpha=0.4, boxstyle='round, pad=0.2'))

        x_array, y_array, c_array = [], [], []
        for idx in file.keys():
            if file[f'{idx}'].attrs.get('subhalo_of') is not None:
                subhalo_idx = idx

                if np.log10(file[str(subhalo_idx)]['tree_data']['bound_mass_dm'][0]) > 9: # MINIMUM satellite mass = 10^9
                    data_subhalo = file[f'{subhalo_idx}']
                    pericenter = data_subhalo['tree_data']['pericenter'][1]
                    radius_rotation = data_subhalo['halo_data']['radius_rotation'][:]
                    rotation_at_radius = data_subhalo['halo_data']['rotation_at_radius'][:]
                    z_accr_type_idx, accretion = data_subhalo['tree_data']['accretion']

                    if colorbar_param == 'accretion':
                        c = accretion
                    elif colorbar_param == 'mass_0':
                        mass_0 = data_subhalo['tree_data']['bound_mass_dm'][0] 
                        c = np.log10(mass_0)
                    elif colorbar_param == 'mass_peak':
                        mass_peak = data_subhalo['tree_data']['bound_mass_dm'][int(z_accr_type_idx)]
                        c = np.log10(mass_peak)
                    
                    p = pericenter[0] if pericenter.shape==(1,) else pericenter
                    if p>6 and p<1.5e3 and rotation_at_radius[0]>3 and rotation_at_radius[0]<100:
                        im = axs[i].scatter(x=pericenter, y=rotation_at_radius, marker='+', linewidths=1, norm=norm, c=c, cmap=cmap, alpha= 0.7 if colorbar_param in ["mass_0", "mass_peak"] else 0.9)
                        x_array.append(p)
                        c_array.append(c)
                        y_array.append(rotation_at_radius[0])
                  
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
            if id_name == "CDM":
                x_CDM = x_array
                y_CDM = y_array
                plot_median_relation(axs[i], bins, x_array, y_array, color='k')
            else:
                plot_median_relation(axs[i], bins, x_CDM, y_CDM, errorbars=False, color='k')
                plot_median_relation(axs[i], bins, x_array, y_array, color='lightslategrey')

        file.close() 
           
    # axis stuff
    for ax in axs:
        ax.set_xscale('log')
        ax.set_xlim(6e0, 1.5e3)
        ax.set_ylim(5, 90)
        ax.set_yticks([20, 40, 60, 80])

    for axi in [3, 4, 5]:
        axs[axi].set_xlabel(r'$r_{{p}}\ [\mathrm{kpc}]$')
    for axi in [0, 3]:
        axs[axi].set_ylabel(fr'$\mathrm{{V_{{circ}}}}(2 \mathrm{{kpc}})\ [\mathrm{{km/s}}]$')

    # colorbar stuff
    cbar = fig.colorbar(im, ax=axs.ravel().tolist(), label=COLORBAR_DICT[colorbar_param], aspect=40, fraction=0.02, pad=0.04)
    if colorbar_param == "accretion":
        cbar.ax.set_yticks([0, 1, 2]) 
    elif colorbar_param in ["mass_0", "mass_peak"]:
        cbar.ax.set_yticks([9, 10, 11, 12]) 

    plt.subplots_adjust(hspace=0.1, wspace=0.1, right=.86)
    if filename is not None:
        fig.savefig(f"figures/{filename}.png",dpi=300)
    plt.show()
    
       