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
myblue = mycmap(0.1)
myred =  mycmap(0.9)

###########################
### Colorbar parameters ###
###########################

COLORBAR_DICT = {
    "accretion" : fr'$z_\mathrm{{accretion}}$',
    "mass_0" : fr'$\log_{{10}}\mathrm{{M_{{bound}}}}$ $\mathrm{{[M_\odot]}}$',
    "mass_peak" : fr'$\log_{{10}}\mathrm{{M_{{peak}}}}$ $\mathrm{{[M_\odot]}}$',
}


def colorbar_args(colorbar_param):
    if colorbar_param == 'accretion':
        vmin, vmax = 0, 2.6
        cmap = mycmap 
        norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=((vmax+vmin)/2), vmax=vmax)

    elif colorbar_param == 'mass_0':
        vmin, vmax = 9, 12
        cmap = matplotlib.colors.ListedColormap([myred, myblue, "olivedrab"])
        norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=((vmax+vmin)/2), vmax=vmax)

    elif colorbar_param == 'mass_peak':
        vmin, vmax = 9, 12
        cmap = matplotlib.colors.ListedColormap([myred, myblue, "olivedrab"])
        norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=((vmax+vmin)/2), vmax=vmax)
        
    else:
        raise KeyError("Wrong key for colorbar_param.")
        
    return cmap, norm


##################################################################
### Density at 150pc versus pericenter radius plotting routine ###
##################################################################
def plot_density_150pc(colorbar_param, print_correlation=False):
    
    print(f'Plotting density at 150pc versus pericenter distance with {colorbar_param} colorbar!')

    fig, axs = plt.subplots(2, 3, sharex=True, sharey=False, figsize=(13, 8), dpi=200, facecolor='white')
    axs = axs.flatten()
    cmap, norm = colorbar_args(colorbar_param)
    
    p_bins = []
    for i, (id, id_name) in enumerate(IDs.items()):
        file = h5py.File(DATA_PATH+f"{id}.hdf5", "r")
        p_array, r_array, c_array = [], [], []
        
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
                    
                    p = pericenter[0] if pericenter.shape==(1,) else pericenter
                    if p>6 and p<1.5e3:
                        im = axs[i].scatter(x=pericenter, y=density_fit, marker='+',ms=2, linewidths=1, norm=norm, c=c, cmap=cmap, alpha= 0.7 if colorbar_param in ["mass_0", "mass_peak"] else 0.9)
                        p_array.append(p)
                        r_array.append(density_fit)
                        c_array.append(c)
                  
        num_bins = 10
        p_array = np.array(p_array)
        r_array = np.array(r_array)
        c_array = np.array(c_array)
        
        if print_correlation:
            pearson = scipy.stats.pearsonr(p_array, r_array)
            spearman = scipy.stats.spearmanr(p_array, r_array)
            print(id_name)
            print("Pearson's r:", pearson)
            print("Spearman's rho:", spearman)
            print("\n")
        
        if colorbar_param in ["mass_0", "mass_peak"]:
            mask_9 = c_array<10
            mask_10 = (c_array>=10)*(c_array<11)
            mask_11 = c_array>=11
            
            for jm, (mask, num_bins, color) in enumerate(zip([mask_9, mask_10], [8, 4], [myred, myblue])):
                # if i == 0: # use same binning everywhere
                    # p_bins.append(np.logspace(np.log10(20), np.log10(600), num_bins))

                p_bins = np.arrange(1, 3, 0.2)
                p_bins = 10**p_bins
                indx = np.digitize(p_array, p_bins, right=True)
                # r_bins_value = [r_array[mask][np.digitize(p_array[mask], np.logspace(np.log10(20), np.log10(600), num_bins)) == i] for i in range(num_bins)]
                # r_bins_value = [r if r.size!=0 else r_bins_value[i+1] for i, r in enumerate(r_bins_value)]

                p_bins_medians = np.array([np.median(p_array[indx == idx]) for idx in indx])
                r_bins_medians = np.array([np.median(r_array[indx == idx]) for idx in indx])
                r_bins_16 = np.array([np.percentile(r_array[indx == idx], 16) for idx in indx])
                r_bins_84 = np.array([np.percentile(r_array[indx == idx], 84) for idx in indx])

                # r_bins_medians = np.array([np.median(bin_y) for bin_y in r_bins_value])
                # r_bins_16 = np.array([np.percentile(bin_y, 16) for bin_y in r_bins_value])
                # r_bins_84 = np.array([np.percentile(bin_y, 84) for bin_y in r_bins_value])

                axs[i].plot(p_bins_medians, r_bins_medians, color=color, ls='-')
                axs[i].plot(p_bins_medians, r_bins_16, '--', color=color)
                axs[i].plot(p_bins_medians, r_bins_84, '--', color=color)

                # axs[i].plot(p_bins[jm], r_bins_medians, color=color, ls = '-')
                # axs[i].plot(p_bins[jm], r_bins_16,  '--', color=color)
                # axs[i].plot(p_bins[jm], r_bins_84,  '--', color=color)
                # axs[i].fill_between(
                #     p_bins[jm],
                #     r_bins_16,
                #     r_bins_84,
                #     color=color,
                #     alpha=0.07,
                # )
            
        else:
            if i == 0: # use same binning everywhere
                p_bins = np.logspace(np.log10(20), np.log10(600), num_bins)
            r_bins_value = [r_array[np.digitize(p_array, np.logspace(np.log10(20), np.log10(600), num_bins + 1)) == i] for i in range(1, num_bins + 1)]
            r_bins_medians = np.array([np.median(bin_y) for bin_y in r_bins_value])
            r_bins_16 = np.array([np.percentile(bin_y, 16) for bin_y in r_bins_value])
            r_bins_84 = np.array([np.percentile(bin_y, 84) for bin_y in r_bins_value])    

            color='k'
            axs[i].plot(p_bins, r_bins_medians, color=color)
            axs[i].plot(p_bins, r_bins_16, '--', color=color)
            axs[i].plot(p_bins, r_bins_84,  '--',color=color)
            axs[i].fill_between(
                p_bins,
                r_bins_16,
                r_bins_84,
                color=color,
                alpha=0.07,
            )

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
    plt.show()
    
    
#############################################################
### Circular velocity at fiducial radius plotting routine ###
#############################################################
def plot_circular_velocity_fiducial_radius(colorbar_param, print_correlation=False):
    
    print(f'Plotting circular velocity at fiducial radius versus pericenter distance with {colorbar_param} colorbar!')

    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(13, 8), dpi=200, facecolor='white')
    axs = axs.flatten()
    cmap, norm = colorbar_args(colorbar_param)
    p_bins, r_bins_medians_CDM = [], []

    for i, (id, id_name) in enumerate(IDs.items()):
        file = h5py.File(DATA_PATH+f"{id}.hdf5", "r")
        axs[i].text(10, 78, fr'$\texttt{{{id_name}}}$', color='black', 
            bbox=dict(facecolor='silver', edgecolor='none', alpha=0.4, boxstyle='round, pad=0.2'))

        p_array, r_array, c_array = [], [], []
        for idx in file.keys():
            if file[f'{idx}'].attrs.get('subhalo_of') is not None:
                subhalo_idx = idx

                if np.log10(file[str(subhalo_idx)]['tree_data']['bound_mass_dm'][0]) > 9: # MINIMUM satellite mass = 10^9
                    data_subhalo = file[f'{subhalo_idx}']
                    pericenter = data_subhalo['tree_data']['pericenter'][1]
                    fiducial_radius_rotation = data_subhalo['halo_data']['fiducial_radius_rotation'][:]
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
                        p_array.append(p)
                        c_array.append(c)
                        r_array.append(rotation_at_fiducial_radius[0])
                  
        num_bins = 8
        p_array = np.array(p_array)
        r_array = np.array(r_array)
        c_array = np.array(c_array)
        
        if print_correlation:
            pearson = scipy.stats.pearsonr(p_array, r_array)
            spearman = scipy.stats.spearmanr(p_array, r_array)
            print(id_name)
            print("Pearson's r:", pearson)
            print("Spearman's rho:", spearman)
            print("\n")
            
        if colorbar_param in ["mass_0", "mass_peak"]:
            mask_9 = c_array<10
            mask_10 = (c_array>=10)*(c_array<11)
            mask_11 = c_array>=11
            
            for jm, (mask, num_bins, color) in enumerate(zip([mask_9, mask_10], [8, 4], [myred, myblue])):
                if i == 0: # use same binning everywhere
                    p_bins.append(np.logspace(np.log10(20), np.log10(600), num_bins))
                
                r_bins_value = [r_array[mask][np.digitize(p_array[mask], np.logspace(np.log10(20), np.log10(600), num_bins)) == i] for i in range(num_bins)]
                r_bins_value = [r if r.size!=0 else r_bins_value[i+1] for i, r in enumerate(r_bins_value)]
                
                r_bins_medians = np.array([np.median(bin_y) for bin_y in r_bins_value])
                r_bins_16 = np.array([np.percentile(bin_y, 16) for bin_y in r_bins_value])
                r_bins_84 = np.array([np.percentile(bin_y, 84) for bin_y in r_bins_value])
                                        
                axs[i].plot(p_bins[jm], r_bins_medians, color=color, ls = '-')
                axs[i].plot(p_bins[jm], r_bins_16,  '--', color=color)
                axs[i].plot(p_bins[jm], r_bins_84,  '--', color=color)
                axs[i].fill_between(
                    p_bins[jm],
                    r_bins_16,
                    r_bins_84,
                    color=color,
                    alpha=0.07,
                )
            
        else:
            
            if i == 0: # use same binning everywhere
                p_bins = np.logspace(np.log10(20), np.log10(600), num_bins)
            r_bins_value = [r_array[np.digitize(p_array, np.logspace(np.log10(20), np.log10(600), num_bins + 1)) == i] for i in range(1, num_bins + 1)]
            r_bins_medians = np.array([np.median(bin_y) for bin_y in r_bins_value])
            r_bins_16 = np.array([np.percentile(bin_y, 16) for bin_y in r_bins_value])
            r_bins_84 = np.array([np.percentile(bin_y, 84) for bin_y in r_bins_value])    

            if id_name == "CDM":
                color='lightslategrey'
                r_bins_medians_CDM = r_bins_medians
            else:
                color='k'
                axs[i].plot(p_bins, r_bins_medians_CDM, color='lightslategrey')
            axs[i].plot(p_bins, r_bins_medians, color=color)
            axs[i].plot(p_bins, r_bins_16, '--', color=color)
            axs[i].plot(p_bins, r_bins_84,  '--',color=color)
            axs[i].fill_between(
                p_bins,
                r_bins_16,
                r_bins_84,
                color=color,
                alpha=0.07,
            )

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
    plt.show()
    
    
###################################################
### Circular velocity at 2 kpc plotting routine ###
###################################################
def plot_circular_velocity_2kpc(colorbar_param, print_correlation=False):
    
    print(f'Plotting circular velocity at 2kpc versus pericenter distance with {colorbar_param} colorbar!')

    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(13, 8), dpi=200, facecolor='white')
    axs = axs.flatten()
    cmap, norm = colorbar_args(colorbar_param)
    p_bins, r_bins_medians_CDM = [], []

    for i, (id, id_name) in enumerate(IDs.items()):
        file = h5py.File(DATA_PATH+f"{id}.hdf5", "r")
        axs[i].text(10, 78, fr'$\texttt{{{id_name}}}$', color='black', 
            bbox=dict(facecolor='silver', edgecolor='none', alpha=0.4, boxstyle='round, pad=0.2'))

        p_array, r_array, c_array = [], [], []
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
                        p_array.append(p)
                        c_array.append(c)
                        r_array.append(rotation_at_radius[0])
                  
        num_bins = 8
        p_array = np.array(p_array)
        r_array = np.array(r_array)
        c_array = np.array(c_array)
            
        if print_correlation:
            pearson = scipy.stats.pearsonr(p_array, r_array)
            spearman = scipy.stats.spearmanr(p_array, r_array)
            print(id_name)
            print("Pearson's r:", pearson)
            print("Spearman's rho:", spearman)
            print("\n")
            
        if colorbar_param in ["mass_0", "mass_peak"]:
            mask_9 = c_array<10
            mask_10 = (c_array>=10)*(c_array<11)
            mask_11 = c_array>=11
            
            for jm, (mask, num_bins, color) in enumerate(zip([mask_9, mask_10], [8, 4], [myred, myblue])):
                if i == 0: # use same binning everywhere
                    p_bins.append(np.logspace(np.log10(20), np.log10(600), num_bins))
                
                r_bins_value = [r_array[mask][np.digitize(p_array[mask], np.logspace(np.log10(20), np.log10(600), num_bins)) == i] for i in range(num_bins)]
                r_bins_value = [r if r.size!=0 else r_bins_value[i+1] for i, r in enumerate(r_bins_value)]
                
                r_bins_medians = np.array([np.median(bin_y) for bin_y in r_bins_value])
                r_bins_16 = np.array([np.percentile(bin_y, 16) for bin_y in r_bins_value])
                r_bins_84 = np.array([np.percentile(bin_y, 84) for bin_y in r_bins_value])
                                        
                axs[i].plot(p_bins[jm], r_bins_medians, color=color, ls = '-')
                axs[i].plot(p_bins[jm], r_bins_16,  '--', color=color)
                axs[i].plot(p_bins[jm], r_bins_84,  '--', color=color)
                axs[i].fill_between(
                    p_bins[jm],
                    r_bins_16,
                    r_bins_84,
                    color=color,
                    alpha=0.07,
                )
            
        else:
            
            if i == 0: # use same binning everywhere
                p_bins = np.logspace(np.log10(20), np.log10(600), num_bins)
            r_bins_value = [r_array[np.digitize(p_array, np.logspace(np.log10(20), np.log10(600), num_bins + 1)) == i] for i in range(1, num_bins + 1)]
            r_bins_medians = np.array([np.median(bin_y) for bin_y in r_bins_value])
            r_bins_16 = np.array([np.percentile(bin_y, 16) for bin_y in r_bins_value])
            r_bins_84 = np.array([np.percentile(bin_y, 84) for bin_y in r_bins_value])    

            if id_name == "CDM":
                color='lightslategrey'
                r_bins_medians_CDM = r_bins_medians
            else:
                color='k'
                axs[i].plot(p_bins, r_bins_medians_CDM, color='lightslategrey')
            axs[i].plot(p_bins, r_bins_medians, color=color)
            axs[i].plot(p_bins, r_bins_16, '--', color=color)
            axs[i].plot(p_bins, r_bins_84,  '--',color=color)
            axs[i].fill_between(
                p_bins,
                r_bins_16,
                r_bins_84,
                color=color,
                alpha=0.07,
            )

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
    plt.show()