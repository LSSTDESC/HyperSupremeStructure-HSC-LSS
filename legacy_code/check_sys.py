from __future__ import print_function
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
#import healpy as hp
from scipy.stats import binned_statistic
import flatmaps as fm
import os
from astropy.io import fits
from optparse import OptionParser

prefix_data = '/global/cscratch1/sd/damonge/HSC/HSC_processed'
# Define options
parser = OptionParser()

parser.add_option('--input-prefix', dest='prefix_in', default=prefix_data, type=str,
                 help='Input prefix')
parser.add_option('--output-prefix', dest='prefix_out', default='', type=str,
                 help='Output prefix')
parser.add_option('--nsys-bins', dest='nbins_sys', default=7, type=int,
                 help='Number of bins for the systematic analyses')
parser.add_option('--map-path', dest='fname_maps', type=str,
                 help='Path to maps to analyze')
parser.add_option('--depth-cut', dest='depth_cut', default=24.5, type=float,
                 help='Depth cut')
parser.add_option('--mask-threshold', dest='mask_thr', default=0.5, type=float,
                 help='Minimum area fraction of a given pixel to be considered in the analysis')
o, args = parser.parse_args()

def stats_on_sysmap(sys_map, mask, data_map, nbins, bintype='equal', perc0=0,njk=50) :
    """ Auxiliary routine that reads a galaxy density map and an observing condition
    density map and computes mean and standard deviation of the galaxy overdensity map
    as a function of the observing condition overdensity map 
    
    Args:
    -----
    
    sys_map: (flatmap) density map of the observing condition that we want to analyze.
    mask: (flatmap) Survey/region mask.
    data_map: (flatmap) Galaxy density map.
    nbins: (int) Number of bins in which to analyze sys_map
    bintype: (str) Binning type: `percentiles` uses the percentiles of sys_map, `equal` makes equal spaced
    bins in sys_map (default `equal`).
    perc0: (float) In case of using the percentiles as binning scheme, starting percentile value (default=0).
    njk: (int) Number of jackknife samples to use to compute errors.

    Returns:
    --------
   
    bin_centers_r: (float) Centers of the bins where we analyze sys_map rescaled by the mean
    bin_centers: (float) Centers of the bins where we analyze sys_map in the original units
    mean: (float) Mean value of data_map at bin_centers.
    err: (float) Uncertainty on the mean value of data_map at bin_centers.
    """
    if bintype not in ['percentiles', 'equal', 'log']:
        raise ValueError('Only `percentiles`, `equal` or `log` bintypes allowed.')
    mean = np.zeros(nbins)
    means=np.zeros([njk,nbins])
    bin_centers = np.zeros(nbins)
    binary_mask = mask > 0

    #Divide by mean
    data_use=data_map[binary_mask]*np.sum(mask[binary_mask])/(mask[binary_mask]*np.sum(data_map[binary_mask]))
    sys_mean=np.mean(sys_map[binary_mask])
    sys_use=sys_map[binary_mask]/sys_mean
    if bintype=='percentiles':
        percentile_edges=np.percentile(sys_use,perc0+(100.-perc0)*(np.arange(nbins+1)+0.)/nbins)
        for i in range(nbins):
            sys_mask = (sys_use < percentile_edges[i+1]) & (sys_use >= percentile_edges[i])
            bin_centers[i] = np.mean(sys_use[sys_mask])
            mean[i] = np.mean(data_use[sys_mask])
            for j in range(njk) :
                djk=len(sys_use)//njk
                jk_mask=np.ones(len(sys_use),dtype=bool); jk_mask[j*djk:(j+1)*djk]=False
                means[j,i] = np.mean(data_use[sys_mask*jk_mask])
                
    else :
        nbins=nbins
        mean, bin_edges, _  = binned_statistic(sys_use, data_use, statistic='mean', bins=nbins)
        bin_centers = 0.5*bin_edges[1:]+0.5*bin_edges[:-1]
        for j in range(njk) :
            djk=len(sys_use)//njk
            jk_mask=np.ones(len(sys_use),dtype=bool); jk_mask[j*djk:(j+1)*djk]=False
            means[j,:],_,_=binned_statistic(sys_use[jk_mask],data_use[jk_mask],statistic='mean',bins=bin_edges)
    err = np.std(means,axis=0)*np.sqrt(njk-1.)

    return bin_centers, bin_centers*sys_mean, mean, err
     
def check_sys(data_hdu, path_sys, mask, nbins, **kwargs):
    """ Routine to check the evolution of the mean
    density as a function of different potential
    sources of systematic biases/uncertanty on 
    galaxy clustering
  
    Args:
    -----
    
    data_hdu: (HDU) HDU containing the data that we want to analyze.

    path_sys: (str) Path to the flatmap of the contaminant(s).

    mask: (flatmap) Mask of the region/survey to use.

    **kwargs: (dict) arguments to pass to `stats_on_sysmap`

    Outputs:
    --------

    xsys_r: Values of the potential source of systematics in each bin rescaled by its mean
    xsys: Values of the potential source of systematics in each bin in the original units
    mean: Mean galaxy density in each bin
    err: Error on the mean density in each bin
    """

    fmi, s_map = fm.read_flat_map(path_sys, i_map=-1)
    fmd, data_map = fm.read_flat_map(None, hdu=data_hdu) 
    mean = []
    err = []
    bin_centers = []
    bin_centers_resc = []
    for sys_map in s_map:
        aux_centers, aux_centers_resc, aux_mean, aux_err = stats_on_sysmap(sys_map, mask, data_map, nbins, **kwargs) 
        mean.append(aux_mean)
        bin_centers.append(aux_centers)
        bin_centers_resc.append(aux_centers_resc)
        err.append(aux_err)
    return np.array(bin_centers), np.array(bin_centers_resc), np.array(mean), np.array(err)

# Set up
band = ['g','r','i','z','y']
cont_maps = ['oc_airmass','oc_ccdtemp','oc_ellipt','oc_exptime','oc_nvisit', \
    'oc_seeing', 'oc_sigma_sky', 'oc_skylevel','syst_dust','syst_nstar_i24.50']
xlabels= ['Airmass', r'CCD Temperature [$^{\circ}$C]', 'PSF Ellipticity', \
    'Exposure Time [s]', 'Number of visits', 'Seeing [pixels]', 'Sky noise [ADU]', \
    'Sky level [ADU]', 'Extinction', 'Stars per pixel'] 
data_hdus = fits.open(o.fname_maps)
if len(data_hdus)%2!=0:
    raise ValueError("Input file should have two HDUs per map")
nbins = len(data_hdus)//2

os.system('mkdir -p '+o.prefix_out)

print("Reading mask")
#Create depth-based mask
fsk,mp_depth=fm.read_flat_map(o.prefix_in+"_10s_depth_mean_fluxerr.fits",2)
mp_depth[np.isnan(mp_depth)]=0; mp_depth[mp_depth>40]=0
msk_depth=np.zeros_like(mp_depth); msk_depth[mp_depth>=o.depth_cut]=1

#Read masked fraction
fskb,mskfrac=fm.read_flat_map(o.prefix_in+'_MaskedFraction.fits',i_map=0)
fm.compare_infos(fsk,fskb)

#Create BO-based mask
msk_bo=np.zeros_like(mskfrac); msk_bo[mskfrac>o.mask_thr]=1
msk_t=msk_bo*msk_depth*mskfrac

for ibin in range(nbins):
    print("Bin %d"%ibin)
    data_hdu = data_hdus[2*ibin]
    for j, cm in enumerate(cont_maps):
        print(" "+cm)
        path_sys = o.prefix_in+'_%s.fits' %(cm)
        xsys, xsys_resc, mean_sys, std_sys = check_sys(data_hdu, path_sys, msk_t, o.nbins_sys) 
        if len(xsys)>1:
            f,ax=plt.subplots(5,1,figsize=(5,20))
            for i in range(len(xsys)):
                ax[i].errorbar(xsys[i], mean_sys[i], std_sys[i], fmt='o', label='%s-band' %band[i], fillstyle='none')
                ax[i].set_ylabel(r'$n/\bar{n}$', fontsize=16)
                ax[i].text(0.9,0.9,band[i],transform=ax[i].transAxes)
            ax[-1].set_xlabel(xlabels[j], fontsize=16)
        else:
            f=plt.figure()
            plt.errorbar(xsys[0], mean_sys[0], std_sys[0], fmt='o', label=xlabels[j], fillstyle='none')
            plt.ylabel(r'$n/\bar{n}$', fontsize=16)
            plt.xlabel(xlabels[j], fontsize=16) 
            
        f.tight_layout()
        sname = cm+'_bin_%d' % ibin
        prefix_save=os.path.join(o.prefix_out,sname)
        f.savefig(prefix_save+".pdf")
        plt.close(f) 
        np.savez(prefix_save,x=xsys_resc,x_rescaled=xsys,mean=mean_sys,error=std_sys)
