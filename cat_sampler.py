from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from createMaps import createCountsMap
from optparse import OptionParser
import flatmaps as fm
import sys
import time
import os

prefix_data='/global/cscratch1/sd/damonge/HSC/'
def opt_callback(option, opt, value, parser):
  setattr(parser.values, option.dest, value.split(','))

parser = OptionParser()
#Options
parser.add_option('--input-prefix', dest='prefix_in', default='NONE', type=str,
                  help='Input prefix. The input catalog will be searched for as input-prefix + _Catalog_<band><limit>.fits.')
parser.add_option('--output-file', dest='fname_out',default=None,type=str,
                  help='Output file name. If None, I\'ll use input-prefix + _bins_ + fname_bins + .fits')
parser.add_option('--no-bo-cut',dest='no_bo_cut',default=False,action='store_true',
                  help='Remove objects within bright-object mask')
parser.add_option('--pz-type',dest='pz_type',default='nnpz',type=str,
                  help='Photo-z to use')
parser.add_option('--pz-mark',dest='pz_mark',default='best',type=str,
                  help='Photo-z summary statistic to use when binning objects')
parser.add_option('--pz-bins',dest='fname_bins',default=None,type=str,
                  help='File containing the redshift bins (format: 1 row per bin, 2 columns: z_ini z_end)')
parser.add_option('--map-sample',dest='map_sample',default=None,type=str,
                  help='Sample map used to determine the pixelization that will be used. If None I\'ll try to find the masked fraction map')
parser.add_option('--analysis-band', dest='band', default='i', type=str,
                  help='Band considered for your analysis (g,r,i,z,y)')
parser.add_option('--depth-cut', dest='depth_cut', default=24.5, type=float,
                  help='Minimum depth to consider in your footprint')

####
# Read options
(o, args) = parser.parse_args()

fname_cat=o.prefix_in+'_Catalog_'+o.band+'%.2lf.fits'%o.depth_cut
if not os.path.isfile(fname_cat) :
  raise KeyError("File "+fname_cat+" doesn't exist")

if o.map_sample is None :
  o.map_sample=o.prefix_in+'_MaskedFraction.fits'
if not os.path.isfile(o.map_sample) :
  raise KeyError("File "+o.map_sample+" doesn't exist")

if (o.fname_bins is None) or (not os.path.isfile(o.fname_bins)) :
  raise KeyError("Can't fine bins file")

if o.fname_out is None :
  o.fname_out=o.prefix_in+'_bins_'+o.fname_bins+'.fits'

if o.pz_type=='ephor_ab' :
  pz_code='eab'
elif o.pz_type=='frankenz' :
  pz_code='frz'
elif o.pz_type=='nnpz' :
  pz_code='nnz'
else :
  raise KeyError("Photo-z method "+o.pz_type+" unavailable. Choose ephor_ab, frankenz or nnpz")

if o.pz_mark  not in ['best','mean','mode','mc'] :
  raise KeyError("Photo-z mark "+o.pz_mark+" unavailable. Choose between best, mean, mode and mc")

column_mark='pz_'+o.pz_mark+'_'+pz_code
column_pdfs='pz_mc_'+pz_code

print(column_mark)

#Read catalog
cat=fits.open(fname_cat)[1].data
if not o.no_bo_cut :
  msk=np.logical_not(cat['iflags_pixel_bright_object_center'])
  msk*=np.logical_not(cat['iflags_pixel_bright_object_any'])
  cat=cat[msk]

#Read map information
fsk,mpdum=fm.read_flat_map(o.map_sample,0)

#Read bins
zi_arr,zf_arr=np.loadtxt(o.fname_bins,unpack=True,ndmin=2)
nbins=len(zi_arr)

#Iterate through bins
maps=[]
nzs=[]
for zi,zf in zip(zi_arr,zf_arr) :
  msk=(cat[column_mark]<=zf) & (cat[column_mark]>zi)
  subcat=cat[msk]
  zmcs=subcat[column_pdfs]
  hz,bz=np.histogram(zmcs,bins=50,range=[0.,4.])
  nmap=createCountsMap(subcat['ra'],subcat['dec'],fsk)
  nzs.append([bz[:-1],bz[1:],hz+0.])
  maps.append(nmap)
nzs=np.array(nzs)
maps=np.array(maps)

#Save maps and N(z)s
if len(maps[0])!=fsk.npix :
  raise ValueError("Map doesn't conform to this pixelization")

header=fsk.wcs.to_header()
hdus=[]
for im,m in enumerate(maps) :
  #Map
  head=header.copy()
  head['DESCR']=('Ngal, bin %d'%(im+1),'Description')
  if im==0 :
    hdu=fits.PrimaryHDU(data=m.reshape([fsk.ny,fsk.nx]),header=head)
  else :
    hdu=fits.ImageHDU(data=m.reshape([fsk.ny,fsk.nx]),header=head)
  hdus.append(hdu)

  #Nz
  cols=[fits.Column(name='z_i',array=nzs[im,0,:],format='E'),
        fits.Column(name='z_f',array=nzs[im,1,:],format='E'),
        fits.Column(name='n_z',array=nzs[im,2,:],format='E')]
  hdus.append(fits.BinTableHDU.from_columns(cols))
hdulist=fits.HDUList(hdus)
hdulist.writeto(o.fname_out,overwrite=True)
