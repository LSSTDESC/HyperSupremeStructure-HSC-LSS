import numpy as np
import astropy.io.fits as fits
from twoPtCorr import *
import flatmaps as fm
import intermediates
from createMaps import *
# We can also move to fitsio but to be consistent with the rest of the programs I'll stick to astropy.io.fits
from optparse import OptionParser
import astropy.table
parser = OptionParser()
parser.add_option('--input-dir','-i',dest='predir',help='Path to processed files',default="/global/cscratch1/sd/damonge/HSC/HSC_processed/")
parser.add_option('--mask-threshold',dest='mask_thr',help='Threshold in completeness to consider the pixel in the analysis',default=0.5)
parser.add_option('--snr',dest='snr',help='SNR threshold',default=10)
parser.add_option('--mag-lim',dest='mlim',help='Minimum limiting magnitude to select the pixel',default=24.5)
parser.add_option('--band',dest='band',help='Selected band',default='i')

(o, args) = parser.parse_args()
mask_thr=o.mask_thr
snr=o.snr
mlim=o.mlim
band=o.band
fields=['WIDE_GAMA09H','WIDE_GAMA15H','WIDE_HECTOMAP','WIDE_VVDS',
        'WIDE_WIDE12H','WIDE_AEGIS','WIDE_XMMLSS']
theta, wtheta, wtheta_sig = {}, {}, {}
w_star, w_dust_gal, w_star_gal, w_star_dust, w_dust_dust = {}, {}, {}, {}, {}
npix={}
for field in fields:
    #Create depth-based mask
    fsk,mp_depth=fm.read_flat_map(predir+field+'/'+field+"_%ds_depth_mean_fluxerr.npz"%snr,2)
    mp_depth[np.isnan(mp_depth)]=0; mp_depth[mp_depth>40]=0
    msk_depth=np.zeros_like(mp_depth); msk_depth[mp_depth>=mlim]=1

    #Read masked fraction
    fskb,mskfrac=fm.read_flat_map(predir+field+'/'+field+"_MaskedFraction.npz")
    if fsk.get_dims()!=fskb.get_dims() :
        raise ValueError("Inconsistent fskys")

    #Create BO-based mask
    msk_bo=np.zeros_like(mskfrac); msk_bo[mskfrac>mask_thr]=1
    msk_t=msk_bo*msk_depth
    goodpix=np.where(msk_t>0.1)[0]

    #Total weight map
    weight=mskfrac*msk_t
    npix[field]=np.sum(weight)

    #Read systematics maps
    fskb,mp_dust=fm.read_flat_map(predir+field+'/'+field+"_syst_dust.npz",2)
    if fsk.get_dims()!=fskb.get_dims() :
        raise ValueError("Inconsistent fskys")
    fskb,mp_star=fm.read_flat_map(predir+field+'/'+field+"_syst_nstar_i%.2lf.npz"%mlim)
    if fsk.get_dims()!=fskb.get_dims() :
           raise ValueError("Inconsistent fskys")
    fsk.view_map(mp_star)
    #Read catalog and cut based on BO mask
    cat=fits.open(predir+field+"/"+field+'_Catalog_i%.2lf.fits'%mlim)[1].data
    mskob=~(cat['iflags_pixel_bright_object_center']*cat['iflags_pixel_bright_object_any'])
    cat=cat[mskob]
    plt.figure()
    plt.hist2d(cat['ra_r'],cat['dec_r'],bins=50)
    #Mask objects out of the footprint
    pixnums = fsk.pos2pix(cat['ra_r'],cat['dec_r'])
    ind= np.where((cat['%scmodel_mag'%band]<mlim) & (cat['iclassification_extendedness']>0.5) & (np.in1d(pixnums,goodpix)))[0]
    galSample_ra, galSample_dec= cat['ra_r'][ind], cat['dec_r'][ind]
    mask_tot = np.zeros(msk_t.shape,dtype=bool)
    mask_tot[goodpix]=True
    #Generate random catalog
    randCatalog= intermediates.getRandomCatalog(fskb, mask_tot,
                                                        minRA= min(cat['ra_r']),
                                                        maxRA= max(cat['ra_r']),
                                                        minDec=  min(cat['dec_r']),
                                                        maxDec=  max(cat['dec_r']),
                                                        nData= 8*len(cat['ra_r']),
                                                        plotMap= False)
    
    #Galaxy autocorrelation
    theta[field], wtheta[field], wtheta_sig[field] = runTreeCorr(data_ra= galSample_ra,
                                                                     data_dec= galSample_dec,
                                                                     random_ra= randCatalog['ra'],
                                                                     random_dec= randCatalog['dec'],
                                                                     minSep= 0.001, maxSep= 5, nBins= 25)
    #Stellar map autocorrelation
    pix_ra, pix_dec = fskb.pix2pos(goodpix)
    randStar = intermediates.getRandomCatalog(fskb, mask_tot, minRA= min(cat['ra_r']), maxRA=max(cat['ra_r']),
                                            minDec = min(cat['dec_r']), maxDec = max(cat['dec_r']),
                                              nData = int(8*np.sum(mp_star[goodpix])), plotMap = False)
    
    mp_rnd_star = createCountsMap(randStar['ra'], randStar['dec'], fskb)
    mp_gal = createCountsMap(galSample_ra,galSample_dec, fskb)
    mp_rnd = createCountsMap(randCatalog['ra'],randCatalog['dec'], fskb)
    _, w_star[field], _ = runTreeCorr_pix(data_ra=pix_ra,data_dec=pix_dec,w_data=mp_star,rnd_ra=pix_ra,rnd_dec=pix_dec,w_rnd=mp_rnd_star,minSep=0.001,maxSep=5,nBins=25)
    #Cross-correlations
    _ , w_dust_gal[field], _ = runXcorr(pix_ra,pix_dec,mp_gal[goodpix],pix_ra,pix_dec,mp_rnd[goodpix],pix_ra,pix_dec,mp_dust[goodpix],pix_ra,pix_dec,mp_rnd[goodpix],minSep=0.001,maxSep=5,nBins=25)
    _ , w_star_gal[field], _ = runXcorr(pix_ra,pix_dec,mp_gal[goodpix],pix_ra,pix_dec,mp_rnd[goodpix],pix_ra,pix_dec,mp_star[goodpix],pix_ra,pix_dec,mp_rnd_star[goodpix],minSep=0.001,maxSep=5,nBins=25)
    _ , w_star_dust[field], _ = runXcorr(pix_ra,pix_dec,mp_star[goodpix],pix_ra,pix_dec,mp_rnd_star[goodpix],pix_ra,pix_dec,mp_dust[goodpix],pix_ra,pix_dec,mp_rnd_star[goodpix],minSep=0.001,maxSep=5,nBins=25)
    _ , w_dust_dust[field], _ = runXcorr(pix_ra,pix_dec,mp_dust[goodpix],pix_ra,pix_dec,mp_rnd[goodpix],pix_ra,pix_dec,mp_dust[goodpix],pix_ra,pix_dec,mp_rnd[goodpix],minSep=0.001,maxSep=5,nBins=25)
    tab = astropy.table.Table([theta[field],wtheta[field],w_star[field],w_dust_dust[field],w_star_gal[field],w_dust_gal[field],w_star_dust[field]],names=('theta','wgg','wss','wdd','wsg','wdg','wsd'))
    fileout='w_results_'+field+'.fits.gz'
    tab.write(fileout,overwrite=True)
