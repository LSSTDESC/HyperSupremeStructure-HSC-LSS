import treecorr
import numpy as np
import time

def runTreeCorr(data_ra, data_dec, random_ra, random_dec, # all in degrees
                minSep, maxSep, nBins): 
    #print 'Running with %s data pts, %s random pts'%(len(data_ra), len(random_ra))
    startTime= time.time()
    dataCatalog= treecorr.Catalog(ra= data_ra, dec=  data_dec,
                                  ra_units='degrees', dec_units='degrees')
    randCatalog = treecorr.Catalog(ra= random_ra, dec= random_dec,
                                   ra_units='degrees', dec_units='degrees')
    DD = treecorr.NNCorrelation(min_sep= minSep, max_sep= maxSep, nbins= nBins, sep_units='degrees')
    RR= treecorr.NNCorrelation(min_sep= minSep, max_sep= maxSep, nbins= nBins, sep_units='degrees')
    DR = treecorr.NNCorrelation(min_sep= minSep, max_sep= maxSep, nbins= nBins, sep_units='degrees')

    # use LS
    DD.process(dataCatalog)
    RR.process(randCatalog)
    DR.process(dataCatalog, randCatalog)
    # calculate the correlation function
    wtheta, varxi = DD.calculateXi(RR, DR)
    theta = np.exp(DD.logr) # use bin centers
    wtheta_sig = np.sqrt(varxi)
    
    #print 'Time taken: %s (s)'%(time.time()-startTime)

    return theta, wtheta, wtheta_sig

def runTreeCorr_pix(data_ra,data_dec,w_data,rnd_ra,rnd_dec,w_rnd, minSep, maxSep, nBins):
    startTime=time.time()
    dataCatalog = treecorr.Catalog(ra= data_ra, dec= data_dec,
                                   ra_units='degrees', dec_units='degrees', w= w_data)
    rndCatalog = treecorr.Catalog(ra= rnd_ra, dec= rnd_dec, 
                                   ra_units='degrees', dec_units='degrees', w= w_rnd)
    DD = treecorr.NNCorrelation(min_sep= minSep, max_sep= maxSep, nbins= nBins, sep_units='degrees')
    RR = treecorr.NNCorrelation(min_sep= minSep, max_sep= maxSep, nbins= nBins, sep_units='degrees')
    DR = treecorr.NNCorrelation(min_sep= minSep, max_sep= maxSep, nbins= nBins, sep_units='degrees')

    DD.process(dataCatalog)
    RR.process(rndCatalog)
    DR.process(dataCatalog, rndCatalog)
    wtheta, varxi = DD.calculateXi(RR, DR)
    theta = np.exp(DD.logr)
    wtheta_sig = np.sqrt(varxi)
    endTime=time.time()
    return theta, wtheta, wtheta_sig

def runXcorr(data1_ra,data1_dec,data1_w,rnd1_ra,rnd1_dec,rnd1_w,data2_ra,data2_dec,data2_w,rnd2_ra,
    rnd2_dec,rnd2_w,minSep,maxSep,nBins):
    startTime=time.time()
    d1 = treecorr.Catalog(ra= data1_ra, dec=data1_dec, ra_units='degrees', dec_units='degrees', w=data1_w)
    d2 = treecorr.Catalog(ra= data2_ra, dec=data2_dec, ra_units='degrees', dec_units='degrees', w=data2_w)
    r1 = treecorr.Catalog(ra= rnd1_ra, dec=rnd1_dec, ra_units='degrees', dec_units='degrees', w=rnd1_w)
    r2 = treecorr.Catalog(ra= rnd2_ra, dec=rnd2_dec, ra_units='degrees', dec_units='degrees', w=rnd2_w)
    d1d2 = treecorr.NNCorrelation(min_sep=minSep,max_sep=maxSep,nbins=nBins,sep_units='degrees')
    r1r1 = treecorr.NNCorrelation(min_sep=minSep,max_sep=maxSep,nbins=nBins,sep_units='degrees')
    d1r1 = treecorr.NNCorrelation(min_sep=minSep,max_sep=maxSep,nbins=nBins,sep_units='degrees')
    d2r2 = treecorr.NNCorrelation(min_sep=minSep,max_sep=maxSep,nbins=nBins,sep_units='degrees')
    d1d2.process(d1,d2)
    d1r1.process(d1,r1)
    d2r2.process(d2,r2)
    r1r1.process(r1,r1)
    wtheta, varxi = d1d2.calculateXi(r1r1,d1r1,d2r2)
    wtheta_sig = np.sqrt(varxi)
    theta = np.exp(d1d2.logr)
    endTime = time.time()
    return theta, wtheta, wtheta_sig

def runXcorr_pix_scalar(data1_ra,data1_dec,data2_ra,data2_dec,scalar_field,minSep,maxSep,nBins):
    d1 = treecorr.Catalog(ra= data1_ra, dec=data1_dec, ra_units='degrees', dec_units='degrees')
    d2 = treecorr.Catalog(ra= data2_ra, dec=data2_dec, k=scalar_field, ra_units='degrees', dec_units='degrees')
    d1d2 = treecorr.NKCorrelation(min_sep=minSep,max_sep=maxSep,nbins=nBins,sep_units='degrees')
    d1d2.process(d1,d2)
    return np.exp(d1d2.logr), d1d2.xi
def runXcorr_scalar_scalar(data1_ra,data1_dec,data2_ra,data2_dec,sc1,sc2,minSep,maxSep,nBins):
    d1 = treecorr.Catalog(ra= data1_ra, dec=data2_dec, ra_units='degrees', dec_units='degrees', k=sc1)
    d2 = treecorr.Catalog(ra= data2_ra, dec=data2_dec, ra_units='degrees', dec_units='degrees', k=sc2)
    d1d2 = treecorr.KKCorrelation(min_sep=minSep,max_sep=maxSep,nbins=nBins,sep_units='degrees')
    d1d2.process(d1,d2)
    return np.exp(d1d2.logr), d1d2.xi

