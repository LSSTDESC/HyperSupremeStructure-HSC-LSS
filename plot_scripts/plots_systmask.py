import flatmaps as fm
import numpy as np
import matplotlib.pyplot as plt

prefix="/global/cscratch1/sd/damonge/HSC_ceci/WIDE_"
field="VVDS"
fsk,msk1=fm.read_flat_map(prefix+field+
                          "_sirius_i24p5_out/masked_fraction.fits")
msk1[msk1<0.5]=0; msk1[msk1>=0.5]=1.;
_,msk2=fm.read_flat_map(prefix+field+
                        "_sirius_i24p5_out/"+
                        "CovAna_NoiAna_MskSiriusSyst_ClFit_Dpj0_DpjBands1/mask_syst.fits")

fig=plt.figure()
ax=fig.add_subplot(111,projection=fsk.wcs)
ax.set_title(field,fontsize=14)
im=ax.imshow((msk1+msk2).reshape([fsk.ny,fsk.nx]),vmin=0,vmax=2,
             origin='lower', interpolation='nearest')
ax.set_xlabel('R.A.', fontsize=14)
ax.set_ylabel('Dec.', fontsize=14)
plt.savefig("../doc/Paper/figures/systmask.pdf",
            bbox_inches='tight')
plt.show()
