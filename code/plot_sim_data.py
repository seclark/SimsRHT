from __future__ import division, print_function
import numpy as np
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pyfits
import healpy as hp
import pylab
from astropy.io import fits

import sys
sys.path.insert(0, '/Users/susanclark/RHT')
import RHT_tools

def get_data_fn(projectax, nres, beta, tstamp, rhtparams=None, type="density"):
    """
    get filename from simulation parameters.
    rhtparams : tuple of [wlen, smr, thresh] if rht data
              : None if simulation snapshot
    """
    fnroot = "/Volumes/DataDavy/CollinsSims/cdb09f/Q_U_Sigma_Maps/self_gravitating/"
    fntot = type+"_"+str(projectax)+"_"+str(nres)+"_B"+str(beta)+"_"+str(tstamp)

    if rhtparams is not None:
        wlen, smr, thresh = rhtparams
        fntot = fntot+"_xyt_w"+str(wlen)+"_s"+str(smr)+"_t"+str(thresh)
        
    return fnroot+fntot+".fits"
    
    
def plot_sims_and_rht():
    
    fig, xarr = plt.subplots(2, 3)
    
    allbetas = ["02", "2", "20"]
    alltimes = ["0010", "0030", "0050"]
    #tstamp = "0050"
    beta = "20"
    
    for i, tstamp in enumerate(alltimes):
        fn_rht = get_data_fn("z", 512, beta, tstamp, rhtparams=[25, 1, 0.7], type="density")
        fn_dens = get_data_fn("z", 512, beta, tstamp, rhtparams=None, type="density")
        fn_Qsim = get_data_fn("z", 512, beta, tstamp, rhtparams=None, type="Q")
        fn_Usim = get_data_fn("z", 512, beta, tstamp, rhtparams=None, type="U")
        
        rht_im = fits.getdata(fn_rht)
        dens_im = fits.getdata(fn_dens)
        Qsim = fits.getdata(fn_Qsim)
        Usim = fits.getdata(fn_Usim)

        ysize, xsize = rht_im.shape
        X, Y = np.meshgrid(np.arange(xsize), np.arange(ysize), indexing='ij')
        xarr[0, i].pcolor(Y, X, np.log10(dens_im))
        xarr[1, i].pcolor(Y, X, rht_im)
        
        xarr[0, i].set_title(r"$t = {}$".format(tstamp))
        
        # plot vectors
        overplot_vectors(Qsim, Usim, ax=xarr[0, i], norm=True, intrht=False, skipint=25)
        #overplot_vectors(Qsim, Usim, ax=xarr[1, i], norm=True, intrht=False, skipint=25, color="yellow")
        QRHT, URHT, URHTsq, QRHTsq, intrht = RHT_tools.grid_QU_RHT(fn_rht, save=False)
        overplot_vectors(QRHT, URHT, ax=xarr[1, i], norm=True, intrht=False, skipint=25)
        
    plt.suptitle(r"z, 512, $\beta$ = {}".format(beta))
    

def overplot_vectors(Q, U, ax=None, norm=True, intrht=False, skipint=10, color="white"): 
    """
    overlay a quiver plot of pseudovectors
    """  

    if norm:
        normc = np.sqrt(Q**2 + U**2)
        Q = Q/normc
        U = U/normc
        thetamap = np.mod(0.5*np.arctan2(U, Q), np.pi)
        Q = np.sin(thetamap)
        U = np.cos(thetamap)
 
    ysize, xsize = thetamap.shape
    X, Y = np.meshgrid(np.arange(xsize), np.arange(ysize), indexing='ij')
    
    if intrht is False:
        ax.quiver(Y[::skipint, ::skipint], X[::skipint, ::skipint], Q[::skipint, ::skipint], -U[::skipint, ::skipint], 
               headaxislength=0, headlength=0, pivot="mid", color=color)

    else:
        intcut = np.where(intrht > 1)
        cutY = Y[intcut]
        cutX = X[intcut]
        cutQ = Q[intcut]
        cutU = U[intcut]
        cutthetamap = thetamap[intcut]
        ax.quiver(cutY[::skipint], cutX[::skipint], cutQ[::skipint], -cutU[::skipint], 
               headaxislength=0, headlength=0, pivot="mid", color=color)

