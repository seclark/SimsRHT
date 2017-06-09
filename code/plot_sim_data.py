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
    """
    plot density and RHT backprojection for a single beta at three timestamps
    """
    
    fig, xarr = plt.subplots(2, 3, figsize=(10,7))
    
    alltimes = ["0010", "0030", "0050"]
    beta = "20"
    
    for i, tstamp in enumerate(alltimes):
        # get relevant simulation and RHT data
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
        
        # plot density and backprojection fields
        xarr[0, i].pcolor(Y, X, np.log10(dens_im))
        xarr[1, i].pcolor(Y, X, rht_im)
        
        # overplot pseudovectors from Bsim and theta_RHT
        overplot_vectors(Qsim, Usim, ax=xarr[0, i], norm=True, intrht=False, skipint=25)
        QRHT, URHT, URHTsq, QRHTsq, intrht = RHT_tools.grid_QU_RHT(fn_rht, save=False)
        overplot_vectors(QRHT, URHT, ax=xarr[1, i], norm=True, intrht=False, skipint=25)
        
        # title from timestamp
        xarr[0, i].set_title(r"$t = {}$".format(tstamp))
        
    plt.suptitle(r"z, 512, $\beta$ = {}".format(beta))
    xarr[0, 0].set_ylabel("log(density), $B_{sim}$ $\mathrm{overlaid}$")
    xarr[1, 0].set_ylabel(r"$\int R(\theta)$, $\theta_{RHT}$ $\mathrm{overlaid}$")
    

def overplot_vectors(Q, U, ax=None, norm=True, intrht=False, skipint=10, color="white"): 
    """
    overlay a quiver plot of pseudovectors
    inputs: Q, U   :: Stokes fields
            ax     :: plot axis
            norm   :: True   : make all pseudovectors the same length
            intrht :: False  : if True, only plot data over given backprojection amplitude threshold
            skipint:: 10     : plot a pseudovector every skipint-th pixel in x and y
            color  :: "white": color for pseudovectors
    """  
    
    # make all pseudovectors same length
    if norm:
        normc = np.sqrt(Q**2 + U**2)
        Q = Q/normc
        U = U/normc

    # quiver takes x- and y- components of the angle
    thetamap = np.mod(0.5*np.arctan2(U, Q), np.pi)
    Q = np.sin(thetamap)
    U = np.cos(thetamap)
 
    ysize, xsize = thetamap.shape
    X, Y = np.meshgrid(np.arange(xsize), np.arange(ysize), indexing='ij')
    
    if intrht is False:
        ax.quiver(Y[::skipint, ::skipint], X[::skipint, ::skipint], Q[::skipint, ::skipint], -U[::skipint, ::skipint], 
               headaxislength=0, headlength=0, pivot="mid", color=color)
    # only display RHT data where RHT backprojection power > 1
    else:
        intcut = np.where(intrht > 1)
        cutY = Y[intcut]
        cutX = X[intcut]
        cutQ = Q[intcut]
        cutU = U[intcut]
        cutthetamap = thetamap[intcut]
        ax.quiver(cutY[::skipint], cutX[::skipint], cutQ[::skipint], -cutU[::skipint], 
               headaxislength=0, headlength=0, pivot="mid", color=color)

