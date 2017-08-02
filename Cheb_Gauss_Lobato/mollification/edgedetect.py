### edgedetect.py
### Short program which contains tools to perform edge detection by applying
### concentration coefficients to spectral expansion of function
### of interest in the Chebyshev First Derivatives basis.
### written by Joanna Piotrowska

from __future__ import print_function

import os, sys
import numpy as np
import scipy as sp
from numpy import pi, sqrt, exp, cos, sin, absolute, log, log10
from numpy.polynomial import chebyshev as T
from scipy.integrate import quad
from matplotlib import rcParams
from spectral_tools import Decompose, EdgeDetectT, EdgeDetectIII, ModalD, NodalD, Lanczos, Vandermonde
import peakutils

############################################

def ChebEdge(a_n, x, modalD):

    N = a_n.shape[0]-1

    prefactor = pi * sqrt(1.0-np.power(x, 2))

    alpha = 1.0
    cf1, cf2, cf3, cf4, cf5, cf6 = EdgeDetectT(N, alpha)
    factors = np.vstack((cf1, cf2, cf3, cf4, cf5, cf6))
    factors = factors.T
    a_mprime = np.dot(modalD, a_n)

    edge = np.empty((len(x), len(factors[0,:])))

    for i in range(len(factors[0,:])):
        
        cf = factors[:,i]
        cfa_n = cf * a_mprime
        edge[:,i] = prefactor * T.chebval(x, cfa_n)

    return edge

#############################################

def ChebEdgeIII(a_n, x, modalD):

    """
    Compute jump functions of the discontinuous function under
    analysis using filtering methods described in A.Gelb & D.Cates
    Detection of Edges in Spectral Data III

    Parameters
    ----------
    a_n : 1D array, shape = (N+1,)
          N - highest order in the Chebyshev expansion
          vector of Chebyshev expansion coefficients

    x : 1D array,
        1D grid of points in real space, where the function is evaluated

    modalD : 2D array, shape = (N+1, N+1)
             Modal Differentiation Matrix

    Returns
    ----------

    edges : 2D array, shape = (len(x), 16)
            jump function values evaluated on the real space grid for 
            different filters in each column

    Notes
    ----------
    Can be generalised to include matching waveform and zero-crossing techniques
    once they are calculated for the Chebyshev polynomial expansion

    """

    N = a_n.shape[0]-1
    k = np.arange(N) + 1
    prefactor = pi * sqrt(1.0-np.power(x, 2))
    a_nprime = np.dot(modalD, a_n)
    sigmas = np.zeros((a_n.shape[0], 16))
    edges = np.empty((len(x), len(sigmas[0,:])))
    
    sigmas[:,0:4], trig_fun, poly_fun, expon_fun = EdgeDetectIII(a_n)
    
    for n in range(3):
        sigmas[:,(n+1)*4:(n+1)*4+4] = Lanczos(sigmas[:,0:4], trig_fun, poly_fun, expon_fun, (n+1))
    
    sigmas_t = sigmas.T
    sigmas_t[:,1:] = np.divide(sigmas_t[:,1:], k)
    sigmas = sigmas_t.T
    
    for i in range(len(sigmas[0,:])):
        
        cf = sigmas[:,i]
        cfa_n = cf * a_nprime
        edges[:,i] = -prefactor * T.chebval(x, cfa_n)

    return edges

############################################

def ConfigurePlots():

    """ Setting global Matplotlib settings """

    rcParams['lines.linewidth'] = 1
    rcParams['lines.color'] = 'green'
    rcParams['lines.markersize'] = 3
    rcParams['mathtext.rm'] = 'serif'
    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 12
    rcParams['axes.grid'] = 'True'
    rcParams['axes.titlesize'] = 15
    rcParams['axes.labelsize'] = 15
    rcParams['axes.ymargin'] = 0.5
    rcParams['figure.figsize'] = 7, 7
    rcParams['legend.fontsize'] = 'small'
    rcParams['savefig.format'] = 'png'

#############################################

def EdgeDetectErr(N, pos, width, x):
    
    c2s, s2c = Vandermonde(N)
    modalD = ModalD(c2s)
    err_e = np.empty((len(pos), len(width)))
    err_me = np.empty((len(pos), len(width)))
    err_eme = np.empty((len(pos), len(width)))
    
    for idxl, loc in enumerate(pos):
        for idxw, w in enumerate(width):
            steps = np.array([loc, loc+w])
            func = lambda x: 1.0 if (x>=loc and x<=(loc+w)) else 0

            a_n = Decompose(func, c2s)
            edge = ChebEdge(a_n, x, modalD)
    
            enhanced_edge = Enhance(a_n, x, edge[:,-1], 4)
            idx_e = peakutils.indexes(absolute(enhanced_edge))
            peaks_e = peakutils.interpolate(x, enhanced_edge, ind=idx_e)
    
            minmod_edge = MinMod(edge)
            idx_me = peakutils.indexes(absolute(minmod_edge))
            peaks_me = peakutils.interpolate(x, minmod_edge, ind=idx_me)
    
            enhanced_minmod = Enhance(a_n, x, minmod_edge, 4)
            idx_eme = peakutils.indexes(absolute(enhanced_minmod))
            peaks_eme = peakutils.interpolate(x, enhanced_minmod, ind=idx_eme)
    
            if peaks_e.shape != steps.shape:
                err_e[idxl, idxw] = np.nan
            else:
                err_e[idxl, idxw] = np.average(absolute(steps-peaks_e))

            if peaks_me.shape != steps.shape:
                err_me[idxl, idxw] = np.nan
            else:
                err_me[idxl, idxw] = np.average(absolute(steps-peaks_me))

            if peaks_eme.shape != steps.shape:
                err_eme[idxl, idxw] = np.nan

            else:
                err_eme[idxl, idxw] = np.average(absolute(steps-peaks_eme))
            
    return err_e, err_me, err_eme

############################################

def Enhance(a_n, x, edge, p):

    """
    Perform non-linear enhancement on the jump function to take advantage 
    of separation of scales in peak detection afterwards.

    Parameters
    ----------

    a_n : 1D array, shape = (N+1,)
          N - highest order in the Chebyshev expansion
          vector of Chebyshev expansion coefficients
    
    x : 1D array,
        1D grid of points in real space, where the function is evaluated
    
    edge : 1D array, shape = (len(x), )
           jump function values evaluated on the real space grid 

    p : int,
        power, which controls the separation of scales in enhancement

    Returns
    ----------

    enhanced_edge : 1D array, shape = (len(x), )
                    enhaned jump function values evaluated on the real space grid

    """

    N = a_n.shape[0]-1
    S_N = T.chebval(x, a_n)
    
    eps = log(N)/N
    J = 0.9*(np.amax(S_N)-np.amin(S_N))*(N)

    K_eps = (eps**(-p/2))*(absolute(edge)**p)
    edge = np.ma.masked_where(K_eps <= J, edge)
    enhanced_edge = edge.filled(0)
    
    return enhanced_edge

#############################################

def MinMod(edges):

    """
    Perform minmod operation on a set of jump function approximations
    evaluated on the real space grid. Requires a 2D array input.

    Parameters
    ----------

    edges : 2D array, 
            jump function values evaluated on the real space grid for 
            different filters in each column

    Returns
    ---------

    edge_minmod : 1D array,
                  jump function after performing edge detection on the
                  real space grid
    
    """
    
    edge_minmod = np.empty(len(edges[:,0]))
    n_e = len(edges[0,:])

    for i in range(len(edges[:,0])):
        neg = np.ma.masked_where(edges[i,:] < 0.0, edges[i,:], copy=True)
        
        if np.ma.count_masked(neg) == n_e:
            edge_minmod[i] = np.amax(edges[i,:])
        
        elif np.ma.count(neg) == n_e:
            edge_minmod[i] = np.amin(edges[i,:])

        else:
            edge_minmod[i] = 0.0

    return edge_minmod

#############################################

def LocateEdges(a_n, x, modalD):

    """
    Locate positions of discontinuities in the data, using its Chebyshev
    spectral expansion coefficients

    Parameters
    ----------
 
    a_n : 1D array, shape = (N+1,)
          N - highest order in the Chebyshev expansion
          vector of Chebyshev expansion coefficients

    x : 1D array,
        1D grid of points in real space, where the function is evaluated
        
    modalD : 2D array, shape = (N+1, N+1)
             Modal Differentiation Matrix

    Returns
    ----------

    c_j : 1D array,
          Array containing x positions of the discontinuities in the data

    """

    edges = ChebEdgeIII(a_n, x, modalD)
    minmod_edge = MinMod(edges)
    enhanced_minmod = Enhance(a_n, x, minmod_edge, 3)
    idx = peakutils.indexes(absolute(enhanced_minmod))
    
    c_j = peakutils.interpolate(x, absolute(enhanced_minmod), ind=idx)
    
    return c_j

#############################################
    
