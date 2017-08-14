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
import matplotlib.pyplot as plt

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
    sigmas = np.zeros((a_n.shape[0], 12))
    edges = np.empty((len(x), len(sigmas[0,:])))
    
    sigmas[:,0:3], trig_fun, poly_fun, expon_fun = EdgeDetectIII(a_n)
    
    for n in range(3):
        sigmas[:,(n+1)*3:(n+1)*3+3] = Lanczos(sigmas[:,0:3], trig_fun, poly_fun, expon_fun, (n+1))
    
    sigmas_t = sigmas.T
    sigmas_t[:,1:] = np.divide(sigmas_t[:,1:], k)
    sigmas = sigmas_t.T
    
    for i in range(len(sigmas[0,:])):
        
        cf = sigmas[:,i]
        cfa_n = cf * a_nprime
        edges[:,i] = prefactor * T.chebval(x, cfa_n)

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
    J = 0.9*(np.amax(S_N)-np.amin(S_N))*(N**2)

    K_eps = (eps**(-p/2))*(absolute(edge)**p)
    edge = np.ma.masked_where(K_eps <= J, edge)
    enhanced_edge = edge.filled(0)
    
    return enhanced_edge

############################################

def GetApproxZeros(x, y):
    
    zeros = np.array([-1.0])
    
    for i in range(len(x)-1):
        
        if y[i+1]*y[i] < 0:
            zero = (x[i+1]+x[i])/2
            zeros = np.append(zeros, zero)

    zeros = np.append(zeros, 1.0)
    
    return zeros

############################################

def GetClosestZeros(zeros, peaks):
    
    closest_zeros = np.empty((len(peaks), 2))

    for i, peak in enumerate(peaks):
        
        closest_idx = np.argmin(absolute(zeros-peak))
        closest_zero = zeros[closest_idx]
    
        if closest_zero < peak:
            closest_zeros[i,0] = closest_zero
            closest_zeros[i,1] = zeros[closest_idx+1]
    
        else:
            closest_zeros[i,0] = zeros[closest_idx-1]
            closest_zeros[i,1] = closest_zero
        
    if (closest_zeros[i,:] == closest_zeros[i-1,:]).all():
        closest_zeros[i,:] = np.nan

    closest_zeros = np.ma.masked_invalid(closest_zeros)
    closest_zeros = np.ma.compress_rows(closest_zeros)
    
    return closest_zeros

############################################

def GetRealJumps(closest_zeros, minmod_peaks):
    
    real_jumps = np.array([])
    
    for i in range(len(closest_zeros[:,0])):
        for j in range(len(minmod_peaks)):
            if ((minmod_peaks[j] > closest_zeros[i,0]) and (minmod_peaks[j] < closest_zeros[i,1])):
                real_jumps = np.append(real_jumps, minmod_peaks[j])
    return real_jumps

############################################


def MinModEnhance(a_n, x, edge):

    """
    Perform non-linear enhancement on the minmod jump function using information
    about differences in function values at neighbouring collocation points.

    Parameters
    ----------

    a_n : 1D array, shape = (N+1,)
          N - highest order in the Chebyshev expansion
          vector of Chebyshev expansion coefficients
    
    x : 1D array,
        1D grid of points in real space, where the function is evaluated
    
    edge : 1D array, shape = (len(x), )
           jump function values evaluated on the real space grid 

    Returns
    ----------

    enhanced_edge : 1D array, shape = (len(x), )
                    enhaned jump function values evaluated on the real space grid

    """

    N = a_n.shape[0]-1
    x_j = T.Chebyshev.basis(N).deriv().roots()
    x_j = np.insert(x_j, 0, -1.0)
    x_j = np.append(x_j, 1.0)
    
    # compute function variation between neighbouring collocation points
    # and the ratio of (local variation)-to-(edge function value)
    
    I_Nval = T.chebval(x_j, a_n)
    
    #diff_val = np.empty(N)
    #diff_x = np.empty(N)
    #approx_grad_x = np.empty(N)
    diff = np.empty(len(x))
    ratio = np.zeros(len(x))
    
    for i in range(N):
    
        x_piece = np.ma.masked_where(x < x_j[i], x)
    
        if i != N-1:
            x_piece = np.ma.masked_where(x > x_j[i+1], x_piece)
    
        min_idx = np.argmin(x_piece)
        max_idx = np.argmax(x_piece)
        diff[min_idx:max_idx+1] = absolute(I_Nval[i+1]-I_Nval[i])
        #diff_val[i] = I_Nval[i+1]-I_Nval[i]
        #diff_x[i] = np.amax(x_piece) - np.amin(x_piece)
        #approx_grad_x[i] = (np.amax(x_piece) + np.amin(x_piece))/2

    #threshold = 0.3*np.amax(diff)
    #diff[diff < threshold] = 0
    
    relevant_edge = np.ma.masked_where(edge == 0.0, edge)
    ratio = np.ma.divide(absolute(relevant_edge), diff)
    
    #ratio = np.ma.divide(diff, edge)
    ratio_masked = np.ma.masked_where(ratio > 10.0, ratio)
    ratio_masked = np.ma.masked_where(ratio < 1.0, ratio_masked)

    enhanced_edge = np.ma.array(edge, mask=ratio_masked.mask)
    enhanced_edge = np.ma.filled(enhanced_edge, 0)
    
    #approx_grad = np.divide(diff_val, diff_x)
    
    return ratio, ratio_masked, enhanced_edge


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
        pos = np.ma.masked_where(edges[i,:] > 0.0, edges[i,:], copy=True)
        
        if np.ma.count_masked(neg) == n_e:
            edge_minmod[i] = np.amax(edges[i,:])
        
        elif np.ma.count_masked(pos) == n_e:
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
    plt.plot(x, minmod_edge)
    enhanced_minmod = Enhance(a_n, x, minmod_edge, 3)
    plt.plot(x, enhanced_minmod)
    idx = peakutils.indexes(absolute(enhanced_minmod))
    c_j = np.empty(len(idx), dtype='float')
    
    for i, indx in enumerate(idx):
        
        c_j[i] = x[indx]
        
    #c_j = np.array(peakutils.interpolate(x, absolute(enhanced_minmod), ind=idx), dtype='float64')
    c_j = np.ma.masked_where(c_j > 1.0, c_j)
    c_j = np.ma.masked_where(c_j < -1.0, c_j)
    c_j = np.ma.compressed(c_j)
    
    return c_j

#############################################

def LocateEdgesDeriv(a_n, x, modalD):
    
    edges = ChebEdgeIII(a_n, x, modalD)
    minmod_edge = MinMod(edges)
    peak_idx = peakutils.indexes(absolute(minmod_edge), thres=0.15)
    minmod_peaks = np.empty(len(peak_idx))
    minmod_peaksval = np.empty(len(peak_idx))

    for i, idx_i in enumerate(peak_idx):
    
        minmod_peaks[i] = x[idx_i]
        minmod_peaksval[i] = minmod_edge[idx_i]
    
    derivative, peaks, peaks_val = PeaksInDeriv(a_n, x, modalD)
    zeros = GetApproxZeros(x, derivative)
    closest_zeros = GetClosestZeros(zeros, peaks)
    real_jumps = GetRealJumps(closest_zeros, minmod_peaks)
    
    return real_jumps

#############################################
    
def PeaksInDeriv(a_n, x, modalD):
    
    N = a_n.shape[0] - 1
    x_j = np.hstack((-1.0, T.Chebyshev.basis(N).deriv().roots(), 1.0))
    
    a_nderivative = np.dot(modalD, a_n)
    derivative = T.chebval(x, a_nderivative)
    derivative[0] = derivative[-1]
    
    idx = peakutils.indexes(absolute(derivative[1:-1]), thres=0.05)
    peaks = np.empty(len(idx))
    peaks_val = np.empty(len(idx))

    for i, idx_i in enumerate(idx):
    
        peaks[i] = x[idx_i]
        peaks_val[i] = derivative[idx_i]

    edge_cases = np.ma.masked_where( (1.0-absolute(peaks)) > absolute(x_j[0]-x_j[5]), peaks_val)

    if np.ma.count(peaks_val) != 1:
        edge_cases = np.ma.masked_equal(absolute(edge_cases), np.amax(absolute(edge_cases)))
        peaks_val = np.ma.masked_array(peaks_val, mask=np.logical_not(edge_cases.mask))

    peaks = np.ma.masked_array(peaks, mask=peaks_val.mask)
    peaks = np.ma.compressed(peaks)
    peaks_val = np.ma.compressed(peaks_val)

    keep_searching = True
    
    while keep_searching == True:
    
        maxidx = np.argmax(absolute(peaks_val))
        idxplus1 = maxidx + 1
        idxplus2 = maxidx + 2
        idxminus1 = maxidx - 1
        idxminus2 = maxidx - 2
        
        if maxidx == (peaks_val.shape[0]-2):
            idxplus2 = 0
        
        if maxidx == (peaks_val.shape[0]-1):
            idxplus1 = 0
            idxplus2 = 1
        
        if peaks_val[idxminus1]*peaks_val[maxidx] < 0 : 
            peaks_val[idxminus1] = 0
    
        elif peaks_val[idxminus2]*peaks_val[maxidx] < 0 :
            peaks_val[idxminus2] = 0
    
        if peaks_val[idxplus1]*peaks_val[maxidx] < 0 :
            peaks_val[idxplus1] = 0
    
        elif peaks_val[idxplus2]*peaks_val[maxidx] < 0 :
            peaks_val[idxplus2] = 0
    
        peaks_val = np.ma.masked_equal(peaks_val, peaks_val[maxidx])
    
        if np.amax(absolute(peaks_val)) == 0 :
            keep_searching = False

    peaks_val.mask = np.ma.nomask
    peaks_val = np.ma.masked_equal(peaks_val, 0)
    peaks = np.ma.masked_array(peaks, peaks_val.mask)
    peaks_val = np.ma.compressed(peaks_val)
    peaks = np.ma.compressed(peaks)
    
    return derivative, peaks, peaks_val

#############################################

