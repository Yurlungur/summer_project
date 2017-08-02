### mollification.py
### Short program which contains tools to perform mollification
### of the spectrally reconstructed function of interest. It can
### be applied in reduction of gibbs phenomenon associated with
### discontinuities in the function of interest.
### written by Joanna Piotrowska

from __future__ import print_function

import os, sys
import numpy as np
import scipy as sp
from numpy import pi, sqrt, exp, cos, sin, absolute, log, log10
from numpy.polynomial import chebyshev as T
from numpy.polynomial import hermite_e as H
from scipy.misc import factorial
from scipy.integrate import quad
from spectral_tools import Decompose, ModalD, NodalD, Vandermonde
from edgedetect import ChebEdgeIII, Enhance, LocateEdges, MinMod


############################################

def MollifyQuad(theta, c_j, a_n, x):
    
    """
    Mollify the spectral reconstruction of a discontinuous function 
    to reduce the effect of Gibbs phenomenon. Perform a real-space convolution 
    of a spectral reconstruction with an adaptive unit-mass mollifier.

    Parameters
    ----------
    theta : float
            free parameter to vary in the range 0 < theta < 1

    c_j : 1D array, 
          Array containing x positions of the discontinuities in the data
 
    a_n : 1D array, shape = (N+1,)
          N - highest order in the Chebyshev expansion
          vector of Chebyshev expansion coefficients

    x : 1D array,
        1D grid of points in real space, where the function is evaluated

    Returns
    ----------
    
    convolution : 1D array, shape = (len(x),)
                  real space mollified representation of a discontinuous function


    """

    N = a_n.shape[0]-1
    offset = x
    convolution = np.empty(len(x))
    
    I_N = lambda y: T.chebval(y, a_n)
    I_Nf = lambda y: I_N(y) if -1 <= y <= 1 else 0
    
    for idx, off_x in enumerate(offset):
        c_jx = c_j - off_x
        dx = lambda y: sqrt(theta*N*min(abs(y-c) for c in c_jx))
        var = lambda y: N/(2*theta*dx(y))
        p_N = lambda y: (theta**2)*dx(y)*N
        j_max = int(np.amax(np.frompyfunc(p_N,1,1)(x)))
        h_j = np.zeros(2*(j_max+1))
    
        for j in range(j_max+1):
            h_j[2*j] = ((-1)**j)/((4**j)*factorial(j))
    
        hermite = lambda y: H.hermeval(sqrt(var(y))*y, h_j)
        expon = lambda y: (1./sqrt(theta*N*dx(y)))*exp(-var(y)*(y**2))
        phi = lambda y: hermite(y)*expon(y)
        phif = lambda y: phi(y-off_x)
        norm = quad(phi, -1.0, 1.0)[0]
        
        convfunc = lambda y: (phif(y) * I_Nf(y))
        convolution[idx] = (1/norm)*quad(convfunc, -1.0, 1.0)[0]
    
    return convolution

############################################

def MollifyQuadBuffer(theta, c_j, a_n, x):
    
    """
    Mollify the spectral reconstruction of a discontinuous function 
    to reduce the effect of Gibbs phenomenon. Perform a real-space convolution 
    of a spectral reconstruction with an adaptive unit-mass mollifier.

    Parameters
    ----------
    theta : float
            free parameter to vary in the range 0 < theta < 1

    c_j : 1D array, 
          Array containing x positions of the discontinuities in the data
 
    a_n : 1D array, shape = (N+1,)
          N - highest order in the Chebyshev expansion
          vector of Chebyshev expansion coefficients

    x : 1D array,
        1D grid of points in real space, where the function is evaluated

    Returns
    ----------
    
    convolution : 1D array, shape = (len(x),)
                  real space mollified representation of a discontinuous function

    """
    
    N = a_n.shape[0]-1
    deltax = 2.0/(len(x)-1)
    
    I_N = lambda y: T.chebval(y, a_n)
    I_Nf = lambda y: I_N(y) if -1 <= y <= 1 else 0
    buff_right = lambda y: I_N(2.0-y) if 1.0<y<1.4 else 0
    buff_left = lambda y: I_N(-(2.0+y)) if -1.4<y<-1.0 else 0
    I_Nnew = lambda y: buff_right(y) + buff_left(y) + I_Nf(y)
    
    add_left = np.arange(-1.4, -1.0, deltax)
    add_right = np.arange(1.0, 1.4, deltax)
    offset = np.hstack((add_left, x, add_right))
    convolution = np.empty(len(offset))    
    
    for idx, off_x in enumerate(offset):
        c_jx = c_j - off_x
        dx = lambda y: sqrt(theta*N*min(abs(y-c) for c in c_jx))
        var = lambda y: N/(2*theta*dx(y))
        p_N = lambda y: (theta**2)*dx(y)*N
        j_max = int(np.amax(np.frompyfunc(p_N,1,1)(x)))
        h_j = np.zeros(2*(j_max+1))
    
        for j in range(j_max+1):
            h_j[2*j] = ((-1)**j)/((4**j)*factorial(j))
    
        hermite = lambda y: H.hermeval(sqrt(var(y))*y, h_j)
        expon = lambda y: (1./sqrt(theta*N*dx(y)))*exp(-var(y)*(y**2))
        phi = lambda y: hermite(y)*expon(y)
        phif = lambda y: phi(y-off_x)
        norm = quad(phi, -1.4, 1.4)[0]
        
        convfunc = lambda y: (phif(y) * I_Nnew(y))
        convolution[idx] = (1/norm)*quad(convfunc, -1.4, 1.4)[0]
    
    convolution = convolution[len(add_left):-(len(add_right))]
    
    return convolution

############################################

def PiecewiseMollify(theta, c_j, a_n, x):

    """
    Piecewise mollify the spectral reconstruction of a discontinuous function 
    to reduce the effect of Gibbs phenomenon. Perform a real-space convolution 
    of a spectral reconstruction with an adaptive unit-mass mollifier.

    Parameters
    ----------
    theta : float
            free parameter to vary in the range 0 < theta < 1

    c_j : 1D array, 
          Array containing x positions of the discontinuities in the data
 
    a_n : 1D array, shape = (N+1,)
          N - highest order in the Chebyshev expansion
          vector of Chebyshev expansion coefficients

    x : 1D array,
        1D grid of points in real space, where the function is evaluated

    Returns
    ----------
    
    mollified : 1D array, shape = (len(x),)
                real space mollified representation of a discontinuous function

    mollified_err : 1D array, shape = (len(x),)
                    error estimate for each point in the convolution, derived from
                    scipy.integrate.quad

    """
    N = a_n.shape[0]-1
    sanity_check = np.empty(len(x))
    mollified = np.array([])
    mollified_err = np.array([])
    
    I_N = lambda y: T.chebval(y, a_n)
    chi_top = lambda y,f : f(y) if -1 <= y <= 1 else 0
    I_N_top = lambda y: chi_top(y, I_N)
    c_jplus = np.append(c_j, 1.0)
    
    for idx_c, pos in enumerate(c_jplus):
        
        if idx_c == 0:
            
            lim_left = -1.0
            lim_right = pos
            
            offset = np.ma.masked_where(x > lim_right, x)
            offset = np.ma.compressed(offset)

        else:
            
            lim_left = c_jplus[idx_c-1]
            lim_right = pos
            
            offset = np.ma.masked_where(x > lim_right, x)
            offset = np.ma.masked_where(x <= lim_left, offset)
            offset = np.ma.compressed(offset)

        chi_cut = lambda y,f : f(y) if lim_left <= y <= lim_right else 0
        convolution = np.empty(len(offset))
        convolution_err = np.empty(len(offset))

        for idx_o, off_x in enumerate(offset):
            c_jx = c_j - off_x
            dx = lambda y: sqrt(theta*N*min(abs(y-c) for c in c_jx))
            var = lambda y: N/(2*theta*dx(y))
            p_N = lambda y: (theta**2)*dx(y)*N
            j_max = int(np.amax(np.frompyfunc(p_N,1,1)(x)))
            h_j = np.zeros(2*(j_max+1))
    
            for j in range(j_max+1):
                h_j[2*j] = ((-1)**j)/((4**j)*factorial(j))
    
            hermite = lambda y: H.hermeval(sqrt(var(y))*y, h_j)
            expon = lambda y: (1./sqrt(theta*N*dx(y)))*exp(-var(y)*(y**2))
            phi0 = lambda y: hermite(y)*expon(y)
            phi_off = lambda y: phi0(y-off_x)
            phi = lambda y: chi_cut(y, phi_off)
            norm = quad(phi, lim_left, lim_right)[0]
        
            convfunc = lambda y: (1/norm)*(phi(y) * I_N_top(y))
            convolution[idx_o], convolution_err[idx_o] = quad(convfunc, lim_left, lim_right)
            
        mollified = np.append(mollified, convolution)
        mollified_err = np.append(mollified_err, convolution_err)
        
    assert mollified.shape == sanity_check.shape, "Piecewise mollification inconsistent with regular one"
    assert mollified_err.shape == sanity_check.shape, "Piecewise mollification inconsistent with regular one"

    return mollified, mollified_err

############################################
def Dist(c_j, x):
 
    dist = np.empty((len(x), len(c_j))) 
    d_x = np.empty(len(x))
 
    for i in range(len(c_j)):
        dist[:,i] = absolute(x-c_j[i])

    for i in range(len(d_x)):
        d_x[i] = np.amin(dist[i,:])


    return d_x

############################################

def Mollify(N, theta, c_j, x):

    d_x = Dist(c_j, x)
    
    phi = np.zeros(len(x))
    exp_part = np.zeros(len(x))
    herm_part = np.zeros(len(x))
    
    delta = sqrt(theta*d_x*N)
    delta = np.ma.masked_where(delta == 0.0, delta)
    
    p_N = (theta**2)*d_x*N
    j_max = int(np.amax(p_N))
    var = (N*np.power(x,2))/(2*theta*d_x)

    exp_part = (1./delta)*exp(-var)

    hermites = np.empty((len(x), j_max+1))
    
    for j in range(j_max+1):    

        Hermite = H.HermiteE.basis(2*j)(sqrt(var))
        hermites[:,j] = ((-1)**j)/((4**j)*factorial(j))*Hermite

    herm_part = np.sum(hermites, axis=1)
    phi = exp_part * herm_part
    phi = np.ma.filled(phi, 0)

    return phi

############################################


