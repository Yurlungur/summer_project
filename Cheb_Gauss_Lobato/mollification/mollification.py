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
from edgedetect import ChebEdgeIII, Enhance, MinMod, LocateEdges

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


