### edgedetect.py
### Short program which performs edge detection by applying
### concentration coefficients to spectral expansion of function
### of interest in the Chebyshev First Derivatives basis.
### written by Joanna Piotrowska

from __future__ import print_function

import time
import os, sys
import numpy as np
import scipy as sp
from numpy import pi, sqrt, exp, cos, sin, absolute, log, log10
from numpy.polynomial import chebyshev as T
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib import rcParams
from spectral_tools import Decompose, EdgeDetectF, EdgeDetectT, \
        FilterCoeff, ModalD, NodalD, Vandermonde

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

def Enhance(a_n, x, edge, p):

    N = a_n.shape[0]-1
    S_N = T.chebval(x, a_n)
    
    eps = log(N)/N
    J = 0.9*(np.amax(S_N)-np.amin(S_N))*(N**2)

    edge = (eps**(-p/2))*(edge**p)
    edge[absolute(edge) <= J] = 0

    return edge

#############################################

if __name__ == "__main__":

    start_time = time.time()
 
    
    
    total_time = time.time() - start_time

    print('------------------Run time %.3f seconds------------------' %total_time)
    
