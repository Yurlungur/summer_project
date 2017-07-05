### spectral_tools2.py
### Short program, which defines useful functions for spectral
### decomposition and edge detection using integral tricks described in README.md
### written by Joanna Piotrowska



from __future__ import print_function

import os, sys
import time
import numpy as np
import scipy as sp
from numpy import pi, sqrt, exp, cos, sin, log
from numpy.polynomial import chebyshev as T
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib import rcParams


#############################################

def VMatrix(N):

    x_j = T.Chebyshev.basis(N+1).roots()
    w = T.chebweight
    Tw_i, T2w_i = np.zeros(N+1), np.zeros(N+1)
    Tmatrix = np.empty((N+1, N+1))
    Vmatrix = np.empty((N+1, N+1))
    w_j = pi/(N+1)*np.ones(N+1)
    
    for n in range(N+1):
        
        f_n = lambda x: T.Chebyshev.basis(n)(x) * w(x)
        f2_n = lambda x: f_n(x) * T.Chebyshev.basis(n)(x)
        Tmatrix[n,:] = T.Chebyshev.basis(n)(x_j) 
        if n < 56:
            T2w_i[n] = 1.0/quad(f2_n, -1.0, 1.0)[0]
        else:
            T2w_i[n] = 1.0/np.sum(w_j[n]*np.power(Tmatrix[n,:],2))
        
    Tmatrix_inv = np.linalg.inv(Tmatrix)
    Vmatrix = np.multiply(np.outer(T2w_i, w_j), Tmatrix)

    return w_j, Vmatrix

#############################################

def Decompose(func, N, Vmatrix):
    
    x_i = T.Chebyshev.basis(N+1).roots()
    func = np.frompyfunc(func, 1, 1)
    u_i = func(x_i)
    a_n = np.dot(Vmatrix, u_i)

    return a_n

#############################################

def FilterCoeff(N, a_n, s):

    c = 36
    eta = 1.0/float(N)*np.arange(N+1)
    sigma = exp((-c)*np.power(eta,s))
    
    af_n = np.multiply(a_n, sigma)

    return af_n, sigma

#############################################

def EdgeDetectI(N, alpha):

    # alpha parameter is only used in Fourier concentration coefficients
    # to calculate int((sin(x)/x)*dx)

    si = lambda x: sin(x)/x
    Si_a = quad(si, 0.0, alpha)[0]
    Si_p = quad(si, 0.0, pi)[0]

    dx = 2*pi/(2*N+1)*np.ones(N)
    k = np.arange(N) + 1
    
    dirichlet = np.divide(sin(np.multiply(k, dx/2.0)), \
            np.multiply(k, dx)*log(N)/(-2.0))
    
    fourier = np.multiply(np.divide(sin(np.multiply(k, dx/2.0)), \
            np.multiply(k, dx)*Si_a/(-2.0)), \
            sin(k*(alpha/N)))

    gibbs = np.multiply(np.divide(sin(np.multiply(k, dx/2.0)), \
            np.multiply(k, dx)*Si_p/(-2.0)), \
            sin(k*(pi/N))) 
    
    poly1 = np.divide(sin(np.multiply(k, dx/2.0)), \
            dx*N/(-2.0))
    
    poly2 = np.divide(sin(np.multiply(k, dx/2.0)), \
            np.divide(dx, k)*(N*(N+1)/(-4.0*pi)))
    
    return dirichlet, fourier, gibbs, poly1, poly2

#############################################
#if __name__ == "__main__":
    
    #start_time = time.time()
    
    #N = 64

    #alpha = 1.0

    #EdgeDetectI(N, alpha)

    #w_j, Vmatrix = VMatrix(N)
    #Vmatrix_inv = np.linalg.inv(Vmatrix)

    #print('----------------------Run time for N=%d is %.2f seconds------------------------------' %(N, time.time()-start_time))
