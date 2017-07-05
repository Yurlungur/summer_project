### spectral_tools.py
### Short program, which defines useful functions for spectral
### decomposition and edge detection
### written by Joanna Piotrowska



from __future__ import print_function

import os, sys
import time
import numpy as np
import scipy as sp
from numpy import pi, sqrt, exp, cos, sin
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

    for n in range(N+1):
        
        f_n = lambda x: T.Chebyshev.basis(n)(x) * w(x)
        f2_n = lambda x: f_n(x) * T.Chebyshev.basis(n)(x)
        Tw_i[n] = quad(f_n, -1.0, 1.0)[0]
        T2w_i[n] = 1/quad(f2_n, -1.0, 1.0)[0] 
        Tmatrix[n,:] = T.Chebyshev.basis(n)(x_j)

    Tmatrix_inv = np.linalg.inv(Tmatrix)
    w_j = np.dot(Tmatrix_inv, Tw_i)
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

if __name__ == "__main__":
    
    start_time = time.time()
    
    N = 254

    w_j, Vmatrix = VMatrix(N)
    Vmatrix_inv = np.linalg.inv(Vmatrix)

    print('----------------------Run time for N=%d is %.2f seconds------------------------------' %(N, time.time()-start_time))
