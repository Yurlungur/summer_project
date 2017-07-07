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

def Vandermonde(N):

    x_j = T.Chebyshev.basis(N+1).roots()    # collocation points
    w_j = pi/(N+1)*np.ones(N+1)             # analytical weights
    invnorm = np.empty(N+1)                 # 1/L2norm(T_i)
    invnorm[0] = 1.0/pi
    invnorm[1:] = 2.0/pi
    Tmatrix = np.empty((N+1, N+1))          # T_i(x_j) matrix
    c2s = np.empty((N+1, N+1))              # collocation-to-spectral
    
    for n in range(N+1):
        
        Tmatrix[n,:] = T.Chebyshev.basis(n)(x_j) 
        
    c2s = np.multiply(np.outer(invnorm, w_j), Tmatrix)
    s2c = np.linalg.inv(c2s)                # spectral-to-collocation

    return c2s, s2c

#############################################

def Decompose(func, N, Vmatrix):
    
    x_i = T.Chebyshev.basis(N+1).roots()    # collocation points
    func = np.frompyfunc(func, 1, 1)        # vectorising lambda func
    u_i = func(x_i)
    a_n = np.dot(Vmatrix, u_i)              # coefficients using
                                            # Vandermonde matrix
    return a_n

#############################################

def FilterCoeff(N, a_n, s):                 # exponential filtering

    c = 36
    eta = 1.0/float(N)*np.arange(N+1)
    sigma = exp((-c)*np.power(eta,s))
    
    af_n = np.multiply(a_n, sigma)

    return af_n, sigma

#############################################

def EdgeDetectI(N, alpha):                  # returns discrete concentration
                                            # coefficients for Fourier expansion

    # alpha parameter aplies to 
    # Fourier c.f.to calculate Si(alpha)

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

def EdgeDetectII(N, alpha):             # returns discrete concentration
                                        # coefficients for Chebyshev expansion                  
    # alpha parameter aplies to         
    # Fourier c.f.to calculate Si(alpha)

    si = lambda x: sin(x)/x
    Si_a = quad(si, 0.0, alpha)[0]
    Si_p = quad(si, 0.0, pi)[0]

    dx = 2*pi/(2*N+1)*np.ones(N)
    k = np.arange(N) + 1
   
    dirichlet = np.divide(-pi/log(N), k)
    dirichlet = np.insert(dirichlet, 0, 0)
    
    fourier = np.divide(np.multiply(-pi/Si_a, sin(np.multiply(k, alpha/N))), k)
    fourier = np.insert(fourier, 0, 0)
    
    gibbs = np.divide(np.multiply(-pi/Si_p, sin(np.multiply(k, pi/N))), k)
    gibbs = np.insert(gibbs, 0, 0)

    poly1 = -pi/N*np.ones(N)
    poly1 = np.insert(poly1, 0, 0)

    poly2 = np.multiply(-2.0*pi/(N*(N+1)), k)
    poly2 = np.insert(poly2, 0, 0)

    a = 6
    b = pi
    e = 1.0/N
    
    fun = lambda x: exp(1.0/(a*x*(x-1)))
    integr = quad(fun, e, 1-e )[0]
    gamma = pi/integr
    eta = k[:-1]/float(N)
    
    expon = -(gamma * eta * fun(eta))/k[:-1]  
    expon = np.insert(expon, 0, 0)
    expon = np.append(expon, 0)

    return dirichlet, fourier, gibbs, poly1, poly2, expon

#############################################

def EdgeDetectIIC(N, alpha):             # returns discrete concentration
                                        # coefficients

    # alpha parameter aplies to 
    # Fourier c.f.to calculate Si(alpha)

    si = lambda x: sin(x)/x
    Si_a = quad(si, 0.0, alpha)[0]
    Si_p = quad(si, 0.0, pi)[0]

    dx = 2*pi/(2*N+1)*np.ones(N)
    k = np.arange(N) + 1
    
    dirichlet = -pi/log(N)*np.ones(N)
    
    fourier = np.multiply(-pi/Si_a, sin(np.multiply(k, alpha/N)))
    
    gibbs = np.multiply(-pi/Si_p, sin(np.multiply(k, pi/N)))

    poly1 = np.multiply(-pi/N, k)

    poly2 = np.multiply(-2.0*pi/(N*(N+1)), np.power(k, 2))

    a = 6
    b = pi
    e = 1.0/N
    
    fun = lambda x: exp(1.0/(a*x*(x-1)))
    integr = quad(fun, e, 1-e )[0]
    gamma = pi/integr
    
    expon = gamma*(k/N)*fun(k/N)
    expon[-1] = 0

    return dirichlet, fourier, gibbs, poly1, poly2, expon 

#############################################

def EdgeDetectIID(N, alpha):                  # returns discrete concentration
                                            # coefficients

    # alpha parameter aplies to 
    # Fourier c.f.to calculate Si(alpha)

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

def ModalD(N, c2s):                    # get modal differentiation matrix

    x_l = T.Chebyshev.basis(N+1).roots()    # collocation points
    T_imatrix = np.empty((N+1, N+1))        # matrix on T' evaluated at T collocantion
                                            # points
    for n in range(N+1):
        T_imatrix[:,n] = T.Chebyshev.basis(n).deriv()(x_l)

    modDtrans = np.matmul(c2s, T_imatrix)
    modalD = modDtrans.T

    return modalD

#############################################

def NodalD(N, c2s, s2c):                    # get nodal differentiation matrix

    modalD = ModalD(N, c2s, s2c)            
    nodalD = np.linalg_multidot([s2c, modalD, c2s])

    return nodalD

#############################################

if __name__ == "__main__":
    
    #start_time = time.time()
    
    N = 32
    #w_j, Vmatrix = VMatrix(N)
    
    ModalD(N)

    sys.exit()
    #alpha = 1.0

    #EdgeDetectI(N, alpha)

    #w_j, Vmatrix = VMatrix(N)
    #Vmatrix_inv = np.linalg.inv(Vmatrix)

    #print('----------------------Run time for N=%d is %.2f seconds------------------------------' %(N, time.time()-start_time))
