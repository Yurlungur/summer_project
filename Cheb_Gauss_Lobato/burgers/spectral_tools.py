### spectral_tools.py
### Short program, which defines useful functions for spectral
### decomposition and edge detection
### written by Joanna Piotrowska



from __future__ import print_function

import os, sys
import time
import numpy as np
import scipy as sp
from numpy import pi, sqrt, exp, cos, sin, log, log10, absolute
from numpy.polynomial import chebyshev as T
from scipy.integrate import quad
import matplotlib.pyplot as plt


#############################################

def Decompose(func, c2s):

    """
    Compute spectral expansion coefficients of a function, given the
    highest Chebyshev polynomial order N and c2s transformation matrix
    using Gauss-Lobatto-Chebyshev quadrature

    Parameters
    ----------
    func : function object
           function to be decomposed

    c2s : 2D array, shape = (N+1, N+1)
          collocation point value-to-spectral coefficient transformation matrix. 

    Returns
    ----------

    a_n : 1D array, shape = (N+1,)
          vector of Chebyshev expansion coefficients

    """
    
    N = c2s.shape[0]-1
    x_j = T.Chebyshev.basis(N).deriv().roots()
    x_j = np.insert(x_j, 0, -1.0)
    x_j = np.append(x_j, 1.0)
    func = np.frompyfunc(func, 1, 1)        
    u_j = func(x_j)
    a_n = np.dot(c2s, u_j)             

    return a_n

#############################################

def EdgeDetectF(N, alpha):                 
    
    """
    Compute discrete concentration coefficients for Fourier expansion in edge
    detection after Gelb&Tadmor I, for k in [1,N]

    Parameters
    ----------

    N : int
        highest order in Chebyshev expansion
        
    alpha : float
            parameter setting the value of Si(alpha) integral. Required for 
            Fourier concentration coefficient

    Returns
    ----------

    dirichlet : 1D array, shape = (N,)
                dirichlet concentration coefficients (-pi/logN) for discrete case

    fourier : 1D array, shape = (N,)
              fourier concentration coefficients (-pi/Si(alpha))*sin(k*alpha/N) 
              for discrete case
    
    gibbs : 1D array, shape = (N,)
            gibbs concentration coefficients (-pi/Si(pi))*sin(k*pi/N) for discrete 
            case
    
    poly1 : 1D array, shape = (N,)
            1st degree polynomial concentration coefficients (-pi*k/N) for discrete 
            case
    
    poly2 : 1D array, shape = (N,)
            2nd degree polynomial concentration coefficients (-2pi/N(N+1))*k^2 for 
            discrete case

    """

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

def EdgeDetectT(N, alpha):             
    
    """
    Compute discrete concentration coefficients for Chebyshev expansion in edge
    detection after Gelb&Tadmor II, for k in [0,N]

    Parameters
    ----------

    N : int
        highest order in Chebyshev expansion
        
    alpha : float
            parameter setting the value of Si(alpha) integral. Required for 
            Fourier concentration coefficient

    Returns
    ----------

    dirichlet : 1D array, shape = (N+1,)
                dirichlet concentration coefficients (-pi/logN)

    fourier : 1D array, shape = (N+1,)
              fourier concentration coefficients (-pi/Si(alpha))*sin(k*alpha/N) 
    
    gibbs : 1D array, shape = (N+1,)
            gibbs concentration coefficients (-pi/Si(pi))*sin(k*pi/N) 
    
    poly1 : 1D array, shape = (N+1,)
            1st degree polynomial concentration coefficients (-pi*k/N) 
    
    poly2 : 1D array, shape = (N+1,)
            2nd degree polynomial concentration coefficients (-2pi/N(N+1))*k^2

    expon : 1D array, shape = (N+1,)
            exponential concentration coefficients -(gamma*eta*exp(1/(a*eta*(eta-1))))
            where gamma = 1/integral(exp(1/(a*x*(x-1)))) between e and 1-e, e = 1/N,
            eta = i/N for i in [0,N] and a is a free parameter. 

    Notes
    ----------
    k=0 coefficients are equal to 0 for all coefficient types

    """
    
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

    a = 6.0
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

def EdgeDetectIII(a_n):

    N = a_n.shape[0]-1
    k = np.arange(N) + 1
    eta = k/float(N)
    B = pi
    a = 4
    p_val = np.array([1])
    eps = 1.0/N 
    si = lambda x: sin(x)/x
    Si = quad(si, 0.0, pi)[0]
    expfun = lambda x: exp(1.0/(a*x*(x-1)))
    gamma = pi/(quad(expfun, eps, 1-eps)[0])

    trig_fun = lambda x: (pi*sin(B*x))/Si
    trig = (pi*sin(B*eta))/(Si)
    trig = np.insert(trig, 0, 0)
    
    poly = np.empty((a_n.shape[0], len(p_val)))
    
    for idx, p in enumerate(p_val):
        poly_p = pi*(eta**p)
        poly_p = np.insert(poly_p, 0, 0)
        poly[:,idx] = poly_p
        poly_fun = lambda x: pi*(x**p)

    expon_fun = lambda x: gamma * x * expfun(x)
    expon = gamma * eta[:-1] * expfun(eta[:-1])
    expon = np.insert(expon, 0, 0)
    expon = np.append(expon, 0)

    sigmas = np.vstack((trig, poly.T, expon))
    #sigmas = np.vstack((trig, poly.T))
    sigmas = sigmas.T

    return sigmas, trig_fun, poly_fun, expon_fun

#############################################

def Lanczos(sigmas, trig_fun, poly_fun, expon_fun, n):

    N = sigmas[:,0].shape[0]-1
    k = np.arange(N) + 1
    eta = k/float(N)
    
    integr_trig = lambda x: (trig_fun(x)/x) * ((sin(pi*x)/(pi*x))**n)
    gamma_trig = pi/(quad(integr_trig, 0, 1)[0])
    trig_lanczos = gamma_trig*sigmas[1:,0]*np.power(sin(eta*pi)/(eta*pi), n)
    trig_lanczos = np.insert(trig_lanczos, 0, 0)
    

    poly1_fun = lambda x: pi*x
    integr_poly1 = lambda x: (poly1_fun(x)/x) * ((sin(pi*x)/(pi*x))**n) 
    gamma_poly1 = pi/(quad(integr_poly1, 0, 1)[0])
    poly1_lanczos = gamma_poly1*sigmas[1:,1]*np.power(sin(eta*pi)/(eta*pi), n)
    poly1_lanczos = np.insert(poly1_lanczos, 0, 0)

    #integr_poly2 = lambda x: (poly_fun(x)/x) * ((sin(pi*x)/(pi*x))**n) 
    #gamma_poly2 = pi/(quad(integr_poly2, 0, 1)[0])
    #poly2_lanczos = gamma_poly2*sigmas[1:,1]*np.power(sin(eta*pi)/(eta*pi), n)
    #poly2_lanczos = np.insert(poly2_lanczos, 0, 0)

    integr_expon = lambda x: (expon_fun(x)/x) * ((sin(pi*x)/(pi*x))**n)
    gamma_expon = pi/(quad(integr_expon, 0, 1)[0])
    expon_lanczos = gamma_expon*sigmas[1:,2]*np.power(sin(eta*pi)/(eta*pi), n)
    expon_lanczos = np.insert(expon_lanczos, 0, 0)
    
    sigmas_lanczos = np.vstack((trig_lanczos, poly1_lanczos, expon_lanczos))
    #sigmas_lanczos = np.vstack((trig_lanczos, poly1_lanczos, poly2_lanczos))
    sigmas_lanczos = sigmas_lanczos.T

    return sigmas_lanczos

#############################################

def ZeroCrossing(sigmas):
    
    N = sigmas[:,0].shape[0] - 1
    k = np.arange(N) + 1
    sigmas_zcw = np.zeros(sigmas.shape)

    for idx in range(len(sigmas[0,:])):

        sigma_sum = np.sum(np.power(sigmas[1:,idx], 2))
        sigmas_zcw[1:,idx] = (pi*k*np.power(sigmas[1:,idx], 2))/sigma_sum


    return sigmas_zcw


#############################################

def MatchingWaveform(sigmas):
    
    N = sigmas[:,0].shape[0] - 1
    k = np.arange(N) + 1
    sigmas_mw = np.zeros(sigmas.shape)

    for idx in range(len(sigmas[0,:])):

        sigma_sum = np.sum(np.power(sigmas[1:,idx]/k, 2))
        sigmas_mw[1:,idx] = ((pi/k)*np.power(sigmas[1:,idx], 2))/sigma_sum


    return sigmas_mw

#############################################

def FilterCoeff(a_n, s):                
    
    """

    Apply exponential filtering to coefficients for a smoother spectral
    representation

    Parameters
    ---------- 

    a_n : 1D array
          vector of Chebyshev expansion coefficients

    s : int
        variable smoothing parameter in the exponential filter of the form
        sigma = exp(-c*eta^s) where eta = i/N for i in [0,N]

    Returns
    ----------

    af_n : 1D array, shape = (N+1,)
           vector of exponentially filtered Chebyshev expansion coefficients

    sigma : 1D array, shape = (N+1,)
            vector of sigma values for given term in the Chebyshev expansion

    Notes
    ----------
    c parameter is fixed at c=36 to ensure sigma(1)=0

    """
    
    N = a_n.shape[0]-1
    c = 36
    eta = 1.0/float(N)*np.arange(N+1)
    sigma = exp((-c)*np.power(eta,s))
    
    af_n = np.multiply(a_n, sigma)

    return af_n, sigma

#############################################

def ModalD(c2s):                    

    """
    Compute Modal Differentiation Matrix D_ij

    Parameters
    ----------
         
    c2s : 2D array, shape = (N+1, N+1)
          collocation point value-to-spectral coefficient transformation matrix. 

    Returns
    ----------

    modalD : 2D array, shape = (N+1, N+1)
             Modal Differentiation Matrix

    Notes
    ----------
    Composed in a manner allowing the following action on expansion coefficients: 
    d_x(a_i) = D_ij(a_j)

    """
    
    N = c2s.shape[0]-1
    x_l = T.Chebyshev.basis(N).deriv().roots()
    x_l = np.insert(x_l, 0, -1.0)
    x_l = np.append(x_l, 1.0)
    T_imatrix = np.empty((N+1, N+1))        
    
    for n in range(N+1):
        T_imatrix[:,n] = T.Chebyshev.basis(n).deriv()(x_l)

    modalD = np.dot(c2s, T_imatrix)
    modalD[absolute(modalD) < 1.0] = 0.0

    return modalD

#############################################

def NodalD(c2s, s2c):

    """
    
    Compute Nodal Differentiation Matrix D_N_ij

    Parameters
    ----------

    c2s : 2D array, shape = (N+1, N+1)
          collocation point value-to-spectral coefficient transformation matrix. 

    s2c : 2D array, shape = (N+1, N+1)
          spectral coefficient-to-collocation point value transformation matrix. 
    
    """
    
    modalD = ModalD(c2s)            
    nodalD = np.linalg.multi_dot([s2c, modalD, c2s])

    return nodalD

#############################################

def Vandermonde(N):

    """
    Compute two matrices c2s and s2c to easily switch between the 
    collocation points and spectral coefficients in the 
    Gauss-Lobatto-Chebyshev quadrature 

    Parameters
    ----------
    N : int
        highest order in Chebyshev expansion

    Returns
    ----------

    c2s : 2D array, shape = (N+1, N+1)
          collocation point value-to-spectral coefficient transformation matrix. 

    s2c : 2D array, shape = (N+1, N+1)
          spectral coefficient-to-collocation point value transformation matrix. 
         
    Notes
    ----------
    Uses analytical Gauss-Lobatto-Chebyshev weights w_j and analytical L2 norms

    """

    x_j = T.Chebyshev.basis(N).deriv().roots()
    x_j = np.insert(x_j, 0, -1.0)
    x_j = np.append(x_j, 1.0)
    w_j = pi/(N)*np.ones(N+1)             
    w_j[0], w_j[-1] = pi/(2*N), pi/(2*N)
    invnorm = np.empty(N+1)                 # 1/L2norm(T_i) 
    invnorm[0] = 1.0/pi
    invnorm[1:] = 2.0/pi
    invnorm[-1] = 1.0/pi
    Tmatrix = np.empty((N+1, N+1))          
    
    for n in range(N+1):    
        Tmatrix[n,:] = T.Chebyshev.basis(n)(x_j) 
        
    c2s = np.outer(invnorm, w_j) * Tmatrix
    s2c = np.linalg.inv(c2s)

    assert c2s.shape == s2c.shape, "Vandermonde matrix incosistent with its inverse"

    return c2s, s2c

#############################################
