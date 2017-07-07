### edgedetect.py
### Short program which computes function projection onto 
### Chebyshev basis,applies exponential filtering and performs
### edge detection, using different methods
### written by Joanna Piotrowska

from __future__ import print_function

import time
import os, sys
import numpy as np
import scipy as sp
from numpy import pi, sqrt, exp, cos, sin, absolute
from numpy.polynomial import chebyshev as T
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib import rcParams
from spectral_tools2 import Vandermonde, Decompose, FilterCoeff, \
        EdgeDetectI, EdgeDetectII, EdgeDetectIID, EdgeDetectIIC, \
        ModalD, NodalD

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

def PlotSigma(func, x, N, Vmatrix, title):
    
    s_val = np.linspace(2, 100, num=50)
    error = np.empty(len(s_val))
    
    func = np.frompyfunc(func, 1, 1)
    y = func(x)

    fig, ax = plt.subplots()
    plt.title('Sigma for different s values')
    sigma_idx = np.arange(N+1)
    
    for idx, s in enumerate(s_val):
        
        w_i = pi/(N+1)
        a_n = Decompose(func, N, Vmatrix)
        af_n, sigma = FilterCoeff(N, a_n, s)
        yN = T.chebval(x, af_n)
        err = w_i * np.power(np.subtract(y,yN), 2)
        err = sqrt(np.sum(err))
        error[idx] = err

        plt.plot(sigma_idx, sigma) 

    plt.savefig('%s_sigma' %title)
    plt.clf()
    
    plt.title('Error vs s')
    plt.plot(s_val, error)
    plt.savefig('%s_error_vs_s_N%d' %(title, N))
    plt.close()

############################################

def PlotEdgeF(func, x, title, s, N, Vmatrix):

    #fig, ax = plt.subplots()
    a_n = Decompose(func, N, Vmatrix)
    af_n, sigma = FilterCoeff(N, a_n, s) 

    alpha = 1.0
    dirichlet, fourier, gibbs, poly1, poly2 = EdgeDetectI(N, alpha)
    factors = np.vstack((dirichlet, fourier, gibbs, poly1, poly2))
    factor_names = np.array([str('dirichlet'), str('fourier'), str('gibbs'), \
            str('poly1'), str('poly2')])
    theta = np.zeros((len(x), N))
    theta[:,0] = np.arccos(x)
    peak1 = np.empty(len(factors[:,0]))
    peak2 = np.empty(len(factors[:,0]))
    
    for k in range(N):
        theta[:,k] = (k+1)*theta[:,0]

    sintheta = np.sin(theta)

    for i in range(len(factors[:,0])):
        
        #plt.xlim(-1,1)
        #plt.ylabel('[f](x)')
        #plt.xlabel('x')
        #plt.title('%s function jump values for s=%.2f N=%.2f' %(title, s, N))
        
        tau = factors[i,:]
        name = factor_names[i]
        a_nk = np.multiply(a_n[1:], tau)
        af_nk = np.multiply(af_n[1:], tau)
        series = np.multiply(a_nk, sintheta)
        seriesf = np.multiply(af_nk, sintheta)
        jump = np.sum(series, axis=1)
        jumpf = np.sum(seriesf, axis=1)

        maxi = np.unravel_index(np.argmax(jumpf), jumpf.shape)
        mini = np.unravel_index(np.argmin(jumpf), jumpf.shape)
        peak1[i] = x[mini]
        peak2[i] = x[maxi]


        #print('First maximum is at x=%.2f' %x[mini])
        #print('Second maximum is at x=%.2f' %x[maxi])

        #plt.plot(x, jump, 'r', label='raw Chebyshev')
        #plt.plot(x, jumpf, 'b', label='filtered Chebyshev s=%d' %s)

        #plt.legend()
        #plt.savefig('F%s_edge_%s_s%d_N%d' %(title, name, s, N))
        #plt.cla()

    #plt.close()

    return peak1, peak2

############################################

def PlotEdgeT(func, x, title, N, c2s, modalD):

    #fig, ax = plt.subplots()
    a_n = Decompose(func, N, c2s)
    P = np.multiply(pi, sqrt(1.0-np.power(x, 2)))

    alpha = 1.0
    dirichlet, fourier, gibbs, poly1, poly2, expon = EdgeDetectII(N, alpha)
    factors = np.vstack((dirichlet, fourier, gibbs, poly1, poly2, expon))
    factor_names = np.array([str('dirichlet'), str('fourier'), str('gibbs'), \
            str('poly1'), str('poly2'), str('expon')])
    peak1, peak2 = np.empty(len(factors[:,0])), np.empty(len(factors[:,0]))
    a_modalD = np.dot(a_n, modalD)

    for i in range(len(factors[:,0])):
        
        tau = factors[i,:]
        name = factor_names[i]
        
        #plt.xlim(-1,1)
        #plt.ylabel('[f](x)')
        #plt.xlabel('x')
        #plt.title('%s %s edge detection for N=%d' %(title, name, N))
        
        a_nk = np.multiply(a_modalD, tau)
        jump = T.chebval(x, a_nk)
        jump = np.multiply(jump, P)

        maxi = np.unravel_index(np.argmax(jump), jump.shape)
        mini = np.unravel_index(np.argmin(jump), jump.shape)
        peak1[i] = x[mini]
        peak2[i] = x[maxi]
        
        #plt.plot(x, jump)
        #plt.savefig('TC%s_edge_%s_N%d' %(title, name, N))
        #plt.cla()

    #plt.close()
    return peak1, peak2

 #############################################


if __name__ == "__main__":

    start_time = time.time()
 
    ConfigurePlots()

    sigma_g = 0.5/3.0
    mu = 0.0
    chi = lambda x: 1.0 if (x>=-0.5 and x<=0.5) else 0
    g = lambda x: exp(-(x-mu)**2/(2.0*(sigma_g**2))) 
    
    x = np.linspace(-1, 1, num=201)
    s_val = np.array([4, 6])
    
    N_val = np.linspace(6, 94, num=45, dtype='int')
    
    N = 32
    c2s, s2c = Vandermonde(N)
    modalD = ModalD(N, c2s)
    PlotEdgeT(chi, x, str('Top_hat'), N, c2s, modalD)

    peaks1T = np.empty((len(N_val), 6))
    peaks2T = np.empty((len(N_val), 6))
    
    for idx, N in enumerate(N_val):
    
        c2s, s2c = Vandermonde(N)
        modalD = ModalD(N, c2s)
        peak1T, peak2T = PlotEdgeT(chi, x, str('Top_hat'), N, c2s, modalD)
        peaks1T[idx,:] = peak1T
        peaks2T[idx,:] = peak2T

    fig, ax = plt.subplots()
    for idx in range(6):

        plt.title('Position of 1st edge vs N filter no. %d' %idx)
        plt.xlabel('N')
        plt.ylabel('Edge position')
        plt.xlim(0,100)
        plt.ylim(-0.65, -0.35)
        plt.scatter(N_val, peaks1T[:,idx], marker='x', c='r', edgecolor='None')
        plt.savefig('peaks%d' %idx)
        plt.clf()

    #for N in N_val:
        
    #    print('Constructing the Vandermonde matrix for N=%d' %N)
    #    w_i, Vmatrix = VMatrix(N)
    #    print('Matrix successfully constructed for N=%d' %N)
        
    #    if rec == True:
    #        PlotSigma(g, x, N, Vmatrix, str('Gaussian'))
    #        PlotSigma(chi, x, N, Vmatrix, str('Top_hat'))

    #    for s in s_val:
    #        Plot(chi, x,  str('Top_hat'), s, N, Vmatrix, rec)
    #        Plot(g, x,  str('Gaussian'), s, N, Vmatrix, rec)

    #total_time = time.time() - start_time

    #print('------------------Run time %.3f seconds------------------' %total_time)
    
