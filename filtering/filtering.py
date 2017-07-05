### filtering.py
### Short program which  computes function projection onto 
### Chebyshev basis and applies exponential filtering
### to calculated coefficients
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
from spectral_tools import VMatrix, Decompose, FilterCoeff

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
    rcParams['axes.labelsize'] = 20
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

def Plot(func, x, title, s, N, Vmatrix, rec):

    fig, ax = plt.subplots()
    
    a_n = Decompose(func, N, Vmatrix)
    af_n, sigma = FilterCoeff(N, a_n, s) 
    af_n2, sigma2 = FilterCoeff(N, a_n, 8)
    
    
    if rec == True:

        plt.xlim(-1,1)
        plt.ylim(-0.5, 1.5)
        plt.title('%s function and interpolants s=%.2f N=%.2f' %(title, s, N))
    
        func = np.frompyfunc(func, 1, 1)
        y = func(x)
        plt.plot(x, y, '-k', label='Function')
    
        yN = T.chebval(x, a_n)
        yN2 = T.chebval(x, af_n)
        yN3 = T.chebval(x, af_n2)

        plt.plot(x, yN2, '--r', label='s=%d' %s)
        plt.plot(x, yN3, '--g', label='s=8')
        plt.plot(x, yN, '-b', label='No filtering')

        plt.legend()
        plt.savefig('%s_filter_s%d_N%d' %(title, s, N))

    else:

        plt.title('Chebyshev coefficients')
        a_n_idx = np.arange(len(a_n))
        a_n = absolute(a_n)
        af_n = absolute(af_n)
        af_n2 = absolute(af_n2)

        isodd = np.mod(a_n_idx, 2)

        a_n = np.ma.masked_where(isodd==1 , a_n) 
        af_n = np.ma.masked_array(af_n, a_n.mask) 
        af_n2 = np.ma.masked_array(af_n2, a_n.mask) 
        a_n_idx = np.ma.masked_array(a_n_idx, a_n.mask)
        a_n, af_n, af_n2, a_n_idx = a_n.compressed(), af_n.compressed(), af_n2.compressed(), a_n_idx.compressed()

        plt.semilogy(a_n_idx, a_n, '-b', label='Raw')
        plt.semilogy(a_n_idx, af_n, '--r', lw=3, label='s=%d filter' %s) 
        plt.semilogy(a_n_idx, af_n2, '--g', lw=3, label='s=8 filter')

        plt.legend(loc='lower left')
        plt.savefig('%s_filter_coeff_s%d_N%d' %(title, s, N))

    print('N=%d s=%d spectral decomposition plotted' %(N, s))
    plt.close()


 #############################################


if __name__ == "__main__":

    start_time = time.time()
 
    ConfigurePlots()

    sigma_g = 0.5/3.0
    mu = 0.0
    chi = lambda x: 1.0 if (x>=-0.5 and x<=0.5) else 0
    g = lambda x: exp(-(x-mu)**2/(2.0*(sigma_g**2))) 
    
    rec = False
    x = np.arange(-1, 1, 0.01)
    N_val = np.array([32, 64])
    s_val = np.array([2, 4, 6, 8, 10])

    for N in N_val:
        
        print('Constructing the Vandermonde matrix for N=%d' %N)
        w_i, Vmatrix = VMatrix(N)
        print('Matrix successfully constructed for N=%d' %N)
        
        if rec == True:
            PlotSigma(g, x, N, Vmatrix, str('Gaussian'))
            PlotSigma(chi, x, N, Vmatrix, str('Top_hat'))

        for s in s_val:
            Plot(chi, x,  str('Top_hat'), s, N, Vmatrix, rec)
            Plot(g, x,  str('Gaussian'), s, N, Vmatrix, rec)

    total_time = time.time() - start_time

    print('------------------Run time %.3f seconds------------------' %total_time)
    
