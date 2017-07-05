### filter.py
### Program for spectral decomposition of a Gaussian (g) and a Top Hat (chi)
### function in the Chebyshev basis (T_i) apllying an exponential filter sigma(eta).
### Outputs plots of function reconstructions and coefficient vs N for an input
### set of s values (s_val) in sigma=exp(-c*(eta**s)) with eta=(i/N)
### written by Joanna Piotrowska

from __future__ import print_function

import time
import os, sys
import numpy as np
import scipy as sp
from numpy import pi, sqrt, exp, cos, sin, absolute
from numpy.polynomial import chebyshev as Tsch
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib import rcParams

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

def Decompose(func, N, rec):

    coll_pts = lambda i: cos((2*i+1)*pi/(2*N+2))
    coll_pts = np.frompyfunc(coll_pts, 1, 1)
    x_i = coll_pts(np.arange(N+1))
    w_i = pi/(N+1)
    func = np.frompyfunc(func, 1, 1)
    u_i = func(x_i)

    if rec == False:
        u_n = np.empty(N/2+1)

        for n in range(N+1):
            if (n % 2 == 1):
                continue
            
            T_n = Tsch.Chebyshev.basis(n)
            p_i = T_n(x_i)
            gamma_n = np.multiply(np.power(p_i, 2), w_i)
            gamma_n = np.sum(gamma_n)
            u_tylda = np.multiply(u_i, p_i)
            u_tylda = np.multiply(u_tylda, w_i)
            u_tylda = (1.0/gamma_n)*np.sum(u_tylda)  
            u_n[n/2] = u_tylda
    
    else:
        u_n = np.empty(N+1)
    
        for n in range(N+1):
            T_n = Tsch.Chebyshev.basis(n)
            p_i = T_n(x_i)
            gamma_n = np.multiply(np.power(p_i, 2), w_i)
            gamma_n = np.sum(gamma_n)
            u_tylda = np.multiply(u_i, p_i)
            u_tylda = np.multiply(u_tylda, w_i)
            u_tylda = (1.0/gamma_n)*np.sum(u_tylda)  
            u_n[n] = u_tylda
    
    return u_n

#############################################

def DecomposeFilter(func, N, s, rec):

    c = 36
    coll_pts = lambda i: cos((2*i+1)*pi/(2*N+2))
    coll_pts = np.frompyfunc(coll_pts, 1, 1)
    x_i = coll_pts(np.arange(N+1))
    w_i = pi/(N+1)
    func = np.frompyfunc(func, 1, 1)
    u_i = func(x_i)

    if rec == False:
        u_n = np.empty(N/2 + 1)
        sigma_n = np.empty(N/2 +1)
    
        for n in range(N+1):
            if (n % 2 == 1):
                continue

            eta = float(n)/float(N)
            eta_pow = eta**s
            sigma = exp((-c)*eta_pow)
            T_n = Tsch.Chebyshev.basis(n)
            p_i = T_n(x_i)
            gamma_n = np.multiply(np.power(p_i, 2), w_i)
            gamma_n = np.sum(gamma_n)
            u_tylda = np.multiply(u_i, p_i)
            u_tylda = np.multiply(u_tylda, w_i)
            u_tylda = (1.0/gamma_n)*np.sum(u_tylda)
            u_filtered = sigma*u_tylda
            sigma_n[n/2] = sigma
            u_n[n/2] = u_filtered  

    else:
        u_n = np.empty(N+1)
        sigma_n = np.empty(N+1)
        
        for n in range(N+1):
            eta = float(n)/float(N)
            eta_pow = eta**s
            sigma = exp((-c)*eta_pow)
            T_n = Tsch.Chebyshev.basis(n)
            p_i = T_n(x_i)
            gamma_n = np.multiply(np.power(p_i, 2), w_i)
            gamma_n = np.sum(gamma_n)
            u_tylda = np.multiply(u_i, p_i)
            u_tylda = np.multiply(u_tylda, w_i)
            u_tylda = (1.0/gamma_n)*np.sum(u_tylda)
            u_filtered = sigma*u_tylda
            sigma_n[n] = sigma
            u_n[n] = u_filtered

    return u_n, sigma_n

#############################################

def PlotSigma(func, x, N, title, rec):
    
    s_val = np.linspace(2, 100, num=50)
    error = np.empty(len(s_val))
    
    func = np.frompyfunc(func, 1, 1)
    y = func(x)

    fig, ax = plt.subplots()
    plt.title('Sigma for different s values')
    sigma_idx = np.arange(N+1)
    
    for idx, s in enumerate(s_val):
        
        w_i = pi/(N+1)
        coeff, sigma = DecomposeFilter(func, N, s, rec)
        yN = Tsch.chebval(x, coeff)
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

def Plot(func, x, title, s, N, rec):

    fig, ax = plt.subplots()
    
    if rec == True:

        plt.xlim(-1,1)
        plt.ylim(-0.5, 1.5)
        plt.title('%s function and interpolants s=%.2f N=%.2f' %(title, s, N))
    
        func = np.frompyfunc(func, 1, 1)
        y = func(x)
        plt.plot(x, y, '-k', label='Function')
    
        w_i = pi/(N+1)
        coeff, sigma = DecomposeFilter(func, N, s, rec)
        coeff2 = Decompose(func, N, rec)
        coeff3, sigma3 = DecomposeFilter(func, N, 8, rec)
        yN = Tsch.chebval(x, coeff)
        yN2 = Tsch.chebval(x, coeff2)
        yN3 = Tsch.chebval(x, coeff3)

        plt.plot(x, yN, '--r', label='s=%d' %s)
        plt.plot(x, yN3, '--g', label='s=8')
        plt.plot(x, yN2, '-b', label='No filtering')

        plt.legend()
        plt.savefig('%s_filter_s%d_N%d' %(title, s, N))

    else:

        plt.title('Chebyshev coefficients')
        coeff, sigma = DecomposeFilter(func, N, s, rec) 
        coeff2 = Decompose(func, N, rec)
        coeff3, sigma3 = DecomposeFilter(func, N, 8, rec)
        coeff_idx = 2*np.arange(len(coeff))
        
        plt.semilogy(coeff_idx, absolute(coeff2), '-b', label='Raw')
        plt.semilogy(coeff_idx, absolute(coeff), '--r', label='s=%d filter' %s) 
        plt.semilogy(coeff_idx, absolute(coeff3), '--g', label='s=8 filter')

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
    N_val = np.array([32, 64], dtype='int')
    s_val = np.array([2, 4, 6, 8, 10], dtype='int')

    for N in N_val:
        
        if rec == True:
            PlotSigma(g, x, N, str('Gaussian'), rec)
            PlotSigma(chi, x, N, str('Top_hat'), rec)

        for s in s_val:
            Plot(chi, x,  str('Top_hat'), s, N, rec)
            Plot(g, x,  str('Gaussian'), s, N, rec)

    total_time = time.time() - start_time

    print('------------------Run time %.3f seconds------------------' %total_time)
    
