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
    rcParams['savefig.format'] = 'pdf'


#############################################

def Decompose(func, N):

    coll_pts = lambda i: cos((2*i+1)*pi/(2*N+2))
    coll_pts = np.frompyfunc(coll_pts, 1, 1)
    x_i = coll_pts(np.arange(N+1))
    w_i = pi/(N+1)
    func = np.frompyfunc(func, 1, 1)
    u_i = func(x_i)

    u_n = np.empty(N/2+1)
    #u_n = np.empty(N+1)
    
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

        #u_n[n] = u_tylda
        u_n[n/2] = u_tylda

       
    return u_n

#############################################

def DecomposeFilter(func, N, s):

    c = 36

    coll_pts = lambda i: cos((2*i+1)*pi/(2*N+2))
    coll_pts = np.frompyfunc(coll_pts, 1, 1)
    x_i = coll_pts(np.arange(N+1))
    w_i = pi/(N+1)
    func = np.frompyfunc(func, 1, 1)
    u_i = func(x_i)

    u_n = np.empty(N/2 + 1)
    sigma_n = np.empty(N/2 +1)
    #u_n = np.empty(N+1)
    #sigma_n = np.empty(N+1)
    
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
        #sigma_n[n] = sigma
        sigma_n[n/2] = sigma
        
        #u_n[n] = u_filtered
        u_n[n/2] = u_filtered  
        

    return u_n, sigma_n

#############################################

def OptimizeS(func, x):
    
    N = 32
    s_val = np.linspace(2, 100, num=50)
    error = np.empty(len(s_val))
    
    func = np.frompyfunc(func, 1, 1)
    y = func(x)

    fig, ax = plt.subplots()
    plt.title('Sigma for different s values')
    
    for idx, s in enumerate(s_val):
        
        w_i = pi/(N+1)
        coeff, sigma = DecomposeFilter(func, N, s)
        yN = Tsch.chebval(x, coeff)

        err = w_i * np.power(np.subtract(y,yN), 2)
        err = sqrt(np.sum(err))
        error[idx] = err

        #sigma_idx = np.arange(len(sigma))

        #plt.plot(sigma_idx, sigma) 

    min_idx = np.unravel_index(np.argmin(error), error.shape)
    s_optimal = int(s_val[min_idx])

    #plt.legend()
    #plt.savefig('sigma')
    #plt.close()

    print('Optimal s found for N=%d is %d' %(N, s_optimal))
    
    plt.clf()
    plt.title('Error vs s')
    plt.plot(s_val, error)
    plt.savefig('Error_vs_s_N%d' %N)
        
    return s_optimal, N

############################################

def OptimizeN(func, x):
    
    N_val = np.logspace(3, 8, num=6, base=2.0, dtype='int')

    error = np.empty(len(N_val))
    
    func = np.frompyfunc(func, 1, 1)
    y = func(x)
    
    for idx, N in enumerate(N_val):
        
        s = N/4
        
        w_i = pi/(N+1)
        coeff, sigma = DecomposeFilter(func, N, s)
        yN = Tsch.chebval(x, coeff)

        err = w_i * np.power(np.subtract(y,yN), 2)
        err = sqrt(np.sum(err))
        error[idx] = err

    min_idx = np.unravel_index(np.argmin(error), error.shape)
    N_optimal = int(N_val[min_idx])
    #s = 2*(N_optimal +1.0)
    s = N/4

    print('Optimal N found is %d' % N_optimal)
        
    return N_optimal, s

#############################################

def Plot(x, func, title, s, N):

    fig, ax = plt.subplots()
    plt.xlim(-1,1)
    plt.ylim(-0.5, 1.5)
    plt.title('%s function and interpolants s=%.2f N=%.2f' %(title, s, N))
    
    func = np.frompyfunc(func, 1, 1)
    y = func(x)
    plt.plot(x, y, '-k', label='Function')
    
    w_i = pi/(N+1)
    coeff, sigma = DecomposeFilter(func, N, s)
    coeff2 = Decompose(func,N)
    yN = Tsch.chebval(x, coeff)
    yN2 = Tsch.chebval(x, coeff2)

    plt.plot(x, yN, '--r', lw='3', label='N=%d, s=%d' % (N, s))
    plt.plot(x, yN2, '-', label='No filtering')

    plt.legend()
    plt.savefig('%s_filter_s%d_N%d' %(title, s, N))
    plt.clf()

    coeff_idx = 2*np.arange(len(coeff))
    plt.title('Chebyshev coefficients')
    coeff2 = Decompose(func, N)
    coeff3, sigma3 = DecomposeFilter(func, N, 8)
    #coeff = np.ma.masked_where(coeff == 0, coeff)
    #coeff_idx = np.ma.masked_where(coeff ==0, coeff_idx)
    #coeff2 = np.ma.masked_where(coeff == 0, coeff2)
    plt.semilogy(coeff_idx, absolute(coeff2), '-b', label='Raw')
    plt.semilogy(coeff_idx, absolute(coeff), '--r', label='Filtered') 
    plt.semilogy(coeff_idx, absolute(coeff3), '--g', label='s=8 filter')


    #ax = plt.gca()
    #ax.scatter(coeff_idx, absolute(coeff2), c='blue', label='Raw')
    #ax.scatter(coeff_idx, absolute(coeff), c='red', label='Filtered')
    #ax.set_yscale('log')
    
    plt.legend()
    plt.savefig('%s_filter_coeff_s%d_N%d' %(title, s, N))

    print('N=%d s=%d spectral decomposition plotted' %(N, s))

    plt.close()


 #############################################


if __name__ == "__main__":

    start_time = time.time()
 
    ConfigurePlots()

    sigma = 0.5/3.0
    mu = 0.0
    chi = lambda x: 1.0 if (x>=-0.5 and x<=0.5) else 0
    g = lambda x: exp(-(x-mu)**2/(2.0*(sigma**2))) 
    
    x = np.arange(-1, 1, 0.01)
    
    #s_optimal, N  = OptimizeS(g, x)
    #Plot(x, g, str('Gaussian'), s_optimal, N)

    #N_optimal, s = OptimizeN(g, x) 
    #Plot(x, g, str('Gaussian'), s, N_optimal)
    
    #s_optimal, N  = OptimizeS(chi, x)
    #Plot(x, chi, str('Top_hat'), s_optimal, N)

    #N_optimal, s = OptimizeN(chi, x) 
    #Plot(x, chi, str('Top_hat'), s, N_optimal)

    N = 32
    s_val = np.linspace(2, 100, num=50)

    for s in s_val:

        Plot(x, chi, str('Top_hat'), s, N)

    total_time = time.time() - start_time

    print('------------------Run time %.3f seconds------------------' %total_time)
    
