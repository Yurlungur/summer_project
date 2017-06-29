### playground.py
### Introductory program for spectral decomposition of a Gaussian (g)
### and a top-hat (chi) function in the Chebyshev basis (T_i)
### comparing the error and coefficient behaviour with increasing
### number of basis functions (N) 
### written by Joanna Piotrowska


from __future__ import print_function

import os, sys
import time
import numpy as np
import scipy as sp
from numpy import pi, sqrt, exp, cos, sin
from numpy.polynomial import chebyshev as Tsch
from scipy.integrate import quad
from scipy.stats import linregress
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

def Integrand(func, N, x_i):

    # calculate coefficients directly by taking
    # weighted integral on [-1,1] i.e. accurate
    # projection

    w = Tsch.chebweight
    u_n = np.empty(N+1)

    for n in range(N+1):

        T_n = Tsch.Chebyshev.basis(n)
        f = lambda x: T_n(x) * func(x) * w(x)
        f2 = lambda x: T_n(x) * T_n(x) * w(x)

        u_i = quad(f, -1, 1)[0]
        p_i = quad(f2, -1, 1)[0]
        u_n[n] = u_i/p_i

    return u_n

#############################################

def Decompose(func, N):

    # calculate coefficients using Chebyshev-Gaussian
    # quadrature i.e. using collocation points

    coll_pts = lambda i: cos((2*i+1)*pi/(2*N+2))
    coll_pts = np.frompyfunc(coll_pts, 1, 1)
    x_i = coll_pts(np.arange(N+1))
    w_i = pi/(N+1)
    func = np.frompyfunc(func, 1, 1)

    u_n = np.zeros(N+1)
    
    for n in range(N+1):

        T_n = Tsch.Chebyshev.basis(n)
        p_i = T_n(x_i)
        u_i = func(x_i)
        gamma_n = np.multiply(np.power(p_i, 2), w_i)
        gamma_n = np.sum(gamma_n)
        u_tylda = np.multiply(u_i, p_i)
        u_tylda = np.multiply(u_tylda, w_i)
        u_tylda = (1.0/gamma_n)*np.sum(u_tylda)
        
        u_n[n] = u_tylda
       
    return u_n

#############################################

def Plot(x, func, N_order, title, regress):

    print('%s analysis' %title)
    
    fig, ax = plt.subplots()
    plt.xlim(-1,1)
    plt.ylim(-0.5, 1.5)
    plt.title('%s function and its spectral reconstruction' %title)

    # calculate values of original function at
    # given x-value grid

    func = np.frompyfunc(func, 1, 1)
    y = func(x)
    yN = np.empty((len(N_order), len(x)))
    yI = np.empty((len(N_order), len(x)))
    error = np.empty(len(N_order))
    
    for idx, N in enumerate(N_order):
        
        w_i = pi/(N+1)                         # Gauss-Chebyshev weights
        coeff = Decompose(func, N)             # coefficients from Gaussian quadratures
        yN[idx,:] = Tsch.chebval(x, coeff)     # evaluating the series at points x

        coeff_intgr = Integrand(func, N, x)         # coefficients from integration
        yI[idx,:] = Tsch.chebval(x, coeff_intgr)    # evaluating the series at points x
        
        diff = np.subtract(y, yN[idx,:])
        
        err = w_i * np.power(diff, 2)              # error estimation using Gauss quadrature
        err = sqrt(np.sum(err))                    # approximation
        error[idx] = err
        
        print('Integral and Gauss quadratures computed for N=%d' %N)
       
    print('Plotting...')
    plt.plot(x, y, '-k', label='Function')          # plot function and Gauss. quad. reconstruction
                                                    # for all N in N_order array
    for idx, yN_i in enumerate(yN[:,0]):
        
        N = N_order[idx]     
        plt.plot(x, yN[idx,:], '--', label='N=%d' %N)
    
    plt.legend()
    plt.savefig('%s_function' %title)
    plt.cla()

    for idx, yN_i in enumerate(yN[:,0]):  # plot function, Gauss. quad. and integral reconstruction 
                                          # for different N separately
        
        plt.title('Integral vs Gaussian quadrature coefficients N=%d' %N_order[idx])
        plt.ylim(-0.5, 1.5)
        plt.plot(x, y, '-k', label='Function')
        plt.plot(x, yN[idx,:], '--b', label='Gauss. quad.')
        plt.plot(x, yI[idx,:], '--r', label='Integral')
        plt.legend()
        plt.savefig('%s_gauss_vs_intgr_N%d' %(title, N_order[idx]))
        plt.cla()
        
    
    plt.title('%s function error behaviour' %title) # plot error vs N
    plt.xlim(np.amin(N_order)-1, np.amax(N_order)+10)
    plt.xlabel('N')
    plt.ylabel('Error')
    plt.grid(True, which='both')
    plt.loglog(N_order, error)

    if regress == True:
        log_error = np.log(error) 
        log_N = np.log(N_order)
        m = linregress(log_N, log_error)[0]
        plt.figtext( 0.6, 0.7 ,'slope = %.2f' %m)

    plt.savefig('%s_error' %title)
    plt.close()

#############################################


if __name__ == "__main__":
    
    start_time = time.time()

    sigma = 0.5/3.0
    mu = 0.0
    chi = lambda x: 1.0 if (x>=-0.5 and x<=0.5) else 0
    g = lambda x: exp(-(x-mu)**2/(2.0*(sigma**2))) 

    x = np.arange(-0.99, 0.99, 0.01) 
    N_order = np.array([4, 8, 16, 32])

    ConfigurePlots()
    Plot(x, chi, N_order, str('Top_hat'), True)
    Plot(x, g, N_order, str('Gaussian'), False)
    
    total_time = time.time() - start_time

    print('---------------------Run time: %.2f seconds -------------------------' %total_time)
