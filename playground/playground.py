### playground.py
### Introductory program for spectral decomposition of
### a Gauusian (g) and a top-hat (chi) function in the Chebyshev basis (T_i)
### comparing the error and coefficient behaviour with increasing
### number of basis functions (N) 
### written by Joanna Piotrowska


import os, sys
import numpy as np
import scipy as sp
from math import pi, sqrt, exp, cos, sin
from numpy.polynomial import chebyshev as Tsch
from scipy.integrate import quad
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sympy.utilities import lambdify

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

def Plot(x, func, title, log):

    fig, ax = plt.subplots()
    plt.xlim(-1,1)
    plt.ylim(-0.5, 1.5)
    plt.title('%s function and spectral interpolants' %title)

    N_order = np.array([4, 8, 10, 12, 16, 32, 50, 64, 128, 200, 250])
    error = np.empty(len(N_order))
    func = np.frompyfunc(func, 1, 1)
    
    y = func(x)
    
    plt.plot(x, y, '-k', label='Function')
    
    for idx, N in enumerate(N_order):
        
        w_i = pi/(N+1)
        coeff = Decompose(func, N)
        yN = Tsch.chebval(x, coeff)

        err = w_i * np.power(np.subtract(y,yN), 2)
        err = sqrt(np.sum(err))
        error[idx] = err
        
        plt.plot(x, yN, '--', label='N=%s' % str(N))

    plt.legend()

    plt.savefig('%s_function' %title)

    plt.clf()
    plt.title('%s function error behaviour' %title)
    plt.xlabel('N')
    plt.ylabel('Error')
    
    if log == True:
        plt.loglog(N_order, error)
        plt.loglog(N_order, error, 'ro', markersize=4)
        log_error = np.log(error) 
        log_N = np.log(N_order)
        m = linregress(log_N, log_error)[0]
        plt.figtext( 0.6, 0.7 ,'slope = %.2f' %m)

    else:
        plt.semilogy(N_order, error)
    
    plt.savefig('%s_error' %title)

    coeff = Decompose(func, np.amax(N_order))
    coeff_idx = np.arange(len(coeff))
    plt.clf()
    plt.ylabel('Coefficient')
    plt.xlabel('N')
    plt.title('%s Chebyshev coefficients' %title)
    plt.plot(coeff_idx, coeff)
    plt.savefig('%s_coeff' %title)


#############################################


if __name__ == "__main__":


    sigma = 0.5/3.0
    mu = 0.0
    chi = lambda x: 1.0 if (x>=-0.5 and x<=0.5) else 0
    g = lambda x: exp(-(x-mu)**2/(2.0*(sigma**2))) 

    x = np.arange(-1, 1, 0.01)
    
    ConfigurePlots()
    Plot(x, chi, str('Top_hat'), True)
    Plot(x, g, str('Gaussian'), False)
    
