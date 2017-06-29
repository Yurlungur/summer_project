### intrgrand.py
### written by Joanna Piotrowska


import os, sys
import numpy as np
import scipy as sp
from numpy import pi, sqrt, exp, cos, sin
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

def Integrand(func, N, x_i):

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

    N_order = np.array([4, 8])
    func = np.frompyfunc(func, 1, 1)
    
    y = func(x)
    
    plt.plot(x, y, '-k', label='Function')
    
    for idx, N in enumerate(N_order):
        
        w_i = pi/(N+1)
        coeff = Decompose(func, N)
        yN = Tsch.chebval(x, coeff)
        
        plt.plot(x, yN, '--', label='N=%s col' % str(N))

        coeff2 = Integrand(func, N, x)
        yN2 = Tsch.chebval(x, coeff2)

        plt.plot(x, yN2, '-', label='N=%s integr' %str(N))


    plt.legend()

    plt.savefig('%s_function_integr' %title)

    
#############################################


if __name__ == "__main__":


    sigma = 0.5/3.0
    mu = 0.0
    chi = lambda x: 1.0 if (x>=-0.5 and x<=0.5) else 0
    g = lambda x: exp(-(x-mu)**2/(2.0*(sigma**2))) 

    x = np.arange(-0.99, 99, 0.01)
    
    ConfigurePlots()
    Plot(x, chi, 'Top_hat', False) 

