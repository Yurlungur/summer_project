### matrix.py
### short program for finding the w_i values of Chebyshev-Gauss weights
### given the colocation points
### written by Joanna Piotrowska

from __future__ import print_function

import os, sys
import numpy as np
import scipy as sp
from math import pi, sqrt, exp, cos, sin
from numpy.polynomial import chebyshev as Tsch
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib import rcParams


#############################################

def MakeMatrix(N):

    coll_pts = lambda i: cos((2*i+1)*pi/(2*N+2))
    coll_pts = np.frompyfunc(coll_pts, 1, 1)
    x_i = coll_pts(np.arange(N+1))
    matrix = np.empty((N+1, N+1))    
    
    for n in range(N+1):

        T_n = Tsch.Chebyshev.basis(n)
        T_i = T_n(x_i)

        matrix[n,:] = T_i 

    return matrix

#############################################

def MakeVector(N):

    vector = np.empty(N+1)
    w = Tsch.chebweight
    
    for n in range(N+1):
        
        T_n = Tsch.Chebyshev.basis(n)
        f_n = lambda x: T_n(x)*w(x)
        vector[n] = quad(f_n, -1.0, 1.0)[0]

    return vector

#############################################

if __name__ == "__main__":

    N = 4

    matrix = MakeMatrix(N)
    vector = MakeVector(N)

    matrix_inv = np.linalg.inv(matrix)

    w_i = np.matmul(matrix_inv, vector)
    
    print(w_i)

