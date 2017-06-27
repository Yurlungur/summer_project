### matrix.py
### short program for finding the w_i values of Chebyshev-Gauss weights
### given the colocation points
### written by Joanna Piotrowska


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
    matrix = np.array([], dtype='float').reshape((0, len(x_i)))
    
    for n in range(N+1):

        T_n = Tsch.Chebyshev.basis(n)
        T_n = np.frompyfunc(T_n, 1, 1)
        T_i = T_n(x_i)

        matrix = np.vstack((matrix, T_i))

    return matrix

#############################################

def MakeVector(N):

    w = lambda x: 1.0/sqrt(1.0-x**2)
    T = np.zeros(N+1)
    vector = np.zeros(len(T))
    T_0 = lambda x: 1.0
    T_1 = lambda x: x
    v_0 = lambda x: T_0(x) * w(x)
    v_1 = lambda x: T_1(x) * w(x)
    vector[0] = quad(w, -1.0, 1.0)[0]
    vector[1] = quad(v_1, -1.0, 1.0)[0]

    T_n = T_1
    T_minus = T_0 

    for n in range(2,N+1):

        T_plus = lambda x: 2*x*T_n(x) - T_minus(x)
        T_minus = T_n
        T_n = T_plus
        v_n = lambda x: T_n(x) * w(x)
        vector[n] = quad(v_n, -1.0, 1.0)[0]

    return vector

#############################################

if __name__ == "__main__":

    N = 4

    matrix = MakeMatrix(N)
    vector = MakeVector(N)
    print(vector)

    sys.exit()
