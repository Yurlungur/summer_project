# README.md for PI summer project
Collection of fragments of code and relevant image descriptions.

# Initial spectral decomposition

First step in developing the pseudospectral tool is projecting a smooth function (Gaussian) and a discontinuous one (Top Hat) onto the Chebyshev basis. playground.py performs this operation using both Gaussian quadrature and direct integration. The program outputs:
- function reconstruction both using integrals and Gaussian quadrature
- error vs number of modes for Gaussian quadrature

<i>Function reconstructions: </i>
<img src="/playground/Gaussian_function.png" alt='' width='400' align='middle'/><img src="/playground/Top_hat_function.png" alt='' width='400' align='middle'/>

As expected, there is a very quick convergence to the correct shape for the smooth Gaussian and prominent Gibbs phenomenon for the discontinuous Top Hat.



<i>Error behaviour</i>
<img src="/playground/Gaussian_error.png" alt='' width='400' align='middle'/><img src="/playground/Top_hat_error.png" alt='' width='400' align='middle'/>

As expected, the error decreases exponentially for the Gaussian, while it decreases as 1/N only for the Top Hat. The program performs linear regression analysis on log(N) and log(error), yielding a value close to -1 (i.e. 1/N behaviour). The unusual kink-like behaviour potentially comes from every 6th expansion coefficient having values close to 0.



<i>Gaussian quadrature vs integral calculation</i>
<img src="/playground/Gaussian_gauss_vs_intgr_N8.png" alt='' width='400' align='middle'/><img src="/playground/Gaussian_gauss_vs_intgr_N16.png" alt='' width='400' align='middle'/>
<img src="/playground/Top_hat_gauss_vs_intgr_N8.png" alt='' width='400' align='middle'/><img src="/playground/Top_hat_gauss_vs_intgr_N16.png" alt='' width='400' align='middle'/>

The difference between reconstruction for truncated sum approximation and integral calculation is visible, especially for low expansion orders. For a Gaussian both results quickly converge to original function, while for the Top Hat they remain offset, even for N=64 (largest N tried out).

 

