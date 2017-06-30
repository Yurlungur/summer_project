# README.md for PI summer project
Collection of fragments of code and relevant image descriptions.

# Initial spectral decomposition

Folder: playground
Code: playground.py

First step in developing the pseudospectral tool was projecting a smooth function (Gaussian) and a discontinuous one (Top Hat) onto the Chebyshev basis. playground.py performs this operation using both Gaussian quadrature and direct integration. The program outputs:
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

The difference between reconstruction for truncated sum approximation and integral calculation is visible, especially for low expansion orders. For a Gaussian both results quickly converge to original function, while for the Top Hat they remain offset (and do not converge), even for N=64 (largest N tried out).

# Exponential filtering

Folder: filtering
Code: filter.py

The next step was introducing an exponential filter, which reduces the significance of higher order Chebyshev polynomials in the expansion. Filter was applied by multiplying every ith coefficient by a factor sigma(i/N) where N is the highest order of polynomials used in given expansion. The chosen sigma was sigma=exp(-c(eta^s)) where eta=(i/N) and c, s are free parameters.
c=36 was chosen, such that sigma(eta=1) = 10e-15 (i.e machine precision) and s was varied when experimenting with filtering.

<i>Sigma and error</i>

   
<img src="/filtering/sigma.png" alt='' width='275' align='middle'/><img src="/filtering/Gaussian_error_vs_s.png" alt='' width='275' align='middle'/><img src="/filtering/Top_hat_error_vs_s.png" alt='' width='275' align='middle'/>

The shape of sigma filter is presented on the left, with general slope steepening with increasing s, as well as an overall shift of the 'turnover point' towards higher N with increasing s. The Gaussian error drops to 0 with increasing s, while it reaches a constant non-zero value for the Top Hat, which is a confirmation of earlier results, which show that for N=32 the Gaussian error is already close to 0, while it is not the case for the Top Hat. Both graphs show that minimisation of error as a function of s is not a correct way of probing the s parameter space in order to find the best degree of smoothing.

<i>Influence of s value on reconstructed waveform</i>

Folders: filtering/64 and filtering/32

TOP HAT:

<img src="/filtering/32/Top_hat_filter_s4_N32.png" alt='' width='400' align='middle'/><img src="/filtering/32/Top_hat_filter_s6_N32.png" alt='' width='400' align='middle'/>
<img src="/filtering/64/Top_hat_filter_s4_N64.png" alt='' width='400' align='middle'/><img src="/filtering/64/Top_hat_filter_s6_N64.png" alt='' width='400' align='middle'/>

s=4, 6, 8 values in the 'eyeballing' comparison seem to represent the waveform well enough without over-smoothing both for N=32 and N=64

GAUSSIAN:

<img src="/filtering/32/Gaussian_filter_s4_N32.png" alt='' width='400' align='middle'/><img src="/filtering/32/Gaussian_filter_s6_N32.png" alt='' width='400' align='middle'/>
<img src="/filtering/64/Gaussian_filter_s4_N64.png" alt='' width='400' align='middle'/><img src="/filtering/64/Gaussian_filter_s6_N64.png" alt='' width='400' align='middle'/>

For the Gaussian s=8 filter already converges onto original function for N=32, while for N=64 already at s=6.






