# README.md for PI summer project
Collection of fragments of code and relevant image descriptions.

# Initial spectral decomposition

<b>Folder: playground <br>
Code: playground.py</b>

First step in developing the pseudospectral tool was projecting a smooth function (Gaussian) and a discontinuous one (Top Hat) onto the Chebyshev basis. playground.py performs this operation using both Gaussian quadrature and direct integration. The program outputs:
- function reconstruction both using integrals and Gaussian quadrature
- error vs number of modes for Gaussian quadrature

<b><i>Function reconstructions: </i></b>

<img src="/playground/Gaussian_function.png" alt='' width='400' align='middle'/><img src="/playground/Top_hat_function.png" alt='' width='400' align='middle'/>

As expected, there is a very quick convergence to the correct shape for the smooth Gaussian and prominent Gibbs phenomenon for the discontinuous Top Hat.



<b><i>Error behaviour</i></b>

<img src="/playground/Gaussian_error.png" alt='' width='400' align='middle'/><img src="/playground/Top_hat_error.png" alt='' width='400' align='middle'/>

As expected, the error decreases exponentially for the Gaussian, while it decreases as 1/N only for the Top Hat. The program performs linear regression analysis on log(N) and log(error), yielding a value close to -1 (i.e. 1/N behaviour). The unusual kink-like behaviour potentially comes from every 6th expansion coefficient having values close to 0.



<b><i>Gaussian quadrature vs integral calculation</i></b>

<img src="/playground/Gaussian_gauss_vs_intgr_N8.png" alt='' width='400' align='middle'/><img src="/playground/Gaussian_gauss_vs_intgr_N16.png" alt='' width='400' align='middle'/>
<img src="/playground/Top_hat_gauss_vs_intgr_N8.png" alt='' width='400' align='middle'/><img src="/playground/Top_hat_gauss_vs_intgr_N16.png" alt='' width='400' align='middle'/>

The difference between reconstruction for truncated sum approximation and integral calculation is visible, especially for low expansion orders. For a Gaussian both results quickly converge to original function, while for the Top Hat they remain offset (and do not converge), even for N=64 (largest N tried out).

# Exponential filtering

<b>Folder: filtering <br>
Code: filter.py</b>

The next step was introducing an exponential filter, which reduces the significance of higher order Chebyshev polynomials in the expansion. Filter was applied by multiplying every ith coefficient by a factor sigma(i/N) where N is the highest order of polynomials used in given expansion. The chosen sigma was sigma=exp(-c(eta^s)) where eta=(i/N) and c, s are free parameters.
c=36 was chosen, such that sigma(eta=1) = 10e-15 (i.e machine precision) and s was varied when experimenting with filtering.

<b><i>Sigma and error</i></b>

   
<img src="/filtering/sigma.png" alt='' width='275' align='middle'/><img src="/filtering/Gaussian_error_vs_s.png" alt='' width='275' align='middle'/><img src="/filtering/Top_hat_error_vs_s.png" alt='' width='275' align='middle'/>

The shape of sigma filter is presented on the left, with general slope steepening with increasing s, as well as an overall shift of the 'turnover point' towards higher N with increasing s. The Gaussian error drops to 0 with increasing s, while it reaches a constant non-zero value for the Top Hat, which is a confirmation of earlier results, which show that for N=32 the Gaussian error is already close to 0, while it is not the case for the Top Hat. Both graphs show that minimisation of error as a function of s is not a correct way of probing the s parameter space in order to find the best degree of smoothing.

<b><i>Influence of s value on reconstructed waveform</i></b>

<b>Folders: filtering/64 and filtering/32</b>

<b>TOP HAT:</b>

<img src="/filtering/32/Top_hat_filter_s4_N32.png" alt='' width='400' align='middle'/><img src="/filtering/32/Top_hat_filter_s6_N32.png" alt='' width='400' align='middle'/>
<img src="/filtering/64/Top_hat_filter_s4_N64.png" alt='' width='400' align='middle'/><img src="/filtering/64/Top_hat_filter_s6_N64.png" alt='' width='400' align='middle'/>

s=4, 6, 8 values in the 'eyeballing' comparison seem to represent the waveform well enough without over-smoothing both for N=32 and N=64

<b>GAUSSIAN:</b>

<img src="/filtering/32/Gaussian_filter_s4_N32.png" alt='' width='400' align='middle'/><img src="/filtering/32/Gaussian_filter_s6_N32.png" alt='' width='400' align='middle'/>
<img src="/filtering/64/Gaussian_filter_s4_N64.png" alt='' width='400' align='middle'/><img src="/filtering/64/Gaussian_filter_s6_N64.png" alt='' width='400' align='middle'/>

General comment under coefficients.

<b><i>Influence of s value on coefficient amplitude</i></b>


<b>TOP HAT:</b>

<img src="/filtering/32/Top_hat_filter_coeff_s4_N32.png" alt='' width='400' align='middle'/><img src="/filtering/32/Top_hat_filter_coeff_s6_N32.png" alt='' width='400' align='middle'/>
<img src="/filtering/64/Top_hat_filter_coeff_s4_N64.png" alt='' width='400' align='middle'/><img src="/filtering/64/Top_hat_filter_coeff_s6_N64.png" alt='' width='400' align='middle'/>

For N=32 s=4 significantly influences coefficients with N>20, while for s=8 same applies for N>26. s=6 influence is between those.
For N=64 s=4 significantly influences coefficients with N>~36, while for s=8 same applies for N>50. s=6 influence is, again, between those

<b>GAUSSIAN:</b>

<img src="/filtering/32/Gaussian_filter_coeff_s4_N32.png" alt='' width='400' align='middle'/><img src="/filtering/32/Gaussian_filter_coeff_s6_N32.png" alt='' width='400' align='middle'/>
<img src="/filtering/64/Gaussian_filter_coeff_s4_N64.png" alt='' width='400' align='middle'/><img src="/filtering/64/Gaussian_filter_coeff_s6_N64.png" alt='' width='400' align='middle'/>

I don't know what to make of the Gaussian. I don't think filtering influences the result a lot (with the exception of extreme smoothing like s=2 or s=4)


# Spectral decomposition using Vandermonde matrix

<b>Folders: filtering/using_vandermonde/32 and filtering/using_vandermonde/64 <br>
Code: filtering.py and spectral_tools.py</b>

Following Jonah's advice, the spectral decomposition method was vectorised, to avoid long for-loop calculations. These can be replaced with single calculation of the Vandermonde matrix (and its inverse) to allow switching between the collocation points' function values u(x<sub>j</sub>) and expansion coefficients a<sub>i</sub>. <br>

The Vendermonde matrix is defined as follows: <br><br>

M<sub>ij</sub>=&phi;<sub>i</sub>(x<sub>j</sub>)w<sub>j</sub>/(&phi;<sub>i</sub>,&phi;<sub>i</sub>) <br><br>

and acts on the u<sub>j</sub> i.e. u(x<sub>j</sub>) with x<sub>j</sub> being the collocation points, in the following manner: <br><br>

M<sub>ij</sub>u<sub>j</sub> = a<sub>i</sub><br><br>

The results seem consistent with the for-loop calculations for N=32 and N=64 in the top-hat case which can be seen in the magnitude of the coefficients presented below. Left column: For-loop, Right column: Vandermonde matrix <br>

<img src="/filtering/32/Top_hat_filter_coeff_s4_N32.png" alt='' width='400' align='middle'/><img src="/filtering/using_vandermonde/32/Top_hat_filter_coeff_s4_N32.png" alt='' width='400' align='middle'/>
<img src="/filtering/64/Top_hat_filter_coeff_s4_N64.png" alt='' width='400' align='middle'/><img src="/filtering/using_vandermonde/64/Top_hat_filter_coeff_s4_N64.png" alt='' width='400' align='middle'/>
<br>

However, it seems not to be the case for N=64 spectral decomposition of Gaussian. Again, left column: for-loop, Right column: Vandermonde matrix <br>

<img src="/filtering/32/Gaussian_filter_coeff_s4_N32.png" alt='' width='400' align='middle'/><img src="/filtering/using_vandermonde/32/Gaussian_filter_coeff_s4_N32.png" alt='' width='400' align='middle'/>
<img src="/filtering/64/Gaussian_filter_coeff_s4_N64.png" alt='' width='400' align='middle'/><img src="/filtering/using_vandermonde/64/Gaussian_filter_coeff_s4_N64.png" alt='' width='400' align='middle'/>
<br>




