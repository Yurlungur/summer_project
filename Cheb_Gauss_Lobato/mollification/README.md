# Mollification attempts summary

All of the efforts presented below are based on a comprehensive paper by Eitan Tadmor - 'Filetrs, mollifiers and computation of the Gibbs phenomenon'. <br>
The mollifier used througout is the exponentially accurate mollifier in 11.20a, applied in real space, using scipy.integrate.quad. Relevant pieces of code are present in: <br>
* <b>spectral_tools.py</b> - contains spectral decomposition tools using Chebyshev-Gauss-Lobatto quadrature
* <b>edgedetect.py</b> - contains edge detection tools
* <b>mollification.py</b> - contains all mollification functions reffered to below <br><br>

The mollification road has taken many turns, my apologies for terrible time efficiency there - a route of wrong coding-up in discrete convolution led me nowhere, loosing precious hours.<br>
However, having moved to traditional quadrature integration I believe to have achieved partial success with the mollifiers. The final results can be divided into two sections: <b>Whole domain mollififcation</b> and <b>Piecewise mollification</b>.

# Whole domain mollification

<b>Functions:<br>
- MollifyQuad<br>
- MollifyQuadBuffer</b><br>

<h4><b>No boundary buffering applied</b></h4>

<img src="N32_nobuffer.png" align='middle'/>

For the whole domain mollification I straigtforwardly applied Tadmor recipe, performing a traditional convolution of the adaptive mollifying kernel with wiggly top-hat and a step (top-hat making its way out of the domain). The successes and failures are the following: <br>

<b>Successes:</b><br>
 * Reduced Gibbs oscillations, already at N=32. Wiggles are smoothed out significantly, the error falls off quickly away from the discontinuity (I haven't quantified 'quickly', I'm sorry!), reaching ~3x10^-4.<br>

<b>Failures:</b><br>
* Boundary is smoothed out, as if another discontinuity was present there. The top-hat leaving domain frame is significantly distorted.
* The discontinuity is lost in the representation, mollifier smooths out the jump too much.<br>

In the attempt to fix the first failure I extended the function outside of the [-1,1] domain with a 'buffer zone' by copying a mirror image of the function close to the domain boundary (figure below):<br>


  
