# Introduction

CSPlib is an open source software library for analyzing general ordinary differential equation (ODE) systems and detailed chemical kinetic ODE systems. It relies on the computational singular perturbation (CSP) method for the analysis of these systems. The software provides support for

* General ODE models (gODE model class) for computing source terms and jacobians for a generic ODE system.

* TChem model (ChemElemODETChem model class) for computing source term, Jacobian, other necessary chemical reaction data, as well as the rates of progress for a homogenous batch reactor using an elementary step detailed chemical kinetic reaction mechanism. This class relies on the [TChem](https://github.com/sandialabs/TChem#homogenousbatchreactors)  library.

* A set of functions to compute essential elements of CSP analysis (Kernel class). This includes computations of the eigensolution of the Jacobian matrix, CSP basis vectors and co-vectors, time scales (reciprocals of the magnitudes of the Jacobian eigenvalues), mode amplitudes, CSP pointers, and the number of exhausted modes. This class relies on the Tines library.

* A set of functions to compute the eigensolution of the Jacobian matrix using the Tines library GPU eigensolver.

* A set of functions to compute CSP indices (Index Class). This includes participation indices and both slow and fast importance indices.  

## Citing

* Oscar Diaz-Ibarra, Kyungjoo Kim, Cosmin Safta, and Habib Najm, CSPlib - A Software Toolkit for the Analysis of Dynamical Systems and Chemical Kinetic Models, Sandia National Laboratories, SAND 2020-XXXXX, 2020.*

## Nomenclature

Symbol|Description
--|--
$\textbf{y}$  |  State vector   
$\textbf{g}$ |  Source vector
$t$ |  Time    
$\textbf{a}_i$ |  CSP basis vector  
$\textbf{b}^i$ |  CSP basis co-vector
$f^i$  |  Mode amplitude  
$J_{ij}$ |  Jacobian matrix of the ODE right hand side (RHS)  
$g_{\mathrm{fast}}$ |  ODE RHS component in the fast subspace
$g_{\mathrm{slow}}$ |  ODE RHS component in the slow subspace  
$M$ |  Number of fast exhausted modes  
$\delta y^{i}_{\mathrm{error}}$ |  Error for variable $i$  
$\mathrm{tol}_{\mathrm{relative}}$ |  Relative error tolerance   
$\mathrm{tol}_{\mathrm{absolute}}$ |  Absolute error tolerance  
$\tau$ |  Time scale    
$\lambda$ |  Eigenvalues of Jacobian matrix  
$N_{\mathrm{spec}}$ |  Number of species  
$N_{\mathrm{reac}}$ |  Number of reactions  
$N_{\mathrm{var}}$ | Number of variables
$S$ | S matrix
$\mathcal{R}_r$ |  Rate of progress or reaction $r$
 RoP | Rate of progess
 CSPpointer$_{ij}$ |  CSP pointer for mode $i$ with respect to variable $j$
$(I^i_r)_{\mathrm{slow}} $|  Slow importance index of reaction $r$ for variable $i$  
$(I^i_r)_{\mathrm{fast}}$ | Fast importance index of reaction $r$ for variable $i$  
$P^i_r$ | Participation index of reaction $r$ for mode $i$ 
