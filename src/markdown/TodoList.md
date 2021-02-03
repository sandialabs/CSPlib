# toDoList

### Documentations
* Introduction

* Building CSP

* Theoretical description of csp method. Index equation for ODE, exhausted mode equation. csp basis vector, eigen vector as basis vector.

* Examples for ode. H2 , CO, GRI, isoOctane.

* Application interface

### Examples
* tchem model : gri3.0, Host and exce
* general model: index, kernel: Davis–Skodje problem (without kokkos or tchem)
* tchem model, index, kernel: ODE, gri3.0, H2, isoctane

### GPU/CPU runs
* run code in GPU/CPU machine: white/Weaver: Done
* run code in CPU machine: Blake: works with g++,  icpc
* run code in mac laptop : clang, g++(gcc-mp-9 and g++-mp-9): Done

### eigen decomposition
* It is not ready to use in GPUs. It uses lapacke in cpu: function call it in model: can I make this call in model class ?

### Code development
* eigen c++?  do we need it ?
* compile csp without tchem and kokkos : Done

* delete files that depend on tchem2 and files that we do not use anymore: Done

* ODE example only host(cpu). First version of a template struct SpT(execution space) as template parameter. Numerical jacobian host interface has a compilation error. Second . only host and device class.    

* check index class: Done

* delete tools class: Done

* add numerical jacobian option for ODE: done

* clean kernel class: Partially done; M  is number of exhausted modes. What happen with eigenvalue is close or equal to zero ?: Numerical rank of jacobian can be used to mark when an eigenvalue is zero.

* csp helper: function to extract information, such as top reaction in importance and participation indexes; python version: make a list of functions.

* state vector TChem vs state vector csp

* list of examples

* example that does not depend on tchem: python example
<!-- * make a class a class template that depends of execution space (optional)   -->
## csp helper
* CSP data
a. M models
b. Numerical rank of jacobian  
c. Time scales (inverse of magnitud of eigenvalue)
d. slow index
e. fast index
f. importance index
g. csp pointer
h. source, state, smatrix, jacobian, and rate of progress

* getTopIndex: Done for one variable or mode

Works for fast, slow and importance indexes. Inputs are: Index (whole data base), Variable or mode index (IndVar), Top number of index (default value 4 ). Threshold; if the absolute value of the index is lower than the threshold value ( default value 1e-2), the value is consider equal to zero.

In each solution:
a. sort abs index
b. check if index past threshold
c. add index to a list
In whole data:
Two outputs:
Maximum  index:
e. get top index for each position
Top index
e. delete duplicate index  
f. select only top values  

* getCSPindex: done eval and get

get a specific (fast, slow, importance ) index for a variable or mode.

* make a list of reaction names with gas, surface reactions (both reverse and forward) and other process (conv or term that do not depend on the reactions )

* get variable names: get variable index.: Done


### csp analysis steps

----- Execution space -----

1. Compute : source terms, rate of progress, S matrix and Jacobian.

2. Compute: Eigenvalues and eigenvectors

---- Host space ----

3. Sort eigenvalues and eigenvaluesvectors.

4. Compute: exhausted modes and time scale.

5. Compute csp basis: right eigenvector and its inverse matrix.

6. Compute mode amplitud

7. csp indexes: participation, fast and slow indexes and csp pointer.    
