# Application Programming Interface

A typical CSPlib analysis involves the following steps:

***Model class or interface***
1.1 Compute : source terms or RHS.
1.2 Compute : Jacobian of RHS.
1.3 Compute : Rate of progress.
1.4 Compute : S matrix.

***Kernel class***
2.1 Compute : Eigenvalues and eigenvectors
2.2 Sort    : Eigenvalues and eigenvaluesvectors.
2.3 Set     : Matrix whose columns are the CSP basis vectors (right eigenvectors of Jacobian), and its inverse matrix.
2.4 Compute : Amplitudes of modes.  
2.5 Compute : Time scales.
2.6 Compute : Number of exhausted modes.
2.7 Compute : Compute CSP pointers.

***Index class***
3.1 Compute : Participation indices.
3.2 Compute : Slow Importance indices.
3.3 Compute : Fast Importance indices.

## Model Class

The model class is responsible for computing the source term (RHS) of the system and its Jacobian matrix. Additionally, if we have a chemical kinetic model, the model class will compute the S matrix and the reaction rates of progress. We have two available model classes, the general ODE class (gODE), and the TChem model class.  

### General ODE Class (gODE)

The general ode class (``CSP_REPOSITORY_PATH/src/core/gODE.cpp``) can handle any ODE system. This class requires a function for RHS and the Jacobian matrix.  

For example for the Davis-Skodje problem [Davis-Skodje 1999](https://aip.scitation.org/doi/10.1063/1.479372), [Valorani 2005](https://www.sciencedirect.com/science/article/pii/S0021999105001981). The RHS and jacobinan functions are:

```cpp
int rhs_Davis_Skodje(const std::vector<double>& state, std::vector<double>& source){
    const double epsilon = 0.01;
    const double y = state[0];
    const double z = state[1];
    source[0] = (-y+z/(1.+z))/epsilon - z/(1.+z)/(1.+z);
    source[1] = -z;
  return(0);
}

int jac_Davis_Skodje(const std::vector<double>& state, std::vector<std::vector<double>>& jac, int flag){
  const double epsilon = 0.01;
  const double y = state[0];
  const double z = state[1];

  jac[0][0] = -1./epsilon;
  jac[1][0] = 0;
  jac[0][1] = 2. * z / std::pow( z + 1. , 3.) - 1. / std::pow( z + 1., 2) +
              ( - z / std::pow( z + 1. , 2.) + 1. / ( z + 1. ) ) / epsilon;
  jac[1][1] = -1;
  return(0);
}
```
We pass these two functions to the gODE class .

```cpp
/// Constructor takes two functions.
GeneralODE mDavis_Skodje(
  std::function<int(const std::vector<double>&, std::vector<double>&)> (std::move(rhs_Davis_Skodje)),
  std::function<int(const std::vector<double>&, std::vector<std::vector<double>>&, int)> (std::move(jac_Davis_Skodje))
);
```  
To evaluate the RHS and Jacobian we do the following:

```cpp
//set state vector
mDavis_Skodje.setStateVector(state);
//eval rhs
mDavis_Skodje.evalSourceVector();
//get g
mDavis_Skodje.getSourceVector(source);
//eval Jacobian
mDavis_Skodje.evalJacMatrix(flag);
// get Jacobian
mDavis_Skodje.getJacMatrix(jac);
```

### TChem Model Class
The TChem model class (``CSP_REPOSITORY_PATH/src/core/chem_elem_ODE_TChem.cpp``) computes the source term, the Jacobian matrix, the rate of progress, and the S matrix for an elementary step chemical kinetic reaction mechanism. This class is a collection of function calls to TChem. TChem is designed and implemented with the Kokkos library. Thus, these computations can be performed in CPUs: (``Kokkos::DefaultHostExecutionSpace``) or in GPUs:  (``Kokkos::DefaultExecutionSpace``).The default execution space is configured as OpenMP or Cuda upon its availability. The default host execution spaces is always configured as OpenMP.  Besides, this class performs the computation in a batched mode.  


To create an instance of this class, we use:
```cpp
/// Constructor takes two input files.
///   [in] mech_gas_file - Chemkin reaction mechanism file
///   [in] thermo_gas_file - Thermo file
ChemElemODETChem( const std::string &mech_gas_file     ,
                  const std::string &thermo_gas_file   )                
```

The Chemkin-input files contain all the parameters of the reaction mechanism.

We can use the TChem model class to read an entire solution from the TChem homogenous batch reactors. This reactor resolves gas temperature and mass fraction in a transient batch reactor.

```cpp
//
///   [in] filename - database filename
///   [out] varnames - vector with variable names from TChem solution
 ChemElemODETChem::readIgnitionZeroDDataBaseFromFile(const std::string &filename,
                            std::vector<std::string> &varnames) ;
```

If we choose to run the computation on the device (GPU), the ``readIgnitionZeroDDataBaseFromFile`` function will copy and move the data to the GPUs. Alternatively, if we want to run the computation on the host space (CPU), we use the function call ``model.run_on_host(true)``.

```cpp
[in] run_on_host: true-run on host space , false -run on execution space
ChemElemODETChem::run_on_host(const bool & run_on_host)
```

If we do not have a TChem's database, we need to pass our database to the model class with the following function.

```cpp
/// [in] state_db: database for CSP analysis
ChemElemODETChem::setStateVectorDB(std::vector<std::vector <double> >& state_db)
```
The database is a 2D std::vector where the rows are the solution for each time. The columns correspond to the "time or iteration, density [kg/m$^3$], pressure [Pascal], Temperature[K], mass fractions".

With the uploaded database, the following function calls compute the source therms, the Jacobian matrix, the S matrix, and the rate of progress.

```cpp
ChemElemODETChem::evalSourceVector();
/// [in] useNumJac: 0 use analytical Jacobian, 1 use numerical jacabian
ChemElemODETChem::evalJacMatrix(unsigned int useJacAnl);
ChemElemODETChem::evalSmatrix();
ChemElemODETChem::evalRoP();
```

The TChem model class copies the computed data to the host space.  To obtain the data from this class, we use the following functions:

```cpp
/// [out] state_db :  state vector for the whole database  
ChemElemODETChem::getStateVector(std::vector<std::vector <double> >& state_db);

/// [out] source_db: source vector for the whole database
ChemElemODETChem::getSourceVector(std::vector<std::vector <double> >& source_db);

// [out] jac_db : Jacobian matrix for the whole database
ChemElemODETChem::getJacMatrix(std::vector <std::vector
                    <std::vector <double> > >& jac_db);

// [out] RoP: rate of progress for the whole database
ChemElemODETChem::getRoP(std::vector<std::vector <double> >& RoP);

// [out] Smatrixdb: S matrix for whole database
ChemElemODETChem::getSmatrix(std::vector < std::vector
                        <std::vector <double> > >& Smatrixdb);

```

The state and source vectors have a size of $N = N_s +1$, involving temperature and mass fractions, the size of the Jacobian matrix is $N \times N$, the size of S matrix and rate of progress vector is $N \times 2N_r$ and $2N_r$ respectively. The rate of progress vector includes the forward and reverse rate of progress.  

This class has additional functions to help post-process the CSP data.

```cpp
/// [out] return the number of species  
ChemElemODETChem::NumOfSpecies()
/// [out] return the number of reactions
ChemElemODETChem::NumOfReactions()
/// [out] spec_name: name of species in the reaction mechanism
ChemElemODETChem::getSpeciesNames(std::vector<std::string>& spec_name)
/// [in] var_name: variable name, use "Temperature" for temperature
///[out] return index of the variable in the csp analysis.
ChemElemODETChem::getVarIndex(const std::string & var_name)
/// [out] return number of variables in the csp analysis
ChemElemODETChem::getNumOfVariables()
/// [out] return  number of elements
ChemElemODETChem::getNumOfElements()
```

## Kernel Class

The second group of steps are implemented in the kernel class (``CSP_REPOSITORY_PATH/src/core/kernel.cpp``). This class computes the eigendecomposition for the Jacobian matrix, the time scales $\tau=\frac{1}{|\lambda|}$ , the number of exhausted model ($M$), the $\bm{a}$ and $\bm{b}$ CSP basis vectors, the amplitude of the modes $f_i$ and the CSP pointers.

We initialize this class with the number of variables, the g (source) vector and the Jacobian matrix(jac).  

```cpp
/// Constructor takes four inputs.
///   [in] nvars - number of state variable
///   [in] state_vec - y vector of state vector
///   [in] source_vec - g vector or rhs vector
///   [in] Jmat - Jacobian matrix of g
Kernel(int nvars,
       std::vector<double> &state_vec,
       std::vector<double> &source_vec,
       std::vector< std::vector<double> > &Jmat
      )
```
This class calculates the eigendecomposition of the Jacobian matrix. Next, It sorts the eigenvalues in descending order with respect to their magnitudes. With the sorted eigenvalues and eigenvectors, it sets the right eigenvectors as the $\bm{a}$ CSP basis vectors, and form the matrix $\bm{A}$ whose columns are the $\bm{a}$ vectors. The matrix $\bm{B}$, whose rows are the $\bm{b}$ vectors, is the inverse of $\mathbf{A}$ (see equation~\ref{eq:b}). The matrix inversion is done by Tines.


```cpp
/// Computation of eigendecomposition
/// This function does not have inputs. The jacobian matrix is a private member of the class.
Kernel::evalEigenValVec();

// sort eigenvalues in descending order, we use new order to sort eigenvectors as well.
Kernel::sortEigValVec();

//Set CSP basis vectors.
Kernel::setCSPVec(); // A = eig_vec_R and B = A^{-1}
//get CSP basis vector csp_vec_R(a) csp_vec_L(b).
Kernel::getCSPVec(csp_vec_L, csp_vec_R);
```

The time scales are computed as $\tau=\frac{1}{|\lambda|}$, where $\lambda$ is the magnitude of the eigenvalues. The amplitude of the mode $f_i$ is computed with [equation [2]](#cspbasicconcepts).

```cpp
// compute time scale.
Kernel::evalTau();
/// [out]  tauvec - time scales
Kernel::getTau(std::vector<double> &tauvec);
// compute the magnitude of the modes.
Kernel::evalModalAmp( );
/// [out] fvec - magnitud of the modes
Kernel::getModalAmp(std::vector<double> &fvec );
```

The number of exhausted modes $M$ is computed using relative and absolute tolerances (see [equation [5]](#cspbasicconcepts)) and a state vector. The tolerances are inputs of the analysis. The value of $M$ cannot be bigger than $N- N_{\mathrm{elements}} - 1$, or the number of eigenvalues with negative real component.

```cpp
/// [in] csp_rtolvar - relative tolerance for CSP analysis.
/// [in] csp_atolvar - absolute tolerance for CSP analysis.
Kernel::setCSPerr(double csp_rtolvar, double csp_atolvar);
/// [in] nel- number of elements in the reaction mechanism or system  
Kernel::evalM(const int &nElem);
/// [out] Number of exhausted M.
Kernel::getM(int &NofDM);
```

The CSP pointers ([equation [7]](#cspindices)) for all modes is computed by:
```cpp
Kernel::evalCSPPointers();
```
To obtain the CSP pointer data from the kernel class we use:
```cpp
/// [out] cspp_ij - csp pointers; row
Kernel::getCSPPointers( std::vector<std::vector<double>> &cspp_ij );
```
We can also use the function :
```cpp
/// [in] modeIndx - mode element position
/// [out] cspp_k - CSP pointer position for mode with element position modeIndx
Kernel::evalAndGetCSPPointers(const int & modeIndx, std::vector<double> &cspp_k)
```
To compute the CSP pointers for one mode.

At this point, the kernel class has computed all CSP data for a basic ODE system. Among these data, the time scales ($\tau$), the amplitude of the modes ($f$), the CSP basis vectors $\bm{a}$ and $\bm{b}$, the eigenvalues and eigenvectors of the system, the number of exhausted modes $M$, and the CSP pointers.

Additionally, the kernel class has diagnostic tools to test if the CSP data is not corrupted by numerical error.

The numerical rank of the Jacobian is used to check how many of the eigenvalues are reliably computed. The number of valid eigenvalues is equal to the numerical rank. Thus, if a Jacobian is not full rank, the smallest eigenvalues are essentially numerical noise.

```cpp
///[out] return the numerical rank of the Jacobian matrix
Kernel::computeJacobianNumericalRank()
```

We check the eigensolution only for the valid eigenvalues, according to the numerical rank of the Jacobian.

```cpp
//If a residual bigger than 1e-6 is obtained. " ---- High residual --- " will print out.
Kernel::DiagEigValVec();
```
We also check the orthonormality condition for the CSP basis vector.  

```cpp
// If a residual bigger than 1e-10 is obtained.
//": --- Orthogonality test failed: .." will print out.
Kernel::DiagOrthogonalityCSPVec();
```



### Eigen Solver With TINES

CSPlib has four different interfaces to Tines' eigensolver depending on the execution spaces and the input type. The first interface performs the eigensolution on the GPUs (CUDA, device execution space), and the inputs are in Kokkos-view format allocated in the GPUs memory space. The second interface carries out the computation on the CPUs (OPENMP, host execution spaces) and the inputs also in Kokkos-view format. The third interface uses the GPUs with the inputs in 3D std::vector format. Finally, in the fourth interface, the computation is performed in CPUs, and the inputs are in 3D std::vector format.  

The input of these interfaces is a database of Jacobians. The outputs are the real and imaginary part of the eigenvalues and the right eigenvectors for the whole database in 3D std vectors format.  

The function to call the GPU's interface with Kokkos-view type is the following:
```cpp
///   [in] jac - database of Jacobians - data is allocated  on the device
///   [out] eig_val_real_bath - real part eigenvalues of database
///   [out] eig_val_imag - imaginary part eigenvalues of database
///   [out] eig_vec_R - right eigenvectors of database
EigenSolver::evalDevice(const value_type_3d_view& jac,
                        std::vector<std::vector <value_type> >& eig_val_real,
                        std::vector<std::vector <value_type> >& eig_val_imag,
                        std::vector < std::vector<std::vector <value_type> > >& eig_vec_R);
```

The function to call the CPU's interface with Kokkos-view:

```cpp
///   [in] jac - database of Jacobians - data exists on the host
///   [out] eig_val_real_bath - real part eigenvalues of database
///   [in] eig_val_imag - imaginary part eigenvalues of database
///   [in] eig_vec_R - right eigenvectors of database
EigenSolver::evalHost(const value_type_3d_view_host& jac,
                        std::vector<std::vector <value_type> >& eig_val_real,
                        std::vector<std::vector <value_type> >& eig_val_imag,
                        std::vector < std::vector<std::vector <value_type> > >& eig_vec_R);
```

The function to call the GPU's interface with 3D std::vectors:

```cpp
///   [in] jac - database of Jacobians
///   [out] eig_val_real_bath - real part eigenvalues of database
///   [out] eig_val_imag - imaginary part eigenvalues of database
///   [out] eig_vec_R - right eigenvectors of database
EigenSolver::evalDevice(const std::vector < std::vector<std::vector <value_type> > >& jac,
                        std::vector<std::vector <value_type> >& eig_val_real,
                        std::vector<std::vector <value_type> >& eig_val_imag,
                        std::vector < std::vector<std::vector <value_type> > >& eig_vec_R);
```

The function to call the CPU's interface with 3D std::vectors:

```cpp
///   [in] jac - database of Jacobians - data exists on the host
///   [out] eig_val_real_bath - real part eigenvalues of database
///   [in] eig_val_imag - imaginary part eigenvalues of database
///   [in] eig_vec_R - right eigenvectors of database
EigenSolver::evalHost(const std::vector < std::vector<std::vector <value_type> > >& jac,
                      std::vector<std::vector <value_type> >& eig_val_real,
                      std::vector<std::vector <value_type> >& eig_val_imag,
                      std::vector < std::vector<std::vector <value_type> > >& eig_vec_R);
```

## Index Class
To instantiate the index class, we need nine inputs from the model and kernel classes.

```cpp
/// Constructor takes eight inputs.
///   [in] Nreac - number of reactions
///   [in] Nvar - number of variables
///   [in]  M - number of exhausted modes
///   [in] eig_val_real - eigenvalues real part
///   [in] eig_val_imag - eigenvalues imaginary part
///   [in] A - a csp basis vector
///   [in] B - b csp basis vector
///   [in] Smat - S matrix
///   [in] RoP  - rate of progress
CSPIndex(
      int Nreac,
      int Nvar,
      int M,
      std::vector<double> &eig_val_real,
      std::vector<double> &eig_val_imag,
      std::vector<std::vector<double> > &A,
      std::vector<std::vector<double> > &B,
      std::vector<std::vector<double> > &Smat,
      std::vector<double> &RoP
     )
```

The following functions compute the Participation indices ([equation [11]](#cspindices)), and the slow and fast Importance indices ([equations [9] and [10]](#cspindices)) for all variables and modes for one state vector.   

```cpp
CSPIndex::evalParticipationIndex();
CSPIndex::evalImportanceIndexSlow();
CSPIndex::evalImportanceIndexFast();
```
To obtain the data produced by the above function:
```cpp
/// [out] P_ik - Participation index; rows: modes, columns: rate of progress
CSPIndex::getParticipationIndex (std::vector<std::vector<double> > &P_ik  );
/// [out] Islow_jk - Slow importance index; rows: variable, columns: rate of progress
CSPIndex::getImportanceIndexSlow( std::vector<std::vector<double> > &Islow_jk );
/// [out] Ifast_jk- Fast importance index;  rows: variables, columns: rate of progress
CSPIndex::getImportanceIndexFast(std::vector<std::vector<double> > &Ifast_jk );
```

Sometimes, one only wants to compute the index for a few modes/variables. In this case, one can use the following functions:

```cpp
/// [in] modeIndx - index (position) for mode
/// [out] P_k - Participation index for mode with indx modeIndx
CSPIndex::evalAndGetParticipationIndex(const int &modeIndx, std::vector<double> &P_k);
/// [in] varIndx - index (position) for variable
/// [out] Islow_k - Slow importance index for variable with index varIndx
CSPIndex::evalAndGetImportanceIndexSlow(const int & varIndx, std::vector<double> &Islow_k);
/// [in] varIndx - index (position) for variable
/// [out] Ifast_k -  Fast importance index for variable with index varIndx
CSPIndex::evalAndGetImportanceIndexFast(const int & varIndx, std::vector<double> &Ifast_k);
```

The ``CSPIndex::getTopIndex`` function returns an std::vector<int> with the reaction number (in the rate of progress vector) for the highest absolute value Participation and slow/fast Importance indices.

```cpp
/// [in] Index - Participation/slow/fast index for one mode or variable
/// [in] Top - only add top absolute values.
/// [in] threshold- only add values bigger than this threshold value.
/// [in/out] IndxList- list of reaction number in the RoP(rate of progress) vector.
CSPIndex::getTopIndex(std::vector<double> &Index,
                const int & Top, const double & threshold,
                std::vector<int> & IndxList );
```
For example, to find out which reactions have the highest contribution in the fastest mode, one can use this function and pass the participation index for mode 0. This participation index (std::vector) is obtained with the function ``CSPIndex::evalAndGetParticipationIndex``  with ``modeIndx=0``. Alternatively, one can use the ``CSPIndex::evalParticipationIndex`` function, and get the Participation indices for all modes with ``CSPIndex::getParticipationIndex``. The output of the function ``CSPIndex::getTopIndex`` is ``IndxList``, which is a vector containing  the reaction number in the rate of progress vector.
