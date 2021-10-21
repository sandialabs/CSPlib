/* =====================================================================================
CSPlib version 1.1.0
Copyright (2021) NTESS
https://github.com/sandialabs/csplib

Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
certain rights in this software.

This file is part of CSPlib. CSPlib is open-source software: you can redistribute it
and/or modify it under the terms of BSD 2-Clause License
(https://opensource.org/licenses/BSD-2-Clause). A copy of the license is also
provided under the main directory

Questions? Contact Habib Najm at <hnnajm@sandia.gov>, or
           Kyungjoo Kim at <kyukim@sandia.gov>, or
           Oscar Diaz-Ibarra at <odiazib@sandia.gov>

Sandia National Laboratories, Livermore, CA, USA
===================================================================================== */


#include "kernel.hpp"
#include "Tines.hpp"


#define sign(x) x==0.0 ? 0.0 : x/fabs(x)

//#define DEBUG

double MatNorm(char norm, // '1' or 'O'= 1-norm; 'I'= Infinity-norm
             int nrow, int ncol, int lda_in,
             std::vector<double> &mat_in)
{
 double anorm;
 switch (norm)
 {
   case '1' : case 'O' : {
     std::cout<< "( 1-norm )"<<"\n";
     std::vector<double> sum_of_col(nrow,0.0);

     for (size_t j=0; j<ncol; j++) {
       for (size_t i=0; i<nrow; i++) {
         sum_of_col[j] += fabs(mat_in[i*ncol+j]);
       }

       if (j==0) {
         anorm = sum_of_col[j];
       } else if ( j>0 && sum_of_col[j] > anorm ) {
         anorm = sum_of_col[j];
       }

     }
     break;
   }
   case 'I' : {
     std::cout<< "( Infinity-norm )"<<"\n";
     std::vector<double> sum_of_row(ncol,0.0);

     for (size_t i=0; i<nrow; i++) {
       for (size_t j=0; j<ncol; j++) {
         sum_of_row[i] += fabs(mat_in[i*ncol+j]);
       }
       //std::cout<< "sum_of_row[i]= "<< sum_of_row[i] <<"\n";
       if (i==0) {
         anorm = sum_of_row[i];
       } else if ( i>0 &&  sum_of_row[i] > anorm ) {
         anorm = sum_of_row[i];
       }

     }
     break;
   }
   default :
     std::cout<<__func__<< ":\n"
                        <<"\t Please select norm =    \n"
                        <<"\t\t 1 or O for 1-norm; or \n"
                        <<"\t\t I for Infinity-norm.  \n";
 }

 return anorm;

}
//-----------------------------------------------------------//
void fill_with_given_vector( int m, double *A,
    std::vector< double > &vec_in )
    {

#ifdef DEBUG
  std::cout<<"kernel.cpp: Start copying vector in fill_with_given_vector" <<"\n";
#endif

  for (int i=0; i < m; ++i) {
    A[i] =  vec_in[i];
  }

#ifdef DEBUG
  std::cout<<"kernel.cpp: Finished copying matrix in fill_with_given_vector" <<"\n";
#endif

}


//-----------------------------------------------------------//
void fill_with_given_matrix(
    int m, int n, double *A, int lda,
    std::vector< std::vector<double> > &matrix_in
    )
{
    //#define A(i_, j_) A[ (i_) + (j_)*lda ] // original with Magma
    #define A(i_, j_) A[ (i_)*lda + (j_) ]

#ifdef DEBUG
    std::cout<<"kernel.cpp: Start copying matrix in fill_with_given_matrix" <<"\n";
#endif

    int i, j;
    for (j=0; j < n; ++j) {
        for (i=0; i < m; ++i) {
            A(i,j) =  matrix_in[i][j] ;
        }
    }
#ifdef DEBUG
    std::cout<<"kernel.cpp: Finished copying matrix in fill_with_given_matrix" <<"\n";
#endif

    #undef A
}
//-----------------------------------------------------------//
/**
  Evaluate eigen values and vectors, and
  store in the private data storge of Kernel class
*/
//-----------------------------------------------------------//

void Kernel::Initialize( const int nvars,
       const std::vector<double> &state_vec,
       const std::vector<double> &source_vec,
       const std::vector< std::vector<double> > &Jmat)
     {
      _nvars = nvars;
      _state_vec = state_vec;
      _source_vec = source_vec;
      _Jmat   = Jmat;
      _nmodes = nvars;

     }


int Kernel::evalEigenValVec()
{

{
  _eig_val_real = std::vector<double>(_nvars);
  _eig_val_imag = std::vector<double>(_nvars);
  _eig_vec_L = std::vector<double>(_nvars*_nvars);
  _eig_vec_R = std::vector<double>(_nvars*_nvars);

#if defined(CSP_ENABLE_VERBOSE)
  std::cout<<"Tools::EigenSolve - just got in EigenSolve" <<"\n";
#endif

  /// repack A to 1D
  std::vector<double> h_A_vec(_nvars*_nvars);
  for (int j=0; j < _nvars; ++j) {
      for (int i=0; i < _nvars; ++i)
          h_A_vec[i*_nvars + j] =  _Jmat[i][j] ;
  }

#if defined(CSP_ENABLE_VERBOSE)
  std::cout<<"tools.cpp: fill_with_given_matrix done------" <<"\n";

  struct timeval begin1, end1;
  gettimeofday( &begin1, NULL );
  std::cout<<"tools.cpp: calling regular Tines TPL Eigendecomposition \n";
#endif

  int info(0);
  {
    const int m = _nvars;
    info = Tines::SolveEigenvaluesNonSymmetricProblem_HostTPL(m,
					     &h_A_vec[0], m, 1,
					     &_eig_val_real[0], &_eig_val_imag[0],
					     &_eig_vec_L[0], m, 1,
					     &_eig_vec_R[0], m, 1);
  }

#if defined(CSP_ENABLE_VERBOSE)
  gettimeofday( &end1, NULL );
  double wall_time = 1.0 * ( end1.tv_sec - begin1.tv_sec ) +
                     1.0e-6 * ( end1.tv_usec - begin1.tv_usec );

  std::cout<<"Eigensolver Tines TPL Eigendecomposition info="<< info <<"\n";
  printf("Wall time for regular Tines TPL Eigendecomposition  =  %7.2f\n", wall_time);
#endif

  if (info != 0) {
    printf("Tines::Eigendecomposition_HostTPL returned error %lld\n",  (long long) info);
    return 1;
  }
  //}


#if defined(CSP_ENABLE_VERBOSE)
   printf("eigen values (real) = \n");
   for (int i=0; i< array_size1; i++) {
     printf("%+5.2e \t" , wr[i]);
   }
   printf("\n");

   printf("eigen values (imag) = \n");
   for (int i=0; i< array_size2; i++) {
     printf("%+5.2e \t" , wi[i]);
   }
   printf("\n");

   printf("eigenvector (left ) arrray size = %6d\n", array_size3);
   printf("eigenvector (right) arrray size = %6d\n", array_size4);
#endif


}
  const int nmodes = _eig_val_real.size();
  if( nmodes != _nvars ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<"\n"
             << "Number fo eigen modes are different than the number of state variables.\n";
    std::cout<< "  _nmodes= "<< nmodes<<std::endl;
    std::cout<< "  _nvars = "<< _nvars<<std::endl;
    exit(1);
  }

return(0);
}

//-----------------------------------------------------------//
/*
  Copy eigen values and vectors to user provided memory space
  from the private database of Kernel.
*/
//-----------------------------------------------------------//
int Kernel::getEigenValVec(
    std::vector<double> &eig_val_real, std::vector<double> &eig_val_imag,
    std::vector<double> &eig_vec_L, std::vector<double> &eig_vec_R )
{

  eig_val_real = _eig_val_real ;
  eig_val_imag = _eig_val_imag ;
  eig_vec_L    = _eig_vec_L    ;
  eig_vec_R    = _eig_vec_R    ;

  return 0;
}

int Kernel::getEigenValVec(
    std::vector<double> &eig_val_real, std::vector<double> &eig_val_imag,
    std::vector<double> &eig_vec_R )
{

  eig_val_real = _eig_val_real ;
  eig_val_imag = _eig_val_imag ;
  eig_vec_R    = _eig_vec_R    ;

  return 0;
}

//-----------------------------------------------------------//
/*
  Setting arbritrary values provided by the user
  for eigen values and vectors
*/
//-----------------------------------------------------------//
void Kernel::setEigenValVec( //
    std::vector<double> &eig_val_real, std::vector<double> &eig_val_imag,
    std::vector<double> &eig_vec_L , std::vector<double> &eig_vec_R )
{

  _eig_val_real = eig_val_real ;
  _eig_val_imag = eig_val_imag ;
  _eig_vec_L    = eig_vec_L    ;
  _eig_vec_R    = eig_vec_R    ;

}

void Kernel::setEigenValVec( //
    std::vector<double> &eig_val_real, std::vector<double> &eig_val_imag,
    std::vector<double> &eig_vec_R )
{

  _eig_val_real = eig_val_real ;
  _eig_val_imag = eig_val_imag ;
  _eig_vec_R    = eig_vec_R    ;

}

//-----------------------------------------------------------//
//  Setting arbritrary basis vector provided by the user
//  for csp analysis
int Kernel::setCSPVec(
    std::vector<double> &csp_vec_L ,
    std::vector<double> &csp_vec_R )
{

  _csp_vec_L = csp_vec_L ;
  _csp_vec_R = csp_vec_R ;

  return 0;
}

void Kernel::evalAndGetgfast(std::vector<double> & gfast)
{

  if( _fvec.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":\n"
             << "\t Modal Amplitude vector is empty.\n"
             << "\t Run Kernel::evalModalAmp.\n";
  exit(1);
  }

  if( _csp_vec_R.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"<< "CSP vector(right) is empty.\n"
             <<"Call Kernel::setCSPVec.\n";
  exit(1);
  }

  for (size_t i = 0; i < _nvars; i++) {
    double sum = 0;
    for (size_t k = 0; k < _NofDM; k++) {
      sum += _csp_vec_R[i*_nvars + k]*_fvec[k];
    }
    gfast[i] = sum;
  }

}
//-----------------------------------------------------------//
//  Setting right eigenvector and its inverse as basis vectors
//  for csp analysis
int Kernel::setCSPVec()
{

#if defined(CSP_ENABLE_VERBOSE)
  std::cout<<"Setting right eigenvector and its inverse as basis vectors for csp analysis"<<std::endl;
#endif

  if( _eig_vec_R.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<"\n"
             << "  Eigen vector(right) is empty.\n"
             << "  Run Kernel::evalEigenValVec.\n";
    exit(1);
  }

  // Getting Left vector by inverting Right vectors:
  int info(0);
  const int m = _nvars;
  std::vector<double> matA = _eig_vec_R;
  std::vector<double> matB(matA.size());
  std::vector<int> ipiv(m);
#if defined(CSP_ENABLE_VERBOSE)
  printf("\n --> Tines InvertMatrix :\n");
#endif
  {
    info = Tines::InvertMatrix_HostTPL(m,
				       &matA[0], m, 1,
				       &ipiv[0],
				       &matB[0], m, 1);
  }
  if (info != 0) {
    printf("    Tines InvertMatrix returned error with info = %d\n", info);
  }

  _csp_vec_L = matB; // setting inverted-eigen-vector-right as csp-basis-left
  _csp_vec_R = _eig_vec_R;  // setting eigen-vector-right as csp-basis-right

  return 0;
}

//-----------------------------------------------------------//
//  Getting the CSP basis vectors already setup
int Kernel::getCSPVec(
    std::vector<double> &csp_vec_L ,
    std::vector<double> &csp_vec_R ) {

  if( _csp_vec_L.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"<< "CSP vector(left) is empty.\n"
             <<"Call Kernel::setCSPVec.\n";
  exit(1);
  }

  if( _csp_vec_R.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"<< "CSP vector(right) is empty.\n"
             <<"Call Kernel::setCSPVec.\n";
  exit(1);
  }

  csp_vec_L = _csp_vec_L ;
  csp_vec_R = _csp_vec_R ;

  return 0;
}

//-----------------------------------------------------------//
int Kernel::setCSPerr(double csp_rtolvar, double csp_atolvar) {

  _csp_rtolvar = csp_rtolvar;
  _csp_atolvar = csp_atolvar;
  return 0;
}

//-----------------------------------------------------------//
void Kernel::ComputeErrVec(
    int np,                   // no. of variables
    std::vector<double>& w,   // state vector
    std::vector<double>& ewt, // error vector (rhs)
    bool scalar,
    double TolRel,
    double TolAbs) {

  double rtol, atol;

//Set Accuracy:
  if (scalar) {
    // printf("using relative error %e absolute error %e\n",TolRel, TolAbs );
//Scalar Error Control
    for (int i=0; i<np; i++) {
      ewt[i] = TolRel * fabs(w[i]) + TolAbs;
    }
  } else {
//Vector Error Control

    for (int i=0; i<np; i++) {

      if (w[i] >= 1.0e-06) {
        rtol=TolRel;
        atol=TolAbs;
      }

      if (w[i] < 1.0e-06) {
        rtol=(double)1.0;
        atol=TolAbs;
      }

      ewt[i] = rtol * fabs(w[i]) + atol;
    }

  }
  return;

}


int Kernel::getErrVec( std::vector<double> &Errorvec)
{

  if( !_errvec.empty() ) {
    Errorvec = _errvec;
  } else {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":\n"
             << "\t Error vector is empty.\n"
             << "\t Run Kernel::evalM.\n";
  exit(1);
  }

  return 0;
}

void Kernel::evalM_WoExp(const int &nElem)
{
  // Maximun number of exhausted modes:
  // nElem number of elements constrains, conservation of mass
  // 1  conservation of enthalpy for adiabatic systems
  int MaxExM(_nvars);

  if (nElem > 0) {
    MaxExM = _nvars - nElem -1 ;
  }

  // negative eigenvalues: exhausted modes cannot be bigger than number of eigen values
  int no_of_neg_eig_val_real=0;
  for (int k=0; k<_eig_val_real.size(); k++ ) {
    if (_eig_val_real[k] < 0.0) ++no_of_neg_eig_val_real;
  }

  MaxExM = std::min( MaxExM, no_of_neg_eig_val_real );

  // convert from 1D to 2D(row-major form.)
  std::vector<std::vector<double>> a(_nvars,std::vector<double>(_nvars));
  for (size_t i=0; i<_nvars*_nvars; ++i)
    a[i/_nvars][i%_nvars] = _eig_vec_R[i];

  _errvec = std::vector<double>(_nvars);
  ComputeErrVec(_nvars, _state_vec, _errvec, true , _csp_rtolvar, _csp_atolvar);
  int exM(0);
  for ( int M = 0; M < _nvars; M++) { // searching M in each variable
    // M is element position
    //check if next eigenvalue is a complex conjugate
    // we do not want to have an exhausted model between two equal eigen values
    // with an equal real part
    exM = M;
    if (M < _nvars-1) {
      if  (_eig_val_real[M]==_eig_val_real[M+1] ) {
        // printf("complex eigen value exM %d, %e _eig_val_real[M-1], %e _eig_val_real[M] \n",exM, _eig_val_real[M-1], _eig_val_real[M] );
        M+=1;
        // if I am in the last model quit, because there is not \tau[_nvars+1]
        if (M==_nvars){
          _NofDM = std::min(_nvars , MaxExM);
          return;
        }
      }
    }

    const int Mp1 = std::min(M+1,_nvars-1);
    for (int i = 0; i < _nvars; i++) { // loop over variables
      double deltaYfast = 0;
      for (size_t r = 0; r < M+1  ; r++) {
        deltaYfast += _fvec[r] * a[i][r];
      }

      if (std::fabs(deltaYfast) * _tauvec[Mp1] > _errvec[i]) {
        //check elements
        _NofDM = std::min(exM, MaxExM);
        _varM = i;
        // printf("New M %d var %d\n",_NofDM, _varM );
        return;
      }

    }

  }

  //check elements
  _NofDM = std::min(_nvars , MaxExM);

  return;


}

void Kernel::evalM(const int &nElem)
{
  // Maximun number of exhausted modes:
  // nElem number of elements constrains, conservation of mass
  // 1  conservation of enthalpy for adiabatic systems
  int MaxExM(_nvars);

  if (nElem > 0) {
    MaxExM = _nvars - nElem -1 ;
  }

  // negative eigenvalues: exhausted modes cannot be bigger than number of eigen values
  int no_of_neg_eig_val_real=0;
  for (int k=0; k<_eig_val_real.size(); k++ ) {
    if (_eig_val_real[k] < 0.0) ++no_of_neg_eig_val_real;
  }

  MaxExM = std::min( MaxExM, no_of_neg_eig_val_real );

  // convert from 1D to 2D(row-major form.)
  std::vector<std::vector<double>> a(_nvars,std::vector<double>(_nvars));
  for (size_t i=0; i<_nvars*_nvars; ++i)
    a[i/_nvars][i%_nvars] = _eig_vec_R[i];

  _errvec = std::vector<double>(_nvars);
  ComputeErrVec(_nvars, _state_vec, _errvec, true , _csp_rtolvar, _csp_atolvar);
  int exM(0);
  for ( int M = 0; M < _nvars; M++) { // searching M in each variable
    // M is element position
    //check if next eigenvalue is a complex conjugate
    // we do not want to have an exhausted model between two equal eigen values
    // with an equal real part
    exM = M;
    if (M < _nvars-1) {
      if  (_eig_val_real[M]==_eig_val_real[M+1] ) {
        // printf("complex eigen value exM %d, %e _eig_val_real[M-1], %e _eig_val_real[M] \n",exM, _eig_val_real[M-1], _eig_val_real[M] );
        M+=1;
        // if I am in the last model quit, because there is not \tau[_nvars+1]
        if (M==_nvars){
          _NofDM = std::min(_nvars , MaxExM);
          return;
        }
      }
    }

    const int Mp1 = std::min(M+1,_nvars-1);
    for (int i = 0; i < _nvars; i++) { // loop over variables
      double deltaYfast = 0;
      for (size_t r = 0; r < M+1  ; r++) {
        deltaYfast += _fvec[r]*a[i][r]*
          (-1.+std::exp(_eig_val_real[r]*_tauvec[Mp1]))/_eig_val_real[r];
      }

      if (std::fabs(deltaYfast) > _errvec[i]) {
        //check elements
        _NofDM = std::min(exM, MaxExM);
        _varM = i;
        // printf("New M %d var %d\n",_NofDM, _varM );
        return;
      }

    }

  }

  //check elements
  _NofDM = std::min(_nvars , MaxExM);

  return;


}

void Kernel::getvarM(int varM)
{
  varM =_varM;
}



//-----------------------------------------------------------//
int Kernel::getM( int &NofDM ) {

  NofDM = _NofDM;

  return 0;
}

//===========================================================//
//-----------------------------------------------------------//
/*
  Modal Amplitudes : fvec=B.g = eig_vec_L * rhs
*/
//-----------------------------------------------------------//

//-----------------------------------------------------------//
int Kernel::evalModalAmp( )
{

  if( _csp_vec_L.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":\n"
             << "\t CSP vector(left) is empty.\n"
             << "\t Call Kernel::setCSPVec.\n";
  exit(1);
  }

  _fvec = std::vector<double>(_nvars);

  //Multiplies two matrices (double-precision).
  {
    const int trans_tag = Tines::Trans::NoTranspose::tag;
    const int m = _nvars;
    const double one(1), zero(0);
    Tines::Gemv_HostTPL(trans_tag,
			m, m,
			one,
			&_csp_vec_L[0], m, 1,
			&_source_vec[0], 1,
			zero,
			&_fvec[0], 1);
  }

#if defined(CSP_TEST_INDEX_PRINT)
  Util::Print::vec<double>("From GetModalAmp:: _csp_vec_L", RM, O2d, Dbl , _nvars, _nvars, _csp_vec_L);
  Util::Print::vec<double>("From GetModalAmp:: _source_vec", Col, O1d, Dbl, _nvars, _source_vec);
  Util::Print::vec<double>("From GetModalAmp:: _fvec", Col, O1d, Dbl, _nvars, _fvec);
#endif

  return 0;
}

//-----------------------------------------------------------//
int Kernel::getModalAmp( std::vector<double> &fvec)
{

  if( !_fvec.empty() ) {
    fvec = _fvec;
  } else {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":\n"
             << "\t Modal Amplitude vector is empty.\n"
             << "\t Run Kernel::evalModalAmp.\n";
  exit(1);
  }

  return 0;
}

//===========================================================//
/*
Time scales from eigenvalues
*/
//-----------------------------------------------------------//
int Kernel::evalTau()
{


  _tauvec = std::vector<double>(_nvars);
  double  evamp;
  for ( int i=0; i<_nvars; i++ ) {
        _tauvec[i]=(double)1.0;     //
        //evamp=dsqrt( eval(i,1)*eval(i,1) + eval(i,2)*eval(i,2) )
        evamp = sqrt( _eig_val_real[i] * _eig_val_real[i]
                     + _eig_val_imag[i] * _eig_val_imag[i] );
        if (evamp != (double)0.0) _tauvec[i] = (double)1.0/evamp;
  }


  return 0;
}


//-----------------------------------------------------------//
int Kernel::getTau(std::vector<double> &tauvec) {

  tauvec = _tauvec;
  return 0;
}

//===========================================================//
//-----------------------------------------------------------//
// Converting orthogonal-complex basis to orthogonal-real basis sets
// obtained by the prescription of 2006_Circadian paper:
int Kernel::ComplexToOrthoReal(
    std::vector<double> &csp_vec_R_out,
    std::vector<double> &csp_vec_L_out)
    {

    int nmode = _nvars;
    int nvar  = _nvars;

  //Note: _eig_vec_L is composed of row vectors and
  //      _eig_vec_R is composed of col vectors
  for ( int j=0; j<(nmode-1); j++ ) {

    if ( _eig_val_real[j] == _eig_val_real[j+1] ) {
      for ( int i=0; i<nvar; i++ ) {
        csp_vec_R_out[i*nmode+j]     =  _eig_vec_R[i*nmode+j];
        csp_vec_R_out[i*nmode+(j+1)] = -_eig_vec_R[i*nmode+(j+1)];

        csp_vec_L_out[j*nvar+i]      = 2.0*_eig_vec_L[j*nvar+i];
        csp_vec_L_out[(j+1)*nvar+i]  = 2.0*_eig_vec_L[(j+1)*nvar+i];
      }
    } else {
      for ( int i=0; i<nvar; i++ ) {
        csp_vec_R_out[i*nmode+j]     = _eig_vec_R[i*nmode+j];
        csp_vec_R_out[i*nmode+(j+1)] = _eig_vec_R[i*nmode+(j+1)];

        csp_vec_L_out[j*nvar+i]      = _eig_vec_L[j*nvar+i];
        csp_vec_L_out[(j+1)*nvar+i]  = _eig_vec_L[(j+1)*nvar+i];
      }
    }

  }

  return 0;
} // end of ComplexToOrthoReal

//===========================================================//

int Kernel::sortEigValVec()
{
  if( _eig_val_real.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":\n"
             << "\t _eig_val_real is empty.\n";
  exit(1);
  }

  if( _eig_val_imag.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":\n"
             << "\t _eig_val_imag is empty.\n";
  exit(1);
  }

  if( _eig_vec_R.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":\n"
             << "\t _eig_vec_R is empty.\n";
  exit(1);
  }

  std::vector<double> eig_val_real_out;
  std::vector<double> eig_val_imag_out;
  std::vector<double> eig_vec_R_out;
  // std::vector<double> eig_vec_L_out;

  // std::cout<< "_nvars, _nmodes = "<< _nvars << ", "<< _nmodes<<"\n";

  std::vector<double> eig_val_mod;
  for (int i=0; i<_eig_val_real.size(); i++)
      eig_val_mod.push_back(sqrt( pow(_eig_val_real[i],2.0)
                                + pow(_eig_val_imag[i],2.0) )  );

  std::vector<double> eig_val_mod_with_real_sign;
  for (int i=0; i<_eig_val_real.size(); i++) {
      eig_val_mod_with_real_sign.push_back( sign(_eig_val_real[i]) * eig_val_mod[i] );
  }

  std::vector< std::pair <int,double> > pair_vec;

  // sort using a custom function object
  struct {
    bool operator()(const std::pair<int,double> &a,
                   const std::pair<int,double> &b) {
      return (a.second < b.second);
    }
  } sortinrev;
  struct {
    bool operator()(const std::pair<int,double> &a,
                   const std::pair<int,double> &b) {
      return (a.second > b.second);
    }
  } sortinfor;

  // Entering values in vector of pairs

  bool sort_modulus = true;           // employ sorting by (signed-or-unsigned) modulus (T) ... or not (F)
  bool sort_signed_modulus = false;   // sort by signed modulus (T) or abs val of modulus (F)

  if (sort_modulus) {
    if (sort_signed_modulus){
      for (int i=0; i<_eig_val_real.size(); i++)
        pair_vec.push_back( std::make_pair(i, eig_val_mod_with_real_sign[i]) );
      std::sort(pair_vec.begin(), pair_vec.end(), sortinrev);
    }
    else{
      for (int i=0; i<_eig_val_real.size(); i++)
        pair_vec.push_back( std::make_pair(i, eig_val_mod[i]) );
      std::sort(pair_vec.begin(), pair_vec.end(), sortinfor);
    }
  } else {
    for (int i=0; i<_eig_val_real.size(); i++)
          pair_vec.push_back( std::make_pair(i, _eig_val_real[i]) );
  }

  const int nvar  = _nvars;
  const int nmode = _nvars;

  // Packed in row-major form:
  for (int i=0; i<nmode; i++) {
    int old_index = pair_vec[i].first;
    eig_val_real_out.push_back(_eig_val_real[old_index]);
    eig_val_imag_out.push_back(_eig_val_imag[old_index]);
  }

  // Packing sorted Right eigenvector matrix (in row-major form)
  for (int j=0; j<nvar; j++) {
    for (int i=0; i<nmode; i++) {
      int old_index = pair_vec[i].first;
      eig_vec_R_out.push_back(_eig_vec_R[j*nmode + old_index]);
      //eig_vec_L_out.push_back(_eig_vec_L[j*ncol + old_index]);
    }
  }

  // Packing sorted Left eigenvector matrix (in row-major form)
  // for (int i=0; i<nmode; i++) {
  //   int old_index = pair_vec[i].first;
  //   for (int j=0; j<nvar; j++) {
  //
  //     //eig_vec_R_out.push_back(_eig_vec_R[j*ncol + old_index]);
  //     eig_vec_L_out.push_back(_eig_vec_L[j + old_index*nvar]);
  //   }
  // }

  // Resetting private eigenvalues and eigenvectors with sorted one:
  _eig_val_real = eig_val_real_out;
  _eig_val_imag = eig_val_imag_out;
  _eig_vec_R    = eig_vec_R_out;
  // _eig_vec_L    = eig_vec_L_out;

  return 0;
}



int Kernel::getTSRcoef(
    std::vector<double>& Wbar /*TSR coeff.*/) {

  if( _Wbar.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"
             << "TSR coefficients vector is empty. "
             << "Call Kernel::evalTSRcoef to fill out the vector.\n";
    exit(1);
  }
  Wbar = _Wbar;

  return 0;
}

//===========================================================//
// TSR : Valorani et al., Combustion and Flame 162 (2015) 2963
//       Eq. 30 and 31.
//-----------------------------------------------------------//

//-----------------------------------------------------------//
int Kernel::evalTSR(

)
{

  if( _csp_vec_R.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<": "
             << "CSP basis (right) vectors are not set.\n"
             << "Run Kernel::setCSPVec to populate it.\n";
    exit(1);
  }

  if( _fvec.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"
             << "Modal Amplitude vector is empty.\n"
             << "Run Kernel::evalModalAmp.\n";
    exit(1);
  }

  {

    /* Norm of a Vector */
    double g_norm = std::inner_product(&_source_vec[0], &_source_vec[0]+_nvars,
                                     &_source_vec[0], 0.0);
    g_norm = sqrt(g_norm);


    std::vector<double> g_dot_a(_nvars, 0.0);

    {
      const int trans_tag = Tines::Trans::Transpose::tag;
      const int m = _nvars;
      const double one(1), zero(0);
      Tines::Gemv_HostTPL(trans_tag,
			  m, m,
			  one,
			  &_csp_vec_R[0], m, 1,
			  &_source_vec[0], 1,
			  zero,
			  &g_dot_a[0], 1);
    }

    _Wbar = std::vector<double>(_nvars);
    double sum_W=0.0;
    for (int i=0; i<_nvars; i++ ) {
      _Wbar[i] = (_fvec[i] * g_dot_a[i]) / (g_norm * g_norm) ;
      sum_W += fabs(_Wbar[i]) ;
    }

    for (int i=0; i<_nvars; i++ ) {
      _Wbar[i] /= sum_W ;
    }
  }
  double w_tau_bar=0.0;

  for (int i=0; i<_nvars; i++ ) {

#ifdef DEBUG
    std::cout<< " _Wbar[i] = "<< _Wbar[i]<<"\n";
#endif
    w_tau_bar += _Wbar[i]
               //* (_eig_val_real[i]/fabs(_eig_val_real[i]))
               * sign(_eig_val_real[i])
               * sqrt(pow(_eig_val_real[i],2.0) + pow(_eig_val_imag[i],2.0));
  }

  _w_tau_bar=w_tau_bar;

#ifdef DEBUG
  std::cout<< " _w_tau_bar = "<< _w_tau_bar<<"\n";
#endif

  return 0;
}

int Kernel::getTSR( double &w_tau_bar ) {

  w_tau_bar = _w_tau_bar;
  return 0;
}

int Kernel::computeJacobianNumericalRank()
{

  int matrix_rank(0);

  // double jac1D[_nvars*_nvars];
  std::vector<double> jac1D(_nvars*_nvars);

  int count=0;
  for (size_t i=0; i<_nvars; i++) {
    for (size_t j=0; j<_nvars; j++) {
      jac1D[count] = _Jmat[i][j];
      count++;
    }
  }

  std::vector<double> tau(_nvars);
  std::vector<int> jpiv(_nvars);
  int m = _nvars, n=_nvars;
  // int lda = _nvars;

  Tines::QR_WithColumnPivoting_HostTPL(m, n,
				       &jac1D[0], n, 1,
				       &jpiv[0],
				       &tau[0],
				       matrix_rank);
  return  matrix_rank;
}

//===========================================================//
// Various diagnostic functions:
//-----------------------------------------------------------//

int Kernel::DiagEigValVec()
{

  if( _eig_val_real.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<"\n"
             << "  Eigen value is empty.\n"
             << "  Run Kernel::evalEigenValVec.\n";
    exit(1);
  }


  if( _eig_vec_R.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<"\n"
             << "  Right eigen vector is empty.\n"
             << "  Run Kernel::evalEigenValVec.\n";
    exit(1);
  }

  int nrow = _nvars;
  int ncol = _nvars;

  bool is_complex = false;

  std::vector<double> A_vec;
  std::vector<std::complex<double> > A_vec_cmplx;
  std::vector<std::complex<double> > AVR_vec_cmplx(nrow*ncol, std::complex<double>(0.0, 0.0) );
  std::vector<std::complex<double> > eig_vec_R_cmplx(nrow*ncol, std::complex<double>(0.0, 0.0) );

  CSP::construct_1D_from_2D<double>( _Jmat, A_vec);


#ifdef DEBUG
  Util::Print::mat<double>("_Jmat", RIF, Out2d, Dbl, nrow, ncol, _Jmat);
  Util::Print::vec<double>("A_vec",  RM, Out2d, Dbl, nrow, ncol, A_vec);
  Util::Print::vec<double>("_eig_val_real",  Row, O1d, Dbl, ncol, _eig_val_real);
  Util::Print::vec<double>("_eig_val_imag",  Row, O1d, Dbl, ncol, _eig_val_imag);
#endif

  // Creating a array of flag indicating real or complex eigenvalue:
  std::vector<bool> real_comlex_eigval_flag(ncol);

  for (int i=0; i<ncol; i++ ) {
    if (_eig_val_imag[i] == 0.0) {
       real_comlex_eigval_flag[i] = false;
    } else {
       real_comlex_eigval_flag[i] = true;
       is_complex = true;
    }
  }

#ifdef DEBUG
  for (size_t i=0; i<ncol; i++ ) {
    if (real_comlex_eigval_flag[i] ) {
      std::cout<<": real_comlex_eigval_flag[i] ->>" << "true" << "\n";
    } else {
      std::cout<<": real_comlex_eigval_flag[i] ->>" << "false" << "\n";
    }
  }
#endif

  //Converting real  "A_vec" to complex "A_vec_cmplx"
  for (size_t i=0; i<nrow*ncol; i++ ) {
    A_vec_cmplx.push_back(std::complex<double>(A_vec[i], 0.0));
  }

#ifdef DEBUG
  Util::Print::vec<std::complex<double> >("A_vec_cmplx", RM, O2d, Cmplx , nrow, ncol, A_vec_cmplx);
  Util::Print::vec<std::complex<double> >("A_vec_cmplx", Cont, O1d, Cmplx , nrow, ncol, A_vec_cmplx);
  Util::Print::vec<double>("_eig_vec_R", RM, O2d, Dbl , nrow, ncol, _eig_vec_R);
  Util::Print::vec<double>("_eig_vec_R", Cont, O1d, Dbl , nrow, ncol, _eig_vec_R);
#endif

  CSP::convert_Real_to_Complex_Vector(real_comlex_eigval_flag, _eig_vec_R, eig_vec_R_cmplx );

#ifdef DEBUG
  Util::Print::vec<std::complex<double> >("eig_vec_R_cmplx", RM, O2d, Cmplx , nrow, ncol, eig_vec_R_cmplx);
  Util::Print::vec<std::complex<double> >("eig_vec_R_cmplx", Cont, O1d, Cmplx , nrow, ncol, eig_vec_R_cmplx);
#endif


  {
    const std::complex<double> one = (1.0); // this initialization by constructor shouls work for both double and complex
    const std::complex<double> zero  = (0.0);

    const int trans_tag = Tines::Trans::NoTranspose::tag;
    const int m = _nvars;

    Tines::Gemm_HostTPL(trans_tag,trans_tag,
			m, m, m,
			one,
			&A_vec_cmplx[0], m, 1,
			&eig_vec_R_cmplx[0], m, 1,
			zero,
			&AVR_vec_cmplx[0], m, 1);
  }

  std::vector<std::complex<double> > eig_val_cmplx(ncol);
  for (size_t j=0; j<ncol; j++ ) {
    eig_val_cmplx[j] = std::complex<double>(_eig_val_real[j] , _eig_val_imag[j]);
  }

  // std::vector<std::vector<std::complex<double> > >  eig_val_cmplx_mat;
  //
  // CSP::tools::construct_diagonal_matrix<std::complex<double> >(ncol, eig_val_cmplx, eig_val_cmplx_mat);
  _high_residual_eigen =0;
  const double eps_1_2 = std::sqrt(std::numeric_limits<double>::epsilon());

  const int num_rank_jac = computeJacobianNumericalRank();

  std::cout<< "\n-- Evaluating residual (in-house):"<<std::endl;
  for (size_t i=0; i<nrow; i++ ) {
    // double meanAVR = 0;
    double maxAVR = eps_1_2;
    for (size_t k = 0; k < ncol; k++) {
      maxAVR = maxAVR > std::abs(AVR_vec_cmplx[i*ncol+k]) ?
      maxAVR : std::abs(AVR_vec_cmplx[i*ncol+k]);
      // meanAVR += std::abs(AVR_vec_cmplx[i*ncol+k]);
    }

    // only check the num_rank_jac fist eigen values, the rest eigen values
    // have numerical noise
    for (size_t j=0; j<ncol - num_rank_jac; j++ ) {
      //A*VR = lamnda * VR
      const std::complex<double> resR = eig_val_cmplx[j] * eig_vec_R_cmplx[i*ncol+j];
      const std::complex<double> resL = AVR_vec_cmplx[i*ncol+j];
      // check only if eigenvalue magnitud is bigger than epsilon
      // if  ( (std::abs(resR) > eps_1_2) || (std::abs(resL) > eps_1_2)  )
      if  ( std::abs(eig_val_cmplx[j] ) > eps_1_2 )
      {
       const std::complex<double> diff = (resL - resR);

       // if ( fabs(diff.real()/meanAVR  >= 1e-6) || fabs(diff.imag()/meanAVR  >= 1e-6) )
       if (std::abs(diff)/std::abs(eig_val_cmplx[j]) >= 1e-6)
       {
         std::cout<< __FILE__<<": "<<__LINE__<<": " << " ---- High residual --- ";
         std::cout<< diff.real() << "   +   " <<  diff.imag() << std::endl;
         std::cout<<"Magnitud (eig_val*VR): " << std::abs(resR) << " Magnitud (A*VR): " <<  std::abs(resR) << std::endl;
         const double val = std::abs(diff)/std::abs(eig_val_cmplx[j]) ;
         _high_residual_eigen = _high_residual_eigen > val? _high_residual_eigen:val;

       }
       }
      }

      }

  std::cout << "Eigen value check: No residual reported implies it's less than 1e-6" << std::endl;

  return 0;
}

double Kernel::getEigenResidual(){
     return _high_residual_eigen;
}

//-----------------------------------------------------------//
// Checking orthogonality between Left and Right CSP vectors:
int Kernel::DiagOrthogonalityCSPVec() {

  if( _csp_vec_L.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"<< "CSP vector(left) is empty.\n"
             <<"Run Kernel::setCSPVec.\n";
    exit(1);
  }

  if( _csp_vec_R.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"<< "CSP vector(right) is empty.\n"
             <<"Run Kernel::setCSPVec.\n";
    exit(1);
  }

  bool write_on_screen = false;
  bool write_in_file   = false;

#ifdef DEBUG
  write_on_screen = true;
  write_in_file   = true;
#endif

  std::vector<double> matC(_nvars*_nvars);

  {
    const double one(1), zero(0);

    const int trans_tag = Tines::Trans::NoTranspose::tag;
    const int m = _nvars;

    Tines::Gemm_HostTPL(trans_tag,trans_tag,
			m, m, m,
			one,
			&_csp_vec_L[0], m, 1,
			&_csp_vec_R[0], m, 1,
			zero,
			&matC[0], m, 1);
  }
  for ( int i=0; i<_nvars; i++ ) {
    for ( int j=0; j<_nvars; j++ ) {

      if ( i==j && std::fabs(matC[i*_nvars+j] - (double)1) > 1.e-10 ) {
        std::cout<<__func__ <<": --- Orthogonality test failed: diagonal element found = " << matC[i*_nvars+j] <<"\n";
        return 1;
      }

      if ( i!=j && std::fabs(matC[i*_nvars+j]) > 1.e-10 ) {
        std::cout<<__func__ <<": --- Orthogonality test failed: off-diagonal element found = " << matC[i*_nvars+j] <<"\n";
        return 1;
      }

    }
  }

  std::cout<<__func__ <<": --- Orthogonality test successful.\n";

  if (write_on_screen) {

    std::cout<<__func__ <<":\n\north_mat :\n";
    for ( int i=0; i<_nvars; i++ ) {
      for ( int j=0; j<_nvars; j++ ) {
        printf(" %5.3e\t", matC[i*_nvars+j]);
      }
      std::cout<<"\n";
    }
  }

  if (write_in_file) {
    FILE *fortho = fopen("diag_ortho.txt","w");

    fprintf(fortho, "%20s\n" , "Ortho_mat" );
    for ( int i=0; i<_nvars; i++ ) {
      for ( int j=0; j<_nvars; j++ ) {
        fprintf(fortho, " %5.3e\t", matC[i*_nvars+j]);
      }
      fprintf( fortho,"\n");
    }
    fclose(fortho);
  }


  return 0;
}


double Kernel::ConditionNumbersJacobian (
        char norm // '1' or 'O'= 1-norm; 'I'= Infinity-norm
         )
  {

  std::vector<double> jac1D;

  int count=0;
  for (size_t i=0; i<_nvars; i++) {
    for (size_t j=0; j<_nvars; j++) {
      jac1D.push_back(_Jmat[i][j]);
      count++;
    }
  }

  double cond_num(0);
  {
    const int m = _nvars;
    std::vector<int> ipiv(m);
    Tines::ComputeConditionNumber_HostTPL(m,
					  &jac1D[0], m, 1,
					  &ipiv[0],
					  cond_num);
  }
  return cond_num;
}

//-----------------------------------------------------------
//-----------------------------------------------------------
/* CSP Pointers */
void Kernel::evalCSPPointers() {

  if( _csp_vec_L.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"<< "CSP vector(left) is empty.\n"
             <<"Call Kernel::setCSPVec.\n";
  exit(1);
  }

  if( _csp_vec_R.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"<< "CSP vector(right) is empty.\n"
             <<"Call Kernel::setCSPVec.\n";
  exit(1);
  }

  _cspp_ij = std::vector<std::vector<double>>(_nvars, std::vector<double>(_nvars,0.0));

  for (int k=0; k<_nvars; k++)
    for (int i=0; i<_nvars; i++)
        _cspp_ij[k][i] = _csp_vec_R[i*_nvars + k ]*_csp_vec_L[i + k*_nvars];
        // _cspp_ij[k][i] = _A[i][k] * _B[k][i];




}
void Kernel::evalAndGetCSPPointersFastSubSpace(std::vector<double>& csp_pointer_fast_space)
{
  if (csp_pointer_fast_space.empty()) {
    csp_pointer_fast_space = std::vector<double>(_nvars,0.0);
  } else {
    for (size_t i = 0; i < _nvars; i++)
      csp_pointer_fast_space[i] = 0.0 ;
  }

  for (int i=0; i<_nvars; i++)
    for (int r=0; r<_NofDM; r++)
      csp_pointer_fast_space[i] += _csp_vec_R[i*_nvars + r ]*_csp_vec_L[i + r*_nvars];

}
//-----------------------------------------------------------
void Kernel::getCSPPointers( std::vector<std::vector<double>> &cspp_ij ) {

  if( _cspp_ij.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"
             << "_cspp_ij matrix is empty.\n"
             << "Call Kernel::evalCSPPointers to fill out the matrix.\n";
    exit(1);
  }

  cspp_ij = _cspp_ij;

}
//-----------------------------------------------------------
void Kernel::evalAndGetCSPPointers(const int & modeIndx,
   std::vector<double> &cspp_k) {

  //
  if( _csp_vec_L.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"<< "CSP vector(left) is empty.\n"
             <<"Call Kernel::setCSPVec.\n";
  exit(1);
  }

  if( _csp_vec_R.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"<< "CSP vector(right) is empty.\n"
             <<"Call Kernel::setCSPVec.\n";
  exit(1);
  }
  cspp_k = std::vector<double >(_nvars,0);
  for (int i=0; i<_nvars; i++)
    cspp_k[i] = _csp_vec_R[i*_nvars + modeIndx ]*_csp_vec_L[i + modeIndx*_nvars];
    // cspp_k[i] = _A[i][modeIndx] * _B[modeIndx][i];

}
