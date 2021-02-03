/* =====================================================================================
CSPlib version 1.0
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


#ifndef KERNEL_CSP
#define KERNEL_CSP

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>
#include <sys/time.h>

#include <algorithm>
#include <functional>
#include <complex>

#include "tools.hpp"
#include "util.hpp"


class Kernel
{
 private:
  int _nvars;
  int _nmodes;

  std::vector<double> _state_vec;
  std::vector<double> _source_vec;
  std::vector< std::vector<double> > _Jmat;

  // std::vector<double> _eig_val_real , _eig_val_imag;
  std::vector<double> _eig_vec_L , _eig_vec_R;
  std::vector<double> _csp_vec_L , _csp_vec_R;
  std::vector<double> _eig_val_real , _eig_val_imag;

  std::vector<double> _tauvec;
  std::vector<double> _fvec; // Modal amplitudes

  double _csp_rtolvar; // relative error
  double _csp_atolvar; // absolute error
  std::vector<double> _errvec;
  int _NofDM;                // Number of exhausted modes
  int _varM; //variable where for M criteria

  std::vector<double> _Wbar; // TSR coeficients
  double _w_tau_bar;         // TSR
  double _high_residual_eigen;
  //csp pointer
  std::vector<std::vector<double> > _cspp_ij;

 public:


  Kernel(int nvars):_nvars(nvars){}

  Kernel(int nvars,
         std::vector<double> &state_vec,
         std::vector<double> &source_vec,
         std::vector< std::vector<double> > &Jmat
        ):
         _nvars(nvars),
         _state_vec(state_vec),
         _source_vec(source_vec),
         _Jmat(Jmat),
         _nmodes(nvars)
        {}

  void Initialize( const int nvars,
         const std::vector<double> &state_vec,
         const std::vector<double> &source_vec,
         const std::vector< std::vector<double> > &Jmat
       );

  //-----------------------------------------------------------//
  // int evalEigenCppValVec();


//-----------------------------------------------------------//
  int getEigenValVec(
      std::vector<double> &eig_val_real ,
      std::vector<double> &eig_val_imag,
      std::vector<double> &eig_vec_L ,
      std::vector<double> &eig_vec_R
  );

//
//-----------------------------------------------------------//
  int getEigenValVec(
      std::vector<double> &eig_val_real ,
      std::vector<double> &eig_val_imag,
      std::vector<double> &eig_vec_R
  );

//-----------------------------------------------------------//
void setEigenValVec(  // what is the use of it? Should be removed.
      std::vector<double> &eig_val_real,
      std::vector<double> &eig_val_imag,
      std::vector<double> &eig_vec_L,
      std::vector<double> &eig_vec_R
  );

//-----------------------------------------------------------//
void setEigenValVec(  // what is the use of it? Should be removed.
      std::vector<double> &eig_val_real,
      std::vector<double> &eig_val_imag,
      std::vector<double> &eig_vec_R
  );

//-----------------------------------------------------------//
//  Setting arbritrary basis vector provided by the user
//  for csp analysis
int setCSPVec(
      std::vector<double> &csp_vec_L ,
      std::vector<double> &csp_vec_R
  );
int getErrVec( std::vector<double> &Errorvec);

// void evalAndGetMvec(std::vector<int>& Mvec, const int &nElem);
//-----------------------------------------------------------//
//  Setting right eigenvector and its inverse as basis vectors
//  for csp analysis
int setCSPVec();

//-----------------------------------------------------------//
int getCSPVec(
      std::vector<double> &csp_vec_L ,
      std::vector<double> &csp_vec_R
  );


//-----------------------------------------------------------//

  int setCSPerr(
      double csp_rtolvar, // relative error
      double csp_atolvar  // absolute error
  );

//-----------------------------------------------------------//
  void ComputeErrVec(
      int np,
      std::vector<double>& w,
      std::vector<double>& ewt,
      bool scalar,
      double TolRel,
      double TolAbs
  );

//-----------------------------------------------------------//
void evalM(const int &nel);
void evalMwoExp(const int &nElem,  int &NexM);


//-----------------------------------------------------------//
  int getM(int &NofDM);


//-----------------------------------------------------------//
  int evalModalAmp();

//-----------------------------------------------------------//
  int getModalAmp(
      std::vector<double> &fvec
  );


//-----------------------------------------------------------//
  int evalTau(
  );

//-----------------------------------------------------------//
  int getTau(
      std::vector<double> &tau /*out*/ // Time scale
  );



//===========================================================//
//-----------------------------------------------------------//
  int ComplexToOrthoReal(
      //int nrow,
      //int ncol,
      //std::vector<double> &eig_val_in,
      //std::vector<double> &eig_vec_R_in,
      //std::vector<double> &eig_vec_L_in,
      std::vector<double> &csp_vec_R_out,
      std::vector<double> &csp_vec_L_out
  );



  int sortEigValVec();


//===========================================================//
// TSR : Valorani et al., Combustion and Flame 162 (2015) 2963
//-----------------------------------------------------------//

int getTSRcoef(
    std::vector<double>& Wbar /* TSR coeff. */
);

int evalTSR(
    //std::vector<double>& csp_vec_R
);

int getTSR(
    double &w_tau_bar /* TSR */
);


//===========================================================//
// Various diagnostic functions:
//-----------------------------------------------------------//

  // interface 2
  int DiagEigValVec(
  );


//-----------------------------------------------------------//
// Checking orthogonality between Left and Right CSP vectors:
  int DiagOrthogonalityCSPVec();


  int evalEigenValVec();

  double ConditionNumbersJacobian(char norm);
  void getvarM(int varM);

  double getEigenResidual();

  void evalAndGetgfast(std::vector<double> & gfast);

  int computeJacobianNumericalRank();

  void evalCSPPointers();

  void getCSPPointers( std::vector<std::vector<double>> &cspp_ij );

  void evalAndGetCSPPointers(const int & modeIndx, std::vector<double> &cspp_k);

};


#endif  //end of header guard
