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


#include <numeric>
#include "index.hpp"

#define sign(x) x==0.0 ? 0.0 : x/fabs(x)

/*
******************************************************************
   participation index for all modes
                         (b^m . S_k) . r^k
   P(m,k)           = -----------------------------         m=1,N
                      sum_kk | (b^m . S_kk) . r^kk |

   fast importance index for fast subspace
                        (a_m b^m . S_k) . r^k
   ImpIndFast(m,k)  = ----------------------------------    m=1,M
                      sum_kk | (a_m b^m . S_kk) . r^kk |

   slow importance index for slow subspace
                        (a_m b^m . S_k) . r^k
   ImpIndSlow(m,k)  = ----------------------------------    m=M+1,N
                      sum_kk | (a_m b^m . S_kk) . r^kk |

 SAME CALCULATION

  importance index    .... Proj =  a_m b^m
  participation index .... Proj =  b^m

   generalized index
                        (Proj^m . S_k) . r^k
   ImpIndGen(m,k) = ----------------------------------
                    sum_kk | (Proj^m . S_kk) . r^kk |
................................................................
*/



//-----------------------------------------------------------
/* Initialization with chemical kinetic model data */
int CSPIndex::initChemKinModel(
    std::vector<std::vector<double> > &Smat,
    std::vector<double> &RoP ,
    std::vector<std::vector<double> > &dRoP,
    std::vector<double> &Wbar /* TSR coefficients */
) {

  _Smat = Smat ;
  _RoP  = RoP  ;
  _dRoP = dRoP ;
  _Wbar = Wbar ;

  evalBeta(); // compute beta

  return 0;
} // end of initChemKinModel

int CSPIndex::initChemKinModel(
    std::vector<std::vector<double> > &Smat,
    std::vector<double> &RoP
) {

  _Smat = Smat ;
  _RoP  = RoP  ;

  evalBeta(); // compute beta

  return 0;
} // end of initChemKinModel

//-----------------------------------------------------------
int CSPIndex::evalBeta() {

  if( _Smat.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"
             << "Matrix for Stoichiometric vectors is empty.\n"
             << "Call CSPIndex::initChemKinModel to fill out the matrix.\n";
    exit(1);
  }

  _beta_ik = std::vector<std::vector<double> >(_Nmode, std::vector<double>(_Nreac, 0.0));

  for (int i=0; i<_Nmode; i++) {
    for (int k=0; k<_Nreac; k++) {
      for (int j=0; j<_Nvar; j++) {
        _beta_ik[i][k] += _B[i][j] * _Smat[j][k];
      }
    }
  }

  return 0;
} // end of evalBeta

//-----------------------------------------------------------
int CSPIndex::getBeta(std::vector<std::vector<double> > &beta_ik ) {

  if( _beta_ik.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"
             << "_beta_ik matrix is empty.\n"
             << "Call CSPIndex::CSPIndex::evalBeta to fill out the matrix.\n";
    exit(1);
  }

  beta_ik = _beta_ik;
  return 0;
} // end of getlBeta

//-----------------------------------------------------------
int CSPIndex::evalAlpha() {

  if( _Smat.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"
             << "Matrix for Stoichiometric vectors is empty.\n"
             << "Call CSPIndex::initChemKinModel to fill out the matrix.\n";
    exit(1);
  }

  // evalBeta(); // this will populate private "_beta_ik"

  _alpha_jk = std::vector<std::vector<double> >( _Nvar, std::vector<double>(_Nreac, 0.0));

  for (int k=0; k<_Nreac; k++) {
    for (int j=0; j<_Nvar; j++) {
      for (int i=_M; i<_Nmode; i++) {
        _alpha_jk[j][k] += _A[j][i] * _beta_ik[i][k];
      }
    }
  }

  return 0;
} // end of evalAlpha

//-----------------------------------------------------------
int CSPIndex::getAlpha( std::vector<std::vector<double> > &alpha_jk ) {

  if( _alpha_jk.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"
             << "_alpha_jk matrix is empty.\n"
             << "Call CSPIndex::CSPIndex::evalAlpha to fill out the matrix.\n";
    exit(1);
  }

  alpha_jk = _alpha_jk;
  return 0;
} // end of getAlpha


//-----------------------------------------------------------
int CSPIndex::evalGamma() {

  if( _Smat.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"
             << "Matrix for Stoichiometric vectors is empty.\n"
             << "Call CSPIndex::initChemKinModel to fill out the matrix.\n";
    exit(1);
  }

 // evalBeta(); // this will populate private "_beta_ik"

  _gamma_jk = std::vector<std::vector<double> >( _Nvar, std::vector<double>(_Nreac, 0.0));

  for (int k=0; k<_Nreac; k++) {
    for (int j=0; j<_Nvar; j++) {
      for (int i=0; i<_M; i++) {
        _gamma_jk[j][k] += _A[j][i] * _beta_ik[i][k];
      }
    }
  }

  return 0;
} // end of evalGamma

//-----------------------------------------------------------
int CSPIndex::getGamma( std::vector<std::vector<double> > &gamma_jk ) {

  if( _gamma_jk.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"
             << "_gamma_jk matrix is empty.\n"
             << "Call CSPIndex::CSPIndex::evalGamma to fill out the matrix.\n";
    exit(1);
  }

  gamma_jk = _gamma_jk;
  return 0;
} // end of getGamma

//-----------------------------------------------------------//
int CSPIndex::evalParticipationIndex() {

  if( _Smat.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"
             << "Matrix for Stoichiometric vectors is empty.\n"
             << "Call CSPIndex::initChemKinModel to fill out the matrix.\n";
    exit(1);
  }

  if( _RoP.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"
             << "Vector of Rate of Progress is empty.\n"
             << "Call CSPIndex::initChemKinModel to fill out the matrix.\n";
    exit(1);
  }

  if( _beta_ik.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"
             << "_beta_ik matrix is empty.\n"
             << "Call CSPIndex::initChemKinModel to fill out the matrix.\n";
    exit(1);
  }

  // evalBeta();  // this will populate private data member "_beta_ik"

  std::vector<double> deno(_Nmode, 0.0);
  for (int i=0; i<_Nmode; i++) {
    for (int k=0; k<_Nreac; k++) {
      deno[i] += fabs( (_beta_ik[i][k] * _RoP[k] ) );
    }
  }

  _P_ik = std::vector<std::vector<double> >(_Nmode, std::vector<double>(_Nreac));
  std::vector<double> ortho_test(_Nmode, 0.0);
  for (int i=0; i<_Nmode; i++) {

    if (deno[i] != 0) {
      for (int k=0; k<_Nreac; k++) {
        _P_ik[i][k] = _beta_ik[i][k] * _RoP[k] / deno[i] ;
        ortho_test[i] += fabs(_P_ik[i][k]);
      }

      if ( fabs( fabs(ortho_test[i]) - 1.0) > 1.e-14 ) {
        std::cout<<"From:"<<__func__
        <<"\n    The sum over reactions of Participation Index for mode,\n\t i= "
        << i << ", as ortho_test[i] = "<< ortho_test[i] <<std::endl;
      }

    } else {
      for (int k=0; k<_Nreac; k++)
        _P_ik[i][k] = 0;
    }

  }

  return 0;
}  // end of evalParticipationIndex

//-----------------------------------------------------------//
int CSPIndex::evalAndGetParticipationIndex(const int &modeIndx, std::vector<double> &P_k) {

  if( _Smat.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"
             << "Matrix for Stoichiometric vectors is empty.\n"
             << "Call CSPIndex::initChemKinModel to fill out the matrix.\n";
    exit(1);
  }

  if( _RoP.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"
             << "Vector of Rate of Progress is empty.\n"
             << "Call CSPIndex::initChemKinModel to fill out the matrix.\n";
    exit(1);
  }

  if( _beta_ik.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"
             << "_beta_ik matrix is empty.\n"
             << "Call CSPIndex::initChemKinModel to fill out the matrix.\n";
    exit(1);
  }

  if (modeIndx > _Nmode -1 ){
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"
             << "mode index is out of bounds \n"
             << "select an index smaller than" << _Nmode -1;
    exit(1);

  }

  double deno(0.0);

  for (int k=0; k<_Nreac; k++) {
      deno += fabs( (_beta_ik[modeIndx][k] * _RoP[k] ) );
  }

  P_k = std::vector<double >(_Nreac, 0);
  for (int k=0; k<_Nreac; k++) {
      P_k[k] = _beta_ik[modeIndx][k] * _RoP[k] / deno ;
  }


  return 0;
}  // end of evalParticipationIndex

//-----------------------------------------------------------
int CSPIndex::evalImportanceIndexSlow() {

  if( _Smat.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"
             << "Matrix for Stoichiometric vectors is empty.\n"
             << "Call CSPIndex::initChemKinModel to fill out the matrix.\n";
    exit(1);
  }

  if( _RoP.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"
             << "Vector of Rate of Progress is empty.\n"
             << "Call CSPIndex::initChemKinModel to fill out the matrix.\n";
    exit(1);
  }

  evalAlpha(); // this will populate private 2D data member "_alpha_jk"

  std::vector<double> deno(_Nmode, 0.0);
  for (int j=0; j<_Nvar; j++) {
    for (int k=0; k<_Nreac; k++) {
      deno[j] += fabs(_alpha_jk[j][k] * _RoP[k]);
    }
  }

  _Islow_jk = std::vector<std::vector<double> >(_Nvar, std::vector<double>(_Nreac));
  std::vector<double> ortho_test(_Nvar, 0.0);

  for (int j=0; j<_Nvar; j++) {
    if (deno[j] != 0 ){

      for (int k=0; k<_Nreac; k++) {
        _Islow_jk[j][k] = _alpha_jk[j][k] * _RoP[k] / deno[j] ;
        ortho_test[j] += fabs(_Islow_jk[j][k]);
      }

      if ( fabs( fabs(ortho_test[j]) - 1.0) > 1.e-14 ) {
        std::cout<<"From:"<<__func__
        <<"\n    The sum over reactions of Importance Index (slow) for state variable,\n\t j= "
        << j << ", as ortho_test[j] = "<< ortho_test[j] <<std::endl;
      }

    } else {

      for (int k=0; k<_Nreac; k++) {
        _Islow_jk[j][k] = 0 ;
      }

    }

  }

  return 0;
}

//-----------------------------------------------------------
int CSPIndex::evalAndGetImportanceIndexSlow(const int & varIndx, std::vector<double> &Islow_k) {

  if( _Smat.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"
             << "Matrix for Stoichiometric vectors is empty.\n"
             << "Call CSPIndex::initChemKinModel to fill out the matrix.\n";
    exit(1);
  }

  if( _RoP.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"
             << "Vector of Rate of Progress is empty.\n"
             << "Call CSPIndex::initChemKinModel to fill out the matrix.\n";
    exit(1);
  }

  if (varIndx > _Nvar -1 ){
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"
             << "variable index is out of bounds \n"
             << "select an index smaller than" << _Nvar -1;
    exit(1);

  }

  std::vector< double > alpha_k( _Nreac, 0.0);

  for (int k=0; k<_Nreac; k++) {
    for (int i=_M; i<_Nmode; i++) {
        alpha_k[k] += _A[varIndx][i] * _beta_ik[i][k];
    }
  }

  double deno(0.0);

  for (int k=0; k<_Nreac; k++) {
      deno += fabs(_alpha_jk[varIndx][k] * _RoP[k]);
  }

  Islow_k = std::vector<double>(_Nreac, 0);

  for (int k=0; k<_Nreac; k++) {
    Islow_k[k] = alpha_k[k] * _RoP[k] / deno ;
  }

  return 0;
}

//-----------------------------------------------------------
int CSPIndex::evalImportanceIndexFast() {

  if( _Smat.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"
             << "Matrix for Stoichiometric vectors is empty.\n"
             << "Call CSPIndex::initChemKinModel to fill out the matrix.\n";
    exit(1);
  }

  if( _RoP.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"
             << "Vector of Rate of Progress is empty.\n"
             << "Call CSPIndex::initChemKinModel to fill out the matrix.\n";
    exit(1);
  }

  evalGamma(); // this will populate private 2D data member "_gamma_jk"

  std::vector<double> deno(_Nmode, 0.0);
  for (int j=0; j<_Nvar; j++) {
    for (int k=0; k<_Nreac; k++) {
      deno[j] += fabs(_gamma_jk[j][k] * _RoP[k]);
    }
  }

  _Ifast_jk = std::vector<std::vector<double> >(_Nvar, std::vector<double>(_Nreac));
  std::vector<double> ortho_test(_Nvar, 0.0);

  for (int j=0; j<_Nvar; j++) {
    if (deno[j] != 0 ){
      for (int k=0; k<_Nreac; k++) {
        _Ifast_jk[j][k] = _gamma_jk[j][k] * _RoP[k] / deno[j] ;
        ortho_test[j] += fabs(_Ifast_jk[j][k]);
      }

      if ( fabs( fabs(ortho_test[j]) - 1.0) > 1.e-14 ) {
        std::cout<<"From:"<<__func__
        <<"\n    The sum over reactions of Importance Index (fast) for state variable,\n\t j= "
        << j << ", as ortho_test[j] = "<< ortho_test[j] <<std::endl;
      }

    } else {
      for (int k=0; k<_Nreac; k++) {
        _Ifast_jk[j][k] = 0 ;
      }

    }




  }

  return 0;
}

//-----------------------------------------------------------
int CSPIndex::evalAndGetImportanceIndexFast(const int & varIndx, std::vector<double> &Ifast_k) {

  if( _Smat.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"
             << "Matrix for Stoichiometric vectors is empty.\n"
             << "Call CSPIndex::initChemKinModel to fill out the matrix.\n";
    exit(1);
  }

  if( _RoP.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"
             << "Vector of Rate of Progress is empty.\n"
             << "Call CSPIndex::initChemKinModel to fill out the matrix.\n";
    exit(1);
  }

  if (varIndx > _Nvar -1 ){
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"
             << "variable index is out of bounds \n"
             << "select an index smaller than" << _Nvar -1;
    exit(1);

  }

  std::vector<double> gamma_k(_Nreac, 0.0);

  for (int k=0; k<_Nreac; k++) {
    for (int i=0; i<_M; i++) {
      gamma_k[k] += _A[varIndx][i] * _beta_ik[i][k];
    }
  }

  double deno(0.0);
  for (int k=0; k<_Nreac; k++) {
   deno += fabs(gamma_k[k] * _RoP[k]);
  }

  Ifast_k = std::vector<double >(_Nreac);

  for (int k=0; k<_Nreac; k++) {
    Ifast_k[k] = gamma_k[k] * _RoP[k] / deno;
  }

  return 0;
}

//-----------------------------------------------------------
int CSPIndex::evalTimeScaleImportanceIndex() {

  if( _Smat.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"
             << "Matrix for Stoichiometric vectors is empty.\n"
             << "Call CSPIndex::initChemKinModel to fill out the matrix.\n";
    exit(1);
  }

  if( _dRoP.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"
             << "Matrix of derivatives of Rate of Progress is empty.\n"
             << "Call CSPIndex::initChemKinModel to fill out the matrix.\n";
    exit(1);
  }

  // evalBeta();  // this will populate private 2D data member "_beta_ik"

  std::vector< std::vector<double> > dRoP_A(_Nreac, std::vector<double>(_Nmode, 0.0) );

  for (int i=0; i<_Nmode; i++) {
  for (int k=0; k<_Nreac; k++) {
    for (int j=0; j<_Nvar; j++) {
      dRoP_A[i][k] += _dRoP[j][k] * _A[j][i] ;
    }
  }
  }

  std::vector<double> deno(_Nmode, 0.0);
  for (int i=0; i<_Nmode; i++) {
    for (int k=0; k<_Nreac; k++) {
      deno[i] += fabs(_beta_ik[i][k] * dRoP_A[i][k] );
    }
  }

  _J_ik = std::vector<std::vector<double> >(_Nmode, std::vector<double>(_Nreac));

  for (int i=0; i<_Nmode; i++) {
    for (int k=0; k<_Nreac; k++) {
      _J_ik[i][k] = _beta_ik[i][k] * dRoP_A[i][k]/deno[i] ;
    }
  }

  return 0;
}


//-----------------------------------------------------------
/* Participation Index of the i-th mode to TSR */
int CSPIndex::evalParticipationIndex_TSR_i() {

  if( _Wbar.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"
             << "TSR coefficients vector is empty. "
             << "Call Kernel::evalTSRcoef to fill out the vector.\n";
    exit(1);
  }

  double sum_Wbar=0.0;
  std::vector<double>WbarL(_Nmode);

  for (int i=0; i<_Nmode; i++ ) {
    WbarL[i] = _Wbar[i] * sqrt( pow(_eig_val_real[i],2.0)
                              + pow(_eig_val_imag[i],2.0) );
    sum_Wbar += fabs( WbarL[i] )  ;
  }

  _Ptsr_i = std::vector<double>(_Nmode);
  for (int i=0; i<_Nmode; i++ ) {
    _Ptsr_i[i] = sign(_eig_val_real[i])
               //(_eig_val_real[i]/fabs(_eig_val_real[i]))
               * WbarL[i]/sum_Wbar;
  }

  return 0;
}

//-----------------------------------------------------------
/* Participation Index of the k-th reaction to i-th mode to TSR */
int CSPIndex::evalParticipationIndex_TSR_ik() {

  if( _P_ik.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"
             << "Matrix of Participation Index of the k-th reaction to i-th mode is empty.\n"
             << "Call CSPIndex::evalParticipationIndex to fill out the vector.\n";
    exit(1);
  }

  if( _Ptsr_i.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"
	     << "Vector for Participation Index of the i-th mode to TSR is empty.\n"
             << "Call CSPIndex::evalParticipationIndex_TSR_i to fill out the vector.\n";
    exit(1);
  }

  _Ptsr_ik = std::vector<std::vector<double> >(_Nmode, std::vector<double>(_Nreac));

  for (int i=0; i<_Nmode; i++ ) {
    for (int k=0; k<_Nreac; k++ ) {
      _Ptsr_ik[i][k] = _Ptsr_i[i] * _P_ik[i][k];
    }
  }

  return 0;
}

//-----------------------------------------------------------
/* Participation Index of the k-th reaction to TSR */
int CSPIndex::evalParticipationIndex_TSR_k() {

  if( _P_ik.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"
             << "Matrix of Participation Index of the k-th reaction to i-th mode is empty.\n"
             << "Call CSPIndex::evalParticipationIndex to fill out the vector.\n";
    exit(1);
  }

  if( _Ptsr_i.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"
       << "Vector for Participation Index of the i-th mode to TSR is empty.\n"
             << "Call CSPIndex::evalParticipationIndex_TSR_i to fill out the vector.\n";
    exit(1);
  }

  _Ptsr_k = std::vector<double>(_Nreac,0.0);

  for (int k=0; k<_Nreac; k++ ) {
    for (int i=0; i<_Nmode; i++ ) {
      _Ptsr_k[k] += _Ptsr_i[i] * _P_ik[i][k];
    }
  }

  return 0;
}

//===========================================================
// All get functions for the csp indices:
//-----------------------------------------------------------
int CSPIndex::getParticipationIndex(
       std::vector<std::vector<double> > &P_ik ) {

  if( _P_ik.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"
             << "_P_ik matrix is empty.\n"
             << "Call CSPIndex::evalParticipationIndex to fill out the matrix.\n";
    exit(1);
  }


  P_ik = _P_ik;
  return 0;
}

//-----------------------------------------------------------
int CSPIndex::getImportanceIndexSlow(
       std::vector<std::vector<double> > &Islow_jk ) {

  if( _Islow_jk.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"
             << "_Islow_jk matrix is empty.\n"
             << "Call CSPIndex::evalImportanceIndexSlow to fill out the matrix.\n";
    exit(1);
  }


  Islow_jk = _Islow_jk;
  return 0;
}

//-----------------------------------------------------------
int CSPIndex::getImportanceIndexFast(
       std::vector<std::vector<double> > &Ifast_jk ) {

  if( _Ifast_jk.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"
             << "_Ifast_jk matrix is empty.\n"
             << "Call CSPIndex::evalImportanceIndexFast to fill out the matrix.\n";
    exit(1);
  }

  Ifast_jk = _Ifast_jk;
  return 0;
}

//-----------------------------------------------------------
int CSPIndex::getTimeScaleImportanceIndex(
         std::vector<std::vector<double> > &J_ik ) {

  if( _J_ik.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"
             << "_J_ik matrix is empty.\n"
             << "Call CSPIndex::evalTimeScaleImportanceIndex to fill out the matrix.\n";
    exit(1);
  }

  J_ik = _J_ik;
  return 0;
}

//-----------------------------------------------------------

void CSPIndex::getTopIndex(std::vector<double> &Index,
                         const int & Top, const double & threshold, std::vector<int> & IndxList ){
     // initialize original index locations
    std::vector<size_t> idx(Index.size());
    std::iota(idx.begin(), idx.end(), 0);

    std::stable_sort(idx.begin(), idx.end(),
       [&Index](size_t i1, size_t i2)
       {return std::fabs(Index[i1]) > std::fabs(Index[i2]);});

    // IndxList.clear();
    for (int i = 0; i < Top; i++) {
      if (std::fabs(Index[idx[i]]) >= threshold ){
        IndxList.push_back(idx[i]);
        // printf("indx %lu, rop %e \n",idx[i],Index[idx[i]]  );
      }
    }
    // delete duplicates
    sort( IndxList.begin(), IndxList.end() );
    IndxList.erase( std::unique( IndxList.begin(), IndxList.end() ), IndxList.end() );

}

//-----------------------------------------------------------
int CSPIndex::getParticipationIndex_TSR_i(
       std::vector<double> &Ptsr_i ) {

  if( _Ptsr_i.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"
             << "_Ptsr_i vector is empty.\n"
             << "Call CSPIndex::evalParticipationIndex_TSR_i to fill out the vector.\n";
    exit(1);
  }

  Ptsr_i = _Ptsr_i;
  return 0;
}

//-----------------------------------------------------------
int CSPIndex::getParticipationIndex_TSR_k(
       std::vector<double> &Ptsr_k ) {

  if( _Ptsr_k.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"
             << "_Ptsr_k vector is empty.\n"
             << "Call CSPIndex::evalParticipationIndex_TSR_k to fill out the vector.\n";
    exit(1);
  }

  Ptsr_k = _Ptsr_k;
  return 0;
}

//-----------------------------------------------------------
int CSPIndex::getParticipationIndex_TSR_ik(
       std::vector<std::vector<double> > &Ptsr_ik ) {

  if( _Ptsr_ik.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"
             << "_Ptsr_ik matrix is empty.\n"
             << "Call CSPIndex::evalParticipationIndex_TSR_ik to fill out the matrix.\n";
    exit(1);
  }

  Ptsr_ik = _Ptsr_ik;
  return 0;
}

//-----------------------------------------------------------
