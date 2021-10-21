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


#ifndef INDEX_CSP
#define INDEX_CSP

#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <vector>
#include <string>
#include <math.h>
//#include "chem_elem_ODE.hpp"
#include "kernel.hpp"
#include "Tines.hpp"
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


class CSPIndex
{
 private:
  int _Nreac; // number of reactions in the chemical model
  int _Nvar;  // number of variables in a state-vector
  int _Nmode; // number of modes
  int _M;     // number of exhausted or fast modes

  /* Eigen values from the eigen decomposition of Jacobian of source term */
  std::vector<double> _eig_val_real;
  std::vector<double> _eig_val_imag;

  /* Matrices of Basis vectors */
  std::vector<std::vector<double> > _A; // csp basis vector (right)
  std::vector<std::vector<double> > _B; // Dual vector of _A (left)

  /* Chemical kinetic model related */
  std::vector<std::vector<double> > _Smat;
  std::vector<double> _RoP ;
  std::vector<std::vector<double> > _dRoP;

  /* Various indices */
  std::vector<std::vector<double> > _beta_ik;  // for i-th  mode
  std::vector<std::vector<double> > _alpha_jk; // contribution of slow modes to the j-th state variable
  std::vector<std::vector<double> > _gamma_jk; // contribution of fast modes to the j-th state variable
  std::vector<std::vector<double> > _P_ik;     // Participation index of k-th reaction to the amplitude of i-th mode
  std::vector<std::vector<double> > _Islow_jk; // j-th comp. of Nvars (slow modes)
  std::vector<std::vector<double> > _Ifast_jk; // j-th comp. of Nvars (fast modes)
  std::vector<std::vector<double> > _J_ik;
  // std::vector<std::vector<double> > _cspp_ij;

  //TSR related:
  std::vector<double> _Wbar;    // TSR coefficients (caculated in Kernel class)
  std::vector<double> _Ptsr_i;  // Participation index of i-th mode to the TSR
  std::vector<double> _Ptsr_k;  // Participation index of k-th reaction to the TSR
  std::vector<std::vector<double> > _Ptsr_ik; // Participation index of
                                              // the k-th reaction to
                                              // identify the contribution
                                              // to the development of TSR

 public:
   // create class before loop
  CSPIndex(
         int Nreac,
         int Nvar
        ) :
         _Nreac(Nreac),
         _Nvar(Nvar),
         _Nmode(Nvar)
          {
            intVariables();
          }

  CSPIndex(
        int Nreac,
        int Nvar,
        int Nmode,
        int M,
        std::vector<double> &eig_val_real,
        std::vector<double> &eig_val_imag,
        std::vector<std::vector<double> > &A,
        std::vector<std::vector<double> > &B
       ) :
        _Nreac(Nreac),
        _Nvar(Nvar),
        _Nmode(Nmode),
        _M(M),
        _eig_val_real(eig_val_real),
        _eig_val_imag(eig_val_imag),
        _A(A),
        _B(B)
        {
          intVariables();
        }


  //
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
       ) :
        _Nreac(Nreac),
        _Nvar(Nvar),
        _Nmode(Nvar),
        _M(M),
        _eig_val_real(eig_val_real),
        _eig_val_imag(eig_val_imag),
        _A(A),
        _B(B),
        _Smat(Smat),
        _RoP(RoP)
         {
           intVariables();
           evalBeta(); // compute beta
         }

  //int IndexGeneralForm();

  /* Initialization with chemical kinetic model data */
  int initChemKinModel(
      std::vector<std::vector<double> > &Smat,
      std::vector<double> &RoP ,
      std::vector<std::vector<double> > &dRoP,
      std::vector<double> &Wbar // TSR coefficients
  );

  int initChemKinModel(
      std::vector<std::vector<double> > &Smat,
      std::vector<double> &RoP
  );

  int initChemKinModel(
      int M,
      std::vector<double> &eig_val_real,
      std::vector<double> &eig_val_imag,
      std::vector<std::vector<double> > &A,
      std::vector<std::vector<double> > &B,
      std::vector<std::vector<double> > &Smat,
      std::vector<double> &RoP
  );

  void intVariables();

  /* Beta matrix */
  int evalBeta();


  void evalBetaV2(std::vector< double > &csp_b, std::vector< double > &smatrix)  ;

  int getBeta(std::vector<std::vector<double> > &beta_ik);

  /* Alpha matrix (for slow modes) */
  int evalAlpha();

  int getAlpha(std::vector<std::vector<double> > &alpha_jk);

  /* Gamma matrix (for fast modes) */
  int evalGamma();

  int getGamma(std::vector<std::vector<double> > &gamma_jk);

  //------------------------------------------------------//
  /* Participation Index */
  int evalParticipationIndex();

  int getParticipationIndex(std::vector<std::vector<double> > &P_ik);


  /*  Importance Index (Slow) */
  int evalImportanceIndexSlow();

  int getImportanceIndexSlow(std::vector<std::vector<double> > &Islow_jk);


  /* Importance Index (Fast) */
  int evalImportanceIndexFast();

  int getImportanceIndexFast(std::vector<std::vector<double> > &Ifast_jk);


  /* TimeScale Importance Index */
  int evalTimeScaleImportanceIndex();

  int getTimeScaleImportanceIndex(std::vector<std::vector<double> > &J_ik);

//-----------------------------------------------------------
/* Participation Index of the i-th mode to TSR */
  int evalParticipationIndex_TSR_i();

  int getParticipationIndex_TSR_i(
       std::vector<double> &Ptsr_i
  );

/* Participation Index of the k-th reaction to TSR */
  int evalParticipationIndex_TSR_k();

  int getParticipationIndex_TSR_k(
       std::vector<double> &Ptsr_k
  );

/* Participation Index of the k-th reaction in the i-th mode to TSR */
  int evalParticipationIndex_TSR_ik();

  int getParticipationIndex_TSR_ik(
       std::vector<std::vector<double> > &Ptsr_ik
  );
  // eval participation index at mode with position modeInx, and get info in P_k vector
  int evalAndGetParticipationIndex(const int &modeIndx,
    std::vector<double> &P_k) ;

  // eval and  slow importance index for one variable
  int evalAndGetImportanceIndexSlow(const int & varIndx,
     std::vector<double> &Islow_k);

  // eval and get fast importance index for one variable
  int evalAndGetImportanceIndexFast(const int & varIndx,
     std::vector<double> &Ifast_k);

  //
  void getTopIndex(std::vector<double> &Index, const int & Top,
     const double & threshold, std::vector<int>& IndxList  );
  //


}; // end of CSPIndex


#endif  //end of header guard
