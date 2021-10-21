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


#ifndef MODEL_CSP_CHEM_DAE
#define MODEL_CSP_CHEM_DAE

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>
#include <unistd.h>
#include <functional>

#include "model.hpp"

class ChemicalElementaryDAE: public Model {

 private:
  /* Intermediate matirces to evaluate target Jacobian _jacMat_Gu */
  int _nalge_var;
  std::vector<double> _alge_vec;
  std::vector<std::vector<double> > _jacMat_gu;
  std::vector<std::vector<double> > _jacMat_gv;
  std::vector<std::vector<double> > _jacMat_fu;
  std::vector<std::vector<double> > _jacMat_fv;
  std::vector<std::vector<double> > _jacMat_vu;

  /* std::function to be passed by users through constructor */
  std::function<void(
                     std::vector<double>&)> _rhsFuncK;

  std::function<void(const std::vector<double>&,        //diff_vec
                     const std::vector<double>&,        //alge_vec
                     std::vector<double>&)> _rhsFunc;


  std::function<void(        //alge_vec
                     std::vector<std::vector<double>>&,  //jacobian
                     std::vector<std::vector<double>>&,  //jacobian
                     std::vector<std::vector<double>>&,  //jacobian
                     std::vector<std::vector<double>>&  //jacobian
                     //unsigned int
                     )> _jacFunc;

  std::function<void(const std::vector<double>&,        //diff_vec
                     const std::vector<double>&,        //alge_vec
                     std::vector<std::vector<double>>&  //jacobian
                     //unsigned int
                    )> _jacFunc_gu;

  std::function<void(const std::vector<double>&,        //diff_vec
                     const std::vector<double>&,        //alge_vec
                     std::vector<std::vector<double>>&  //jacobian
                     //unsigned int
                    )> _jacFunc_gv;

  std::function<void(const std::vector<double>&,        //diff_vec
                     const std::vector<double>&,        //alge_vec
                     std::vector<std::vector<double>>&  //jacobian
                     //unsigned int
                    )> _jacFunc_fu;

  std::function<void(const std::vector<double>&,        //diff_vec
                     const std::vector<double>&,        //alge_vec
                     std::vector<std::vector<double>>&  //jacobian
                     //unsigned int
                    )> _jacFunc_fv;

  std::function<void(const std::vector<double>&,        //diff_vec
                     const std::vector<double>&,        //alge_vec
                     std::vector<std::vector<double>>&  //jacobian to be solved
                     //unsigned int
                    )> _jacFunc_vu;

 public:
  //constructor:
  ChemicalElementaryDAE(
    std::function<int(const std::vector<double> &state_vec,
                      const std::vector<double> &alge_vec,
                      std::vector<double> &source_vec
                     )> rhsFunc,
    //
    std::function<void(const std::vector<double> &state_vec,
                       const std::vector<double> &alge_vec,
                       std::vector<std::vector<double> > &jacMat_gu
                                         //unsigned int flag
                      )> jacFunc_gu,
    //
    std::function<void(const std::vector<double> &state_vec,
                       const std::vector<double> &alge_vec,
                       std::vector<std::vector<double> > &jacMat_gv
                       //unsigned int flag
                      )> jacFunc_gv,
    //
    std::function<void(const std::vector<double> &state_vec,
                       const std::vector<double> &alge_vec,
                       std::vector<std::vector<double> > &jacMat_fu
                       //unsigned int flag
                      )> jacFunc_fu,
    //
    std::function<void(const std::vector<double> &state_vec,
                       const std::vector<double> &alge_vec,
                       std::vector<std::vector<double> > &jacMat_fv
                       //unsigned int flag
                      )> jacFunc_fv
  ) :
    //_alge_vec(alge_vec),
    _rhsFunc(rhsFunc),
    _jacFunc_gu(jacFunc_gu),
    _jacFunc_gv(jacFunc_gv),
    _jacFunc_fu(jacFunc_fu),
    _jacFunc_fv(jacFunc_fv)
  {}
  ChemicalElementaryDAE(
      std::function<int(std::vector<double> &source_vec)> rhsFunc,
      //
      std::function<void(std::vector<std::vector<double> > &jacMat_gu,
                         std::vector<std::vector<double> > &jacMat_gv,
                         std::vector<std::vector<double> > &jacMat_fu,
                         std::vector<std::vector<double> > &jacMat_fv
                        )> jacFunc
    ) :
    _rhsFuncK(rhsFunc),
    _jacFunc(jacFunc)
   {}

  int initmore(int nalge_var);
  int setAlgeVarVector(const std::vector<double>& alge_vec);
  int getAlgeVarVector(std::vector<double>& alge_vec);

  int evalSourceVector();                     // from base class Model
  int evalSourceVectorK();
  int evalJacMatrix(unsigned int useJacAnl);
  int evalJacMatFuFvGuGv(); // get matrix Fu,Fv,Gu,Gv
  int evalJacMat_gu();
  int evalJacMat_gv();
  int evalJacMat_fu();
  int evalJacMat_fv();
  int evalJacMat_vu();


};

#endif  //end of header guard
