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


#include <iostream>
#include <vector>
#include <string>
#include "chem_elem_DAE.hpp"
#include "util.hpp"

#define TEST3

#ifdef TEST1
//====================================================================
int rhsFunc(const std::vector<double>& state_vec,
        const std::vector<double>& alge_vec,
        std::vector<double>& source_vec) {

  source_vec[0]= -0.04*state_vec[0] + 1.e+4 * state_vec[1] * alge_vec[0];
  source_vec[1]=  0.04*state_vec[0] - 1.e+4 * state_vec[1] * alge_vec[0]
                                - 3.e+7 * state_vec[1] * state_vec[1];

  return(0);
}

//====================================================================
int jacFunc_fv(const std::vector<double>& state_vec,
               const std::vector<double>& alge_vec,
               std::vector<std::vector<double> >& jacMat_fv) {

  jacMat_fv = std::vector<std::vector<double> >(1,std::vector<double>(1,0.0));
  jacMat_fv[0][0] = 1.0;

  return(0);
}

//====================================================================
int jacFunc_fu(const std::vector<double>& state_vec,
               const std::vector<double>& alge_vec,
               std::vector<std::vector<double> >& jacMat_fu) {

  jacMat_fu = std::vector<std::vector<double> >(1,std::vector<double>(2,0.0));
  jacMat_fu[0][0] = -1;
  jacMat_fu[0][1] = -1;

  return(0);
}

//====================================================================
int jacFunc_gv(const std::vector<double>& state_vec,
               const std::vector<double>& alge_vec,
               std::vector<std::vector<double> >& jacMat_gv) {

  jacMat_gv = std::vector<std::vector<double> >(2,std::vector<double>(1,0.0));
  jacMat_gv[0][0] =  1.e+4 * state_vec[1];
  jacMat_gv[1][0] = -1.e+4 * state_vec[1] ;

  return(0);
}

//====================================================================
int jacFunc_gu(const std::vector<double>& state_vec,
               const std::vector<double>& alge_vec,
               std::vector<std::vector<double> >& jacMat_gu) {

  jacMat_gu = std::vector<std::vector<double> >(2,std::vector<double>(2,0.0));
  jacMat_gu[0][0] = -0.04;
  jacMat_gu[0][1] =  1.e+4 * alge_vec[0];
  jacMat_gu[1][0] =  0.04;
  jacMat_gu[1][1] = -1.e+4 * alge_vec[0] - (6.e+7) * state_vec[1] ;

  return(0);
}
#endif

#ifdef TEST2
//====================================================================
int rhsFunc(const std::vector<double>& state_vec,
        const std::vector<double>& alge_vec,
        std::vector<double>& source_vec) {

  source_vec[0]= -0.04*state_vec[0] + 1.e+4 * state_vec[1] * state_vec[2];
  source_vec[1]=  0.04*state_vec[0] - 1.e+4 * state_vec[1] * state_vec[2]
                                - 3.e+7 * state_vec[1] * state_vec[1];
  source_vec[2]=  3.e+7 * state_vec[1] * state_vec[1]
                                + state_vec[2] * alge_vec[0];

  return(0);
}

//====================================================================
int jacFunc_fv(const std::vector<double>& state_vec,
               const std::vector<double>& alge_vec,
               std::vector<std::vector<double> >& jacMat_fv) {

  jacMat_fv = std::vector<std::vector<double> >(2,std::vector<double>(2,0.0));
  jacMat_fv[0][0] = 1.0;
  jacMat_fv[0][1] = 0.0;
  jacMat_fv[1][0] = 0.0;
  jacMat_fv[1][1] = 1.0;

  return(0);
}

//====================================================================
int jacFunc_fu(const std::vector<double>& state_vec,
               const std::vector<double>& alge_vec,
               std::vector<std::vector<double> >& jacMat_fu) {

  jacMat_fu = std::vector<std::vector<double> >(2,std::vector<double>(3,0.0));
  jacMat_fu[0][0] = -1;
  jacMat_fu[0][1] = -1;
  jacMat_fu[0][2] = -1;
  jacMat_fu[1][0] =  0;
  jacMat_fu[1][1] = -3.e+7 * state_vec[1]*state_vec[1] ;
  jacMat_fu[1][2] =  0;

  return(0);
}

//====================================================================
int jacFunc_gv(const std::vector<double>& state_vec,
               const std::vector<double>& alge_vec,
               std::vector<std::vector<double> >& jacMat_gv) {

  jacMat_gv = std::vector<std::vector<double> >(3,std::vector<double>(2,0.0));
  jacMat_gv[0][0] = 0.0;
  jacMat_gv[0][1] = 0.0;
  jacMat_gv[1][0] = 0.0;
  jacMat_gv[1][1] = 0.0;
  jacMat_gv[2][0] = state_vec[2] ;
  jacMat_gv[2][1] = 0.0;

  return(0);
}

//====================================================================
int jacFunc_gu(const std::vector<double>& state_vec,
               const std::vector<double>& alge_vec,
               std::vector<std::vector<double> >& jacMat_gu) {

  jacMat_gu = std::vector<std::vector<double> >(3,std::vector<double>(3,0.0));
  jacMat_gu[0][0] = -0.04;
  jacMat_gu[0][1] =  1.e+4 * state_vec[2];
  jacMat_gu[0][2] =  1.e+4 * state_vec[1];
  jacMat_gu[1][0] =  0.04;
  jacMat_gu[1][1] = -1.e+4 * state_vec[2] - (6.e+7) * state_vec[1] ;
  jacMat_gu[1][2] = -1.e+4 * state_vec[1];
  jacMat_gu[2][0] =  0.0;
  jacMat_gu[2][1] =  6.e+7 * state_vec[1] ;
  jacMat_gu[2][2] =  alge_vec[0];

  return(0);
}
#endif

#ifdef TEST3
//====================================================================
int rhsFunc(const std::vector<double>& state_vec,
        const std::vector<double>& alge_vec,
        std::vector<double>& source_vec) {

  source_vec[0]=  -0.04 *state_vec[0] + 1.e+4 *state_vec[1]*state_vec[2]
                                      +        state_vec[2]*alge_vec[0]
                                      +   4.0 *state_vec[1]*alge_vec[1];
  source_vec[1]=   0.04 *state_vec[0] - 1.e+4 *state_vec[1]*state_vec[2]
                                      - 3.e+7 *state_vec[1]*state_vec[1]
                                      +        state_vec[1]*alge_vec[1];
  source_vec[2]=  3.e+7 *state_vec[1] * state_vec[1]
                                      + state_vec[2]*alge_vec[0]
                                      - state_vec[1]*alge_vec[1];

  return(0);
}

//====================================================================
int jacFunc_fv(const std::vector<double>& state_vec,
               const std::vector<double>& alge_vec,
               std::vector<std::vector<double> >& jacMat_fv) {

  jacMat_fv = std::vector<std::vector<double> >(2,std::vector<double>(2,0.0));
  jacMat_fv[0][0] = 1.0;
  jacMat_fv[0][1] = 1.0;
  jacMat_fv[1][0] = state_vec[1];
  jacMat_fv[1][1] = state_vec[2];

  return(0);
}

//====================================================================
int jacFunc_fu(const std::vector<double>& state_vec,
               const std::vector<double>& alge_vec,
               std::vector<std::vector<double> >& jacMat_fu) {

  jacMat_fu = std::vector<std::vector<double> >(2,std::vector<double>(3,0.0));
  jacMat_fu[0][0] = -1;
  jacMat_fu[0][1] = -1;
  jacMat_fu[0][2] = -1;
  jacMat_fu[1][0] =  0;
  jacMat_fu[1][1] = -alge_vec[0];
  jacMat_fu[1][2] = -alge_vec[1];

  return(0);
}

//====================================================================
int jacFunc_gv(const std::vector<double>& state_vec,
               const std::vector<double>& alge_vec,
               std::vector<std::vector<double> >& jacMat_gv) {

  jacMat_gv = std::vector<std::vector<double> >(3,std::vector<double>(2,0.0));
  jacMat_gv[0][0] = state_vec[2];
  jacMat_gv[0][1] = 4.0*state_vec[1];
  jacMat_gv[1][0] = 0.0;
  jacMat_gv[1][1] = state_vec[1];
  jacMat_gv[2][0] = state_vec[2];
  jacMat_gv[2][1] =-state_vec[1];

  return(0);
}

//====================================================================
int jacFunc_gu(const std::vector<double>& state_vec,
               const std::vector<double>& alge_vec,
               std::vector<std::vector<double> >& jacMat_gu) {

  jacMat_gu = std::vector<std::vector<double> >(3,std::vector<double>(3,0.0));
  jacMat_gu[0][0] = -0.04;
  jacMat_gu[0][1] =  1.e+4 * state_vec[2]+4.0*alge_vec[1];
  jacMat_gu[0][2] =  1.e+4 * state_vec[1]+alge_vec[0];
  jacMat_gu[1][0] =  0.04;
  jacMat_gu[1][1] = -1.e+4 * state_vec[2] - (6.e+7)*state_vec[1]+alge_vec[1] ;
  jacMat_gu[1][2] = -1.e+4 * state_vec[1];
  jacMat_gu[2][0] =  0.0;
  jacMat_gu[2][1] =  6.e+7 * state_vec[1] - alge_vec[1] ;
  jacMat_gu[2][2] =  alge_vec[0];

  return(0);
}
#endif

//====================================================================
int main() {

#ifdef TEST1
  std::vector<double> state (2, 1.0);
  std::vector<double> source (2, 0.0);

  std::vector<double> alge_vec (1, 0.0);
  alge_vec[0] = -1.0;
  int ndiff_var=2;
  int nalge_var=1;
#endif

#ifdef TEST2
  std::vector<double> state (3, 1.0);
  std::vector<double> source (3, 0.0);

  std::vector<double> alge_vec (2, 0.0);
  alge_vec[0] = -2.0;
  alge_vec[1] = -1.e+7 - 1.0;
  int ndiff_var=3;
  int nalge_var=2;
#endif

#ifdef TEST3
  std::vector<double> state (3, 0.0);
  std::vector<double> source (3, 0.0);

  state[0] = 1;
  state[1] = 0;
  state[2] = 1;

  std::vector<double> alge_vec (2, 0.0);
  alge_vec[0] = -0.5;
  alge_vec[1] = -0.5;
  int ndiff_var=3;
  int nalge_var=2;
#endif

  std::vector<std::vector<double>> jac (ndiff_var,std::vector<double>(ndiff_var, 0.0) );
  int flag=1;

  // instantiation:
  ChemicalElementaryDAE ma(
    std::function<int(const std::vector<double>&, const std::vector<double>&, std::vector<double>&)> (std::move(rhsFunc)),
    std::function<int(const std::vector<double>&, const std::vector<double>&, std::vector<std::vector<double> >&)> (std::move(jacFunc_gu)),
    std::function<int(const std::vector<double>&, const std::vector<double>&, std::vector<std::vector<double> >&)> (std::move(jacFunc_gv)),
    std::function<int(const std::vector<double>&, const std::vector<double>&, std::vector<std::vector<double> >&)> (std::move(jacFunc_fu)),
    std::function<int(const std::vector<double>&, const std::vector<double>&, std::vector<std::vector<double> >&)> (std::move(jacFunc_fv))
  );


  ma.init(ndiff_var);
  ma.setStateVector(state);

  ma.initmore(nalge_var);
  ma.setAlgeVarVector(alge_vec);

  ma.evalSourceVector();

  ma.getSourceVector(source);
  std::cout << "source: a: " << std::endl;
  for (auto s : source)
      std::cout << s << std::endl;

#if 1
  ma.evalJacMat_fv();
  ma.evalJacMat_fu();
  ma.evalJacMat_gv();
  ma.evalJacMat_gu();
  ma.evalJacMat_vu();

  ma.evalJacMatrix(flag);
  ma.getJacMatrix(jac);
  std::cout << "jac: a: " << std::endl;
  for (auto s : jac) {
    for (auto z : s) {
      std::cout << std::scientific;
      std::cout.width(20); std::cout<< std::right << z<<"\t";
      //printf("%10.5e\t", z);
    }
    std::cout<< std::endl;
  }

  Util::Print::mat<double>("jac", RIF, Out2d, Dbl, 3, 3, jac);
#endif

  return 0;
}
