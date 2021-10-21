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
#include "gODE.hpp"

int rhs_a(const std::vector<double>& state, std::vector<double>& source){
  for (size_t i=0; i<state.size(); ++i){
    source[i]=2.0*state[i];
  }
  return(0);
}

int rhs_b(const std::vector<double>& state, std::vector<double>& source){
  for (size_t i=0; i<state.size(); ++i){
    source[i]=-2.0*state[i];
  }
  return(0);
}

int rhs_Davis_Skodje(const std::vector<double>& state, std::vector<double>& source){
    const double epsilon = 0.01;
    const double y = state[0];
    const double z = state[1];
    source[0] = (-y+z/(1.+z))/epsilon - z/(1.+z)/(1.+z);
    source[1] = -z;
  return(0);
}

int jac_a(const std::vector<double>& state, std::vector<std::vector<double>>& jac, int flag){
  for (size_t i=0; i<state.size(); ++i){
  for (size_t j=0; j<state.size(); ++j){
    jac[i][j]=2.0*state[i]*state[j];
  }
  }
  return(0);
}

int jac_b(const std::vector<double>& state, std::vector<std::vector<double>>& jac, int flag){
  for (size_t i=0; i<state.size(); ++i){
  for (size_t j=0; j<state.size(); ++j){
    jac[i][j]=-2.0*state[i]*state[j];
  }
  }
  return(0);
}

int jac_Davis_Skodje(const std::vector<double>& state, std::vector<std::vector<double>>& jac, int flag){
  const double epsilon = 0.01;
  const double y = state[0];
  const double z = state[1];
  jac[0][0] = -1./epsilon;
  jac[1][0] = 0;
  jac[0][1] = 2. * z / pow( z + 1. , 3.) - 1. / pow( z + 1., 2) +
              ( - z / pow( z + 1. , 2.) + 1. / ( z + 1. ) ) / epsilon;
  jac[1][1] = -1;
  return(0);
}

int state_Davis_Skodje(std::vector<double>& state, const double t, std::vector<double>& state0){
    const double epsilon = 0.01;
    const double y0 = state0[0];
    const double z0 = state0[1];

    state[0] = (y0-z0/(1.+z0)) * exp(-t/epsilon) + z0*exp(-t)/(1.+z0*exp(-t));
    state[1] = z0*exp(-t);

    return 1;


}

int main() {

  std::vector<double> state (2,1);
  std::vector<double> source (2,0);
  std::vector<std::vector<double>> jac (2,std::vector<double>(2,0));
  int flag=1;

  //====================================================================
  GeneralODE ma(
    std::function<int(const std::vector<double>&, std::vector<double>&)> (std::move(rhs_a)),
    std::function<int(const std::vector<double>&, std::vector<std::vector<double>>&, int)> (std::move(jac_a))
  );

  ma.init(2);
  ma.setStateVector(state);

  ma.evalSourceVector();

  ma.getSourceVector(source);
  std::cout << "source: a: " << std::endl;
  for (auto s : source)
      std::cout << s << std::endl;

  ma.evalJacMatrix(flag);
  ma.getJacMatrix(jac);
  std::cout << "jac: a: " << std::endl;
  for (auto s : jac) {
    for (auto z : s)
      std::cout << z << " ";
    std::cout << std::endl;
  }

  //====================================================================

  GeneralODE mb(
    std::function<int(const std::vector<double>&, std::vector<double>&)> (std::move(rhs_b)),
    std::function<int(const std::vector<double>&, std::vector<std::vector<double>>&, int)> (std::move(jac_b))
  );

  mb.init(2);
  mb.setStateVector(state);

  mb.evalSourceVector();
  mb.getSourceVector(source);
  std::cout << "source: b: " << std::endl;
  for (auto s : source)
      std::cout << s << std::endl;

  mb.evalJacMatrix(flag);
  mb.getJacMatrix(jac);
  std::cout << "jac: b: " << std::endl;
  for (auto s : jac){
    for (auto z : s)
      std::cout << z << " ";
    std::cout << std::endl;
  }

//====================================================================
GeneralODE mDavis_Skodje(
  std::function<int(const std::vector<double>&, std::vector<double>&)> (std::move(rhs_Davis_Skodje)),
  std::function<int(const std::vector<double>&, std::vector<std::vector<double>>&, int)> (std::move(jac_Davis_Skodje))
);


std::vector<double> stateDS (2.,1e-2);
mDavis_Skodje.init(2);
mDavis_Skodje.setStateVector(stateDS);

std::vector<double> sourceDS (2,0);
mDavis_Skodje.evalSourceVector();
mDavis_Skodje.getSourceVector(sourceDS);
std::cout << "source: Davis_Skodje: " << std::endl;
for (auto s : sourceDS)
    std::cout << s << std::endl;

//
std::vector<std::vector<double>> jacSD (2,std::vector<double>(2,0));

mDavis_Skodje.evalJacMatrix(flag);
mDavis_Skodje.getJacMatrix(jacSD);
std::cout << "jac: Davis_Skodje: " << std::endl;
for (auto s : jacSD){
  for (auto z : s)
    std::cout << z << " ";
  std::cout << std::endl;
}


//====================================================================
  return 0;
}
