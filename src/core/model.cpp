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
#include "model.hpp"

//==============================================================================
int  Model::init() {

  if (Nvars == 0) {
    std::cout << "Model::init has Nvars = 0\n";
    exit(1);
  }

  _state_vec.clear();
  _state_vec.shrink_to_fit();
  _state_vec=std::vector<double>(Nvars,0.0);

  _source_vec.clear();
  _source_vec.shrink_to_fit();
  _source_vec=std::vector<double>(Nvars,0.0);

  _jmat.clear();
  _jmat.shrink_to_fit();
  _jmat = std::vector<std::vector<double>>(Nvars,std::vector<double>(Nvars,0.0));

  return(0);
}
//==============================================================================
int  Model::init(int Nvars) {

  this->Nvars = Nvars;

  _state_vec.clear();
  _state_vec.shrink_to_fit();
  _state_vec=std::vector<double>(Nvars,0.0);

  _source_vec.clear();
  _source_vec.shrink_to_fit();
  _source_vec=std::vector<double>(Nvars,0.0);

  _jmat.clear();
  _jmat.shrink_to_fit();
  _jmat = std::vector<std::vector<double>>(Nvars,std::vector<double>(Nvars,0.0));

  return(0);
}
//==============================================================================
int  Model::setNumOfVariables(int Nvars) {
  this->Nvars = Nvars;
  return (0);
}
//==============================================================================
int  Model::getNumOfVariables() {
  return (Nvars);
}

//==============================================================================
int Model::setStateDB(const std::vector<double>& state_db) {
  _state_db = state_db;
  return(0);
}
//==============================================================================
int Model::getStateDB(std::vector<double>& state_db) {
  state_db = _state_db;
  return(0);
}

//==============================================================================
int Model::setStateVector(const std::vector<double>& state_vec) {
  _state_vec = state_vec;
  return(0);
}
//==============================================================================
int Model::getStateVector(std::vector<double>& state_vec) {
  if( !_state_vec.empty() ) {
    state_vec = _state_vec;
  } else {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"<<"State Vector is empty!"<<std::endl;
    exit(1);
  }
  return(0);
}

//==============================================================================
int Model::getSourceVector(std::vector<double>& source_vec) {
  if ( !_source_vec.empty() ) {
    source_vec = _source_vec;
  } else {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"<<"Source Vector is empty!"<<std::endl;
    exit(1);
  }
  return(0);
}
//==============================================================================
int Model::getJacMatrix(std::vector<std::vector<double> >& jmat) {
  if ( !_jmat.empty() ) {
    jmat = _jmat;
  } else {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"<<"_jmat is empty!"<<std::endl;
    exit(1);
  }
  return(0);
}
//==============================================================================
