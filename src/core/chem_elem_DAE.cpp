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
#include "tools.hpp"
#include "util.hpp"

#include "Tines.hpp"

#define DEBUG
//===========================================================
int ChemicalElementaryDAE::initmore(int nalge_var) {
  // This function should read in all chemical kinetic model info.

  _nalge_var = nalge_var;

  return(0);
}

//===========================================================
// Setting the algebraic variable:
int ChemicalElementaryDAE::setAlgeVarVector(
    const std::vector<double>& alge_vec) {

  _alge_vec = alge_vec;
  return(0);
}
//===========================================================
// Getting the algebraic variable:
int ChemicalElementaryDAE::getAlgeVarVector(
    std::vector<double> &alge_vec) {

  if( _alge_vec.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":\n"
             << "Vector for algebraic variable is empty.\n"
             << "Use ChemicalElementaryDAE::setAlgeVarVector to fill out the vector.\n";
    exit(1);
  }

  alge_vec = _alge_vec;
  return(0);
}
//===========================================================
int ChemicalElementaryDAE::evalSourceVector() {

  if( _state_vec.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":\n"
             << "State vector is empty.\n"
             << "Use ChemicalElementaryDAE::setStateVector(state) to set the state vector.\n";
    exit(1);
  }

  if( _alge_vec.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":\n"
             << "Vector for algebraic variable is empty.\n"
             << "Use ChemicalElementaryDAE::setAlgeVarVector to fill out the vector.\n";
    exit(1);
  }

  _rhsFunc(_state_vec, _alge_vec, _source_vec);
  return(0);
}

int ChemicalElementaryDAE::evalSourceVectorK() {

  if( _state_vec.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":\n"
             << "State vector is empty.\n"
             << "Use ChemicalElementaryDAE::setStateVector(state) to set the state vector.\n";
    exit(1);
  }

  if( _alge_vec.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":\n"
             << "Vector for algebraic variable is empty.\n"
             << "Use ChemicalElementaryDAE::setAlgeVarVector to fill out the vector.\n";
    exit(1);
  }

  _rhsFuncK(_source_vec);
  return(0);
}

//===========================================================
//int ChemicalElementaryDAE::evalJacMatrix(unsigned int useJacAnl) {
//  jacfunc(_state_vec, _jmat, useJacAnl);
//  return(0);
//}
//===========================================================
int ChemicalElementaryDAE::evalJacMat_gu() {

  if( _state_vec.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":\n"
             << "State vector is empty.\n"
             << "Use ChemicalElementaryDAE::setStateVector(state) to set the state vector.\n";
    exit(1);
  }

  if( _alge_vec.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":\n"
             << "Vector for algebraic variable is empty.\n"
             << "Use ChemicalElementaryDAE::setAlgeVarVector to fill out the vector.\n";
    exit(1);
  }

  _jacFunc_gu(_state_vec, _alge_vec, _jacMat_gu);

  Util::Print::mat<double>("_jacMat_gu", RIF, Out2d, Dbl, _state_vec.size(), _state_vec.size(), _jacMat_gu);

  return(0);
}

//===========================================================
int ChemicalElementaryDAE::evalJacMat_gv() {

  if( _state_vec.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":\n"
             << "State vector is empty.\n"
             << "Use ChemicalElementaryDAE::setStateVector(state) to set the state vector.\n";
    exit(1);
  }

  if( _alge_vec.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":\n"
             << "Vector for algebraic variable is empty.\n"
             << "Use ChemicalElementaryDAE::setAlgeVarVector to fill out the vector.\n";
    exit(1);
  }

  _jacFunc_gv(_state_vec, _alge_vec, _jacMat_gv);

  Util::Print::mat<double>("_jacMat_gv", RIF, Out2d, Dbl, _state_vec.size(), _alge_vec.size(), _jacMat_gv);

  return(0);
}

//===========================================================
int ChemicalElementaryDAE::evalJacMat_fu() {

  if( _state_vec.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":\n"
             << "State vector is empty.\n"
             << "Use ChemicalElementaryDAE::setStateVector(state) to set the state vector.\n";
    exit(1);
  }

  if( _alge_vec.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":\n"
             << "Vector for algebraic variable is empty.\n"
             << "Use ChemicalElementaryDAE::setAlgeVarVector to fill out the vector.\n";
    exit(1);
  }

  _jacFunc_fu(_state_vec, _alge_vec, _jacMat_fu);

  Util::Print::mat<double>("_jacMat_fu", RIF, Out2d, Dbl, _alge_vec.size(), _state_vec.size(), _jacMat_fu);

  return(0);
}

//===========================================================
int ChemicalElementaryDAE::evalJacMat_fv() {

  if( _state_vec.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":\n"
             << "State vector is empty.\n"
             << "Use ChemicalElementaryDAE::setStateVector(state) to set the state vector.\n";
    exit(1);
  }

  if( _alge_vec.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":\n"
             << "Vector for algebraic variable is empty.\n"
             << "Use ChemicalElementaryDAE::setAlgeVarVector to fill out the vector.\n";
    exit(1);
  }

  _jacFunc_fv(_state_vec, _alge_vec, _jacMat_fv);

  Util::Print::mat<double>("_jacMat_fv", RIF, Out2d, Dbl, _alge_vec.size(), _alge_vec.size(), _jacMat_fv);

  return(0);
}

//===========================================================
int ChemicalElementaryDAE::evalJacMat_vu() {

  if( _state_vec.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":\n"
             << "State vector is empty.\n"
             << "Use ChemicalElementaryDAE::setStateVector(state) to set the state vector.\n";
    exit(1);
  }

  if( _alge_vec.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":\n"
             << "Vector for algebraic variable is empty.\n"
             << "Use ChemicalElementaryDAE::setAlgeVarVector to fill out the vector.\n";
    exit(1);
  }

  if( _jacMat_fv.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":\n"
             << "Matrix of 1st-order partial derivatives of algebraic function (f) w.r.t. algebraic variable (v) is empty.\n"
             << "Use ChemicalElementaryDAE::evalJacMat_fv.\n";
    exit(1);
  }

  if( _jacMat_fu.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":\n"
             << "Matrix of 1st-order partial derivatives of algebraic function (f) w.r.t. differential variable (u) is empty.\n"
             << "Use ChemicalElementaryDAE::evalJacMat_fu.\n";
   exit(1);
  }

  int nrowfv = _alge_vec.size();
  int ncolfv = _alge_vec.size();
  int nrowfu = _alge_vec.size();
  int ncolfu = _state_vec.size();

  if (_nalge_var == 1) {
    _jacMat_vu = _jacMat_fu;
    for (int j=0; j<ncolfu; j++)
        _jacMat_vu[0][j] /=(_jacMat_fv[0][0]+1e-23) ;
  } else {


  CSP::LinearSystemSolve(
      nrowfv, ncolfv,
      nrowfu, ncolfu,
      _jacMat_fv,
      _jacMat_fu,
      _jacMat_vu);
  }


#ifdef DEBUG

  std::vector<std::vector<double> > diff;
  diff = std::vector<std::vector<double> >(nrowfu, std::vector<double>(ncolfu, 0.0));

  std::vector<double> matA_1d, matB_1d;
  std::vector<double> matC_1d(ncolfu*nrowfu, 0.0);

  CSP::construct_1D_from_2D<double>( nrowfv, ncolfv, _jacMat_fv, matA_1d );
  CSP::construct_1D_from_2D<double>( nrowfu, ncolfu, _jacMat_vu, matB_1d );

  {
    const double one(1), zero(0);
    const int trans_tag = Tines::Trans::NoTranspose::tag;

    Tines::Gemm_HostTPL(trans_tag,trans_tag,
			nrowfv, ncolfu, ncolfv,
			one,
			&matA_1d[0], ncolfv, 1,
			&matB_1d[0], ncolfu, 1,
			zero,
			&matC_1d[0], ncolfu, 1);
  }
  CSP::construct_2D_from_1D<double>(nrowfv, ncolfu, matC_1d, diff); // add some line for defining row and col appropriately


  Util::Print::mat<double>("_jacMat_vu", RIF, Out2d, Dbl, _alge_vec.size(), _state_vec.size(), _jacMat_vu);
  Util::Print::mat<double>("fv*uv", RIF, Out2d, Dbl, _alge_vec.size(), _state_vec.size(), diff);

  for (int i=0; i<nrowfu; i++)  // loop over rows
    for (int j=0; j<ncolfu; j++)  // loop over columns
      diff[i][j] -= _jacMat_fu[i][j];

  Util::Print::mat<double>("fv*vu - fu", RIF, Out2d, Dbl, _alge_vec.size(), _state_vec.size(), diff);
  Util::Print::mat<double>("_jacMat_fu", RIF, Out2d, Dbl, _alge_vec.size(), _state_vec.size(), _jacMat_fu);
#endif

  return 0;
}

//===========================================================
//int ChemicalElementaryDAE::evalJacMat_Gu(unsigned int useJacAnl) {
int ChemicalElementaryDAE::evalJacMatrix(unsigned int useJacAnl) {

  if( _jacMat_gv.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":\n"
             << "Matrix of 1st-order partial derivatives of the source term (g) w.r.t. algebraic variable (v) is empty.\n"
             << "Use ChemicalElementaryDAE::evalJacMat_gv.\n";
    exit(1);
  }

  if( _jacMat_gu.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":\n"
             << "Matrix of 1st-order partial derivatives of the source term (g) w.r.t. differential variable (u) is empty.\n"
             << "Use ChemicalElementaryDAE::evalJacMat_gu.\n";
   exit(1);
  }

  if( _jacMat_vu.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":\n"
             << "Matrix of 1st-order partial derivatives of algebraic variables (v) w.r.t. differential variable (u) is empty.\n"
             << "Use ChemicalElementaryDAE::evalJacMat_vu.\n";
    exit(1);
  }

  int nrow_gv = _state_vec.size();
  int ncol_gv = _alge_vec.size();
  int nrow_vu = _alge_vec.size();
  int ncol_vu = _state_vec.size();
  // const char trans_gv = 'N'; // only use char literal
  // const char trans_vu = 'N'; // T

  // Allocating memory for the final Jaconian:
  _jmat = std::vector<std::vector<double> >(nrow_gv, std::vector<double>(ncol_vu, 0.0));


  std::vector<double> matA_1d, matB_1d;
  std::vector<double> matC_1d(nrow_gv*nrow_gv, 0.0);

  CSP::construct_1D_from_2D<double>( nrow_gv, ncol_gv, _jacMat_gv, matA_1d );
  CSP::construct_1D_from_2D<double>( nrow_vu, ncol_vu, _jacMat_vu, matB_1d );

  //
  {
    const double one(1), zero(0);
    const int trans_tag = Tines::Trans::NoTranspose::tag;

    Tines::Gemm_HostTPL(trans_tag,trans_tag,
			nrow_gv, ncol_vu, ncol_gv,
			one,
			&matA_1d[0], ncol_gv, 1,
			&matB_1d[0], ncol_gv, 1,
			zero,
			&matC_1d[0], ncol_vu, 1);
  }

//
  CSP::construct_2D_from_1D<double>(nrow_gv, ncol_vu, matC_1d, _jmat);

  Util::Print::mat<double>("_jmat before sum", RIF, Out2d, Dbl, nrow_gv, ncol_vu, _jmat);

  for (int i=0; i<nrow_gv; i++)
  for (int j=0; j<ncol_vu; j++) {
    _jmat[i][j] = _jmat[i][j] + _jacMat_gu[i][j];
  }

  Util::Print::mat<double>("_jmat", RIF, Out2d, Dbl, nrow_gv, ncol_vu, _jmat);

  return 0;
}
//===========================================================
int ChemicalElementaryDAE::evalJacMatFuFvGuGv(){


  if( _state_vec.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":\n"
             << "State vector is empty.\n"
             << "Use ChemicalElementaryDAE::setStateVector(state) to set the state vector.\n";
    exit(1);
  }

  if( _alge_vec.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":\n"
             << "Vector for algebraic variable is empty.\n"
             << "Use ChemicalElementaryDAE::setAlgeVarVector to fill out the vector.\n";
    exit(1);
  }

  _jacFunc(_jacMat_gu,_jacMat_gv,_jacMat_fu,_jacMat_fv);


  return(0);

}
