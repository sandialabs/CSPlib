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


#ifndef INTERFACE_TCHEM
#define INTERFACE_TCHEM

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cstring>

#include "TChem_Util.hpp"
#include "TChem_KineticModelData.hpp"
#include "TChem_PlugFlowReactorRHS.hpp"
#include "TChem_PlugFlowReactorNumJacobian.hpp"

using ordinal_type = TChem::ordinal_type;
using real_type = TChem::real_type;
using real_type_1d_view = TChem::real_type_1d_view;
using real_type_2d_view = TChem::real_type_2d_view;
using real_type_3d_view = TChem::real_type_3d_view;

using real_type_0d_view_host = TChem::real_type_0d_view_host;
using real_type_1d_view_host = TChem::real_type_1d_view_host;
using real_type_2d_view_host = TChem::real_type_2d_view_host;

class InterfaceTChem {

 private:
  double _pressure;
  int _Nelem, _Nspec, _Nreac, _Nvars, _NspecSurf,_Nalge ;

  std::vector<double> _state_vec  ;   // differential variables (or state vector)
  std::vector<double> _alge_vec   ;   // algebraic variables
  std::vector<double> _source_vec ;   // rhs/source vector
  std::vector<std::vector<double> > _jmat       ;   // jacobian matrix

  std::vector<std::string> _spec_name;

  std::string _mech_gas_file   ;
  std::string _mech_surf_file  ;
  std::string _thermo_gas_file ;
  std::string _thermo_surf_file;
  std::string _periodic_table_file;

  TChem::KineticModelData kmdSurf;
  KineticSurfModelConstDataDevice kmcdSurf;
  KineticModelConstDataDevice kmcd;
  real_type_2d_view _state;
  real_type_2d_view _siteFraction;
  real_type_1d_view _velocity;
  int _nBatch;
  real_type_2d_view _rhs;

  real_type_3d_view _Gu;
  real_type_3d_view _Fu;
  real_type_3d_view _Gv;
  real_type_3d_view _Fv;

 public:


  //constructor:

  InterfaceTChem(
                const std::string &mech_gas_file     ,
                const std::string &mech_surf_file    ,
                const std::string &thermo_gas_file   ,
                const std::string &thermo_surf_file
              );
  //{



  //}

  // InterfaceTChem(std::string prefixPath):
  //                 _prefixPath(prefixPath)
  // {}
  // InterfaceTChem();

  //destructor
  // ~InterfaceTChem();

  // Member functions:
  void InitializeTChem();
  void InitializeSurTChem();
  int  init()           ;
  void ResetTChem()     ;
  int  NumOfElements()  ;
  int  NumOfSpecies()   ;
  int  NumOfReactions() ;
  int  NumOfVariables() ;
  int  NumOfAlgebraicConstraints();

  // Handling species
  int loadSpeciesNames();
  int getSpeciesNames(std::vector<std::string>& spec_name);
  int setSpeciesNames(std::vector<std::string>& spec_name);

  // Handling state vector
  void genRandStateVector();
  int setStateVector(const std::vector<double>& state_vec);
  int getStateVector(std::vector<double>& state_vec);
  int getAlgeVector(std::vector<double>& alge_vec);

  //
  void readfromfileStateVector();
  void readfromfileStateVector(const std::string&,
                               const std::string&,
                               const std::string& );

  int rhsFunc ();

  int rhsFuncH (
      std::vector<double>& source_vec);

  int jacFunc();

  int jacFuncH(
      std::vector<std::vector<double> >& jacMat_gu,
      std::vector<std::vector<double> >& jacMat_gv,
      std::vector<std::vector<double> >& jacMat_fu,
      std::vector<std::vector<double> >& jacMat_fv);

  int jacFunc_gu(
      const std::vector<double>& state_vec,
      const std::vector<double>& alge_vec,
      std::vector<std::vector<double> >& jacMat_gu);

  int jacFunc_gv(
      const std::vector<double>& state_vec,
      const std::vector<double>& alge_vec,
      std::vector<std::vector<double> >& jacMat_gv);

  int jacFunc_fu(
      const std::vector<double>& state_vec,
      const std::vector<double>& alge_vec,
      std::vector<std::vector<double> >& jacMat_fu);

  int jacFunc_fv(
      const std::vector<double>& state_vec,
      const std::vector<double>& alge_vec,
      std::vector<std::vector<double> >& jacMat_fv);

  //int linkFunc( ChemicalElementaryDAE &chem_dae);

};

#endif
