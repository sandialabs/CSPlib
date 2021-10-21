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


#ifndef MODEL_CSP_CHEM_DAE_PFR_TCHEM
#define MODEL_CSP_CHEM_DAE_PFR_TCHEM

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cstring>

#include "model.hpp"

#include "TChem.hpp"

#include "eigendecomposition_kokkos.hpp"
#include "tools_tines.hpp"

using ordinal_type = TChem::ordinal_type;
using real_type = TChem::real_type;
using real_type_1d_view = TChem::real_type_1d_view;
using real_type_2d_view = TChem::real_type_2d_view;
using real_type_3d_view = TChem::real_type_3d_view;

using real_type_0d_view_host = TChem::real_type_0d_view_host;
using real_type_1d_view_host = TChem::real_type_1d_view_host;
using real_type_2d_view_host = TChem::real_type_2d_view_host;

using pfr_data_type = TChem::PlugFlowReactorData;
// using pfr_data_type_0d_view = TChem::pfr_data_type_0d_view;

class ChemElemDAE_PFR_TChem : public Model
{

 private:
  // double _pressure;
  int _Nelem, _Nspec, _Nreac,  _NspecSurf,_Nalge, _Nvars, _Nrate_of_processes;

  // std::vector<double> _state_vec  ;   // differential variables (or state vector)
  std::vector<double> _alge_vec   ;   // algebraic variables
  // std::vector<double> _source_vec ;   // rhs/source vector
  // std::vector<std::vector<double> > _jmat       ;   // jacobian matrix
  std::vector< std::vector<double> > _state_db;

  std::vector<std::string> _spec_name;

  using host_device_type = typename Tines::UseThisDevice<TChem::host_exec_space>::type;
  using device_type      = typename Tines::UseThisDevice<TChem::exec_space>::type;

  TChem::KineticModelData kmdSurf;
  KineticSurfModelConstData<device_type> kmcdSurf;
  KineticModelConstData<device_type> kmcd;

  //host TChem object
  KineticSurfModelConstData<host_device_type> kmcdSurf_host;
  KineticModelConstData<host_device_type> kmcd_host;

  real_type_2d_view _state;
  real_type_2d_view _siteFraction;
  real_type_1d_view _velocity;
  int _nBatch;
  real_type_2d_view _rhs;

  real_type_3d_view _Gu;
  real_type_3d_view _Fu;
  real_type_3d_view _Gv;
  real_type_3d_view _Fv;
  real_type_2d_view _RoPFor, _RoPRev, _RoPForSurf, _RoPRevSurf;
  real_type_3d_view _Smat, _Ssmat;

  pfr_data_type _pfrd;

 public:
  real_type_3d_view _jac;


  //constructor:
  ChemElemDAE_PFR_TChem( const std::string &mech_gas_file     ,
                         const std::string &mech_surf_file    ,
                         const std::string &thermo_gas_file   ,
                         const std::string &thermo_surf_file  );

  // Handling state vector
  void genRandStateVector();
  int getAlgeVector(std::vector<double>& alge_vec);
  int getSourceVector(std::vector<double>& source_vec);
  int getNumOfElements();

  //
  void readfromfileStateVector();
  void readfromfileStateVector(const std::string&,
                               const std::string&,
                               const std::string& );
  //
  void readDataBaseFromFile(const std::string &filename,
                            std::vector<std::string> &varnames);

  void getStateVector(std::vector<double>& state_vec);

  void getStateDBonHost(std::vector<std::vector <double> >& state_db);

  void getSourceDBonHost(std::vector<std::vector <double> >& source_db);

  void getRoPDBonHost(std::vector<std::vector <double> >& RoP);

  void getSmatrixDBonHost(std::vector < std::vector
                                            <std::vector <double> > >& Smatrixdb);
  void getJacDBonHost(std::vector < std::vector
                                            <std::vector <double> > >& jmat);

  void getSpeciesNames(std::vector<std::string>& spec_name);

  int evalSourceVector();

  void computeNumJac();
  void computeSacadoJacobian();

  int evalJacMatrix(unsigned int useJacAnl);

  // get rate of progress for gas and surface
  // call to TChem to compute rate of progress for
  // gas and surface
  int evalRoP();
  // convert rop view and convert to std vectors
  // if code runs on GPUs, data is obtaind from GPUs
  int getRoP(std::vector<double>& RoP);

  //compute Smat for gas and surface for pfr on devices
  int evalSmatrix();
  //make a copy from devices to host, convert view to std vector
  int getSmatrix(std::vector<std::vector<double>>& Smat);

  int verifySmatRoP(std::vector<std::vector<double> >&  Smat,
                    std::vector<double>& RoP, std::vector<double>& rhs);

  int getNumofGasReactions();
  int getNumofSurfaceReactions();
  int getNumofGasSpecies();
  int getNumofSurfaceSpecies();

  void getJacMatrix(std::vector<std::vector<double> >& jmat);
  void setPlugFlowReactor(const double Area, const double Pcat ) ;
  int getNumOfRateOfProcesses();
  void getJacobianDBonHost(std::vector <std::vector
                      <std::vector <double> > >& jac_db);

};

#endif
