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


#ifndef MODEL_CHEM_TCSTR_TCHEM
#define MODEL_CHEM_TCSTR_TCHEM

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cstring>

#include "model.hpp"
#include "TChem.hpp"
#include "tools_tines.hpp"
#include "eigendecomposition_kokkos.hpp"


using ordinal_type = TChem::ordinal_type;
using real_type = TChem::real_type;
using real_type_1d_view = TChem::real_type_1d_view;
using real_type_2d_view = TChem::real_type_2d_view;
using real_type_3d_view = TChem::real_type_3d_view;

using real_type_0d_view_host = TChem::real_type_0d_view_host;
using real_type_1d_view_host = TChem::real_type_1d_view_host;
using real_type_2d_view_host = TChem::real_type_2d_view_host;


class ChemElemTCSTR_TChem : public Model
{

 private:
  // double _pressure;

  int _Nelem, _Ngas_species, _Ngas_reactions, _Nsurface_reactions, _Nsurface_species;
  int  _Nvars, _Nalgebraic_constraints, _Ntotal_variables;
  int _Nsurface_equations, _Nrate_of_processes, _Nspecies_p_temperature;

  std::vector< std::vector<double> > _state_db;

  using host_device_type = typename Tines::UseThisDevice<TChem::host_exec_space>::type;
  using device_type      = typename Tines::UseThisDevice<TChem::exec_space>::type;

  TChem::KineticModelData kmdSurf;
  // device TChem object
  KineticSurfModelConstData<device_type> kmcdSurf;
  KineticModelConstData<device_type> kmcd;

  //host TChem object
  KineticSurfModelConstData<host_device_type> kmcdSurf_host;
  KineticModelConstData<host_device_type> kmcd_host;

  real_type_2d_view _siteFraction;

  TransientContStirredTankReactorData<device_type> _cstr;

  real_type_2d_view _state;
  real_type_2d_view _RoPFor, _RoPRev, _RoPForSurf, _RoPRevSurf;
  real_type_3d_view _Smat, _Ssmat, _Cmat;
  real_type_2d_view _Sconv;
  int _nBatch;
  real_type_2d_view _rhs;

 public:
  real_type_3d_view _jac;

  //
  //constructor:
  ChemElemTCSTR_TChem( const std::string &mech_gas_file     ,
                         const std::string &mech_surf_file    ,
                         const std::string &thermo_gas_file   ,
                         const std::string &thermo_surf_file  ,
                         const int& Nalgebraic_constraints=0 );



  //
  void readDataBaseFromFile(const std::string &filename,
                            std::vector<std::string> &varnames);

  void getStateVector(std::vector<double>& state_vec);

  void getSourceVector(std::vector<double>& source_vec);

  void getJacMatrix(std::vector<std::vector<double> >& jmat);

  void getSpeciesNames(std::vector<std::string>& spec_name);

  void evalRoP();
  int getNumOfRateOfProcesses();
  //get only first index
  void getRoP(std::vector<double>& RoP);

  // make a copy of state vector(Device) and convert to std::vector
  void getStateDBonHost(std::vector<std::vector <double> >& state_db);

  void getSourceDBonHost(std::vector<std::vector <double> >& source_db);

  void getJacobianDBonHost(std::vector <std::vector
                      <std::vector <double> > >& jac_db);

  void getRoPDBonHost(std::vector<std::vector <double> >& RoP);

  void getSmatrixDBonHost(std::vector < std::vector
                          <std::vector <double> > >& Smatrixdb);
  //
  int evalSourceVector();


  void evalSmatrix();
  void getSmatrix(std::vector<std::vector<double> >& Smat);

  int evalJacMatrix(unsigned int useJacAnl);

  void computeNumJac();

  void computeSacadoJacobian();

  int verifySmatRoP(std::vector<std::vector<double> >&  Smat,
                    std::vector<double>& RoP, std::vector<double>& rhs);

  int getNumOfElements();
  int getNumofGasReactions();
  int getNumofSurfaceReactions();
  int getNumofGasSpecies();
  int getNumofSurfaceSpecies();

  void setCSTR(const std::string& input_condition_file_name,
               const double& mdotIn,  const double& Vol,
               const double& Acat, const bool& isoThermic);

};

#endif
