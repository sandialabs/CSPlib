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


#ifndef MODEL_CHEM_ODE_TCHEM
#define MODEL_CHEM_ODE_TCHEM

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cstring>

#include "model.hpp"
#include "TChem.hpp"
#include "util.hpp"
#include "eigendecomposition_kokkos.hpp"

using ordinal_type = TChem::ordinal_type;
using real_type = TChem::real_type;
using real_type_1d_view = TChem::real_type_1d_view;
using real_type_2d_view = TChem::real_type_2d_view;
using real_type_3d_view = TChem::real_type_3d_view;

using real_type_0d_view_host = TChem::real_type_0d_view_host;
using real_type_1d_view_host = TChem::real_type_1d_view_host;
using real_type_2d_view_host = TChem::real_type_2d_view_host;


class ChemElemODETChem : public Model
{

 private:
   using host_device_type = typename Tines::UseThisDevice<TChem::host_exec_space>::type;
   using device_type      = typename Tines::UseThisDevice<TChem::exec_space>::type;
   using policy_type = typename TChem::UseThisTeamPolicy<TChem::exec_space>::type;

   // double _pressure;
  int _Nspec, _Nreac, _Nvars, _Nelem;

  std::vector< std::vector<double> > _state_db;

  int _nBatch;
  TChem::KineticModelData kmd;

  bool _run_on_device;
  // device
  KineticModelConstData<device_type> kmcd;
  real_type_2d_view _state;
  real_type_2d_view _RoPFor, _RoPRev;
  // real_type_3d_view _jac;
  real_type_3d_view _Smat;
  real_type_2d_view _rhs;

  KineticModelConstData<host_device_type> kmcd_host;
  real_type_2d_view_host _state_host;
  real_type_2d_view_host _RoPFor_host, _RoPRev_host;

  real_type_3d_view_host _Smat_host;
  real_type_2d_view_host _rhs_host;

  int _rhs_need_sync, _jac_need_sync, _smat_need_sync;
  int _RoP_need_sync, _state_need_sync;

  enum {
     NeedSyncToDevice = 1,
     NeedSyncToHost = -1,
     NoNeedSync = 0
   };

 public:

  real_type_3d_view _jac;
  real_type_3d_view_host _jac_host;


  //constructor:
  ChemElemODETChem( const std::string &mech_gas_file     ,
                    const std::string &thermo_gas_file   );

  ~ChemElemODETChem();


  //read data base from ignition problem
  // it populates _state view and _state_db
  void readIgnitionZeroDDataBaseFromFile(const std::string &filename,
                                    std::vector<std::string> &varnames,
                                    const ordinal_type &increment=1);

  int NumOfSpecies();
  int NumOfReactions();
  void getSpeciesNames(std::vector<std::string>& spec_name);

  void evalRoP(const ordinal_type& team_size=-1,
               const ordinal_type& vector_size=-1);

  // make a copy of state vector(Device) and convert to std::vector
  void getStateVector(std::vector<std::vector <double> >& state_db);

  void getSourceVector(std::vector<std::vector <double> >& source_db);

  void getJacMatrix(std::vector <std::vector
                      <std::vector <double> > >& jac_db);

  void getRoP(std::vector<std::vector <double> >& RoP);

  void getRoPDevice(real_type_2d_view& RoP);

  void getSmatrix(std::vector < std::vector
                          <std::vector <double> > >& Smatrixdb);

  // get smatrix device
  void getSmatrixDevice(real_type_3d_view& Smatrixdb);
  void getSourceVectorDevice(real_type_2d_view& rhs);

  int evalSourceVector();
  
  // model class does not all add new inputs because of virtual function
  void evalSourceVectorDevice(const ordinal_type& team_size=-1,
                              const ordinal_type& vector_size=-1);

  void evalSmatrix(const ordinal_type& team_size=-1,
                   const ordinal_type& vector_size=-1);

  int evalJacMatrix(unsigned int useJacAnl);

  int evalJacMatrix(const ordinal_type& useJacAnl, const bool& use_shared_workspace  );

  void evalJacMatrixDevice(const ordinal_type& useJacAnl,
                           const ordinal_type& team_size,
                           const ordinal_type&vector_size,
                           const bool& use_shared_workspace);

  int getNumOfElements() ;

  int getVarIndex(const std::string & var_name);

  void run_on_host(const bool & run_on_host);

  void getStateVector(std::vector<double>& state_vec);

  void getSourceVector(std::vector<double>& source_vec);

  void getJacMatrix(std::vector<std::vector<double> >& jmat);

  void getRoP(std::vector<double>& RoP);

  void getSmatrix(std::vector<std::vector<double> >& Smat);

  void setStateVectorDB(std::vector<std::vector <double> >& state_db);

  void getStateVectorDevice(real_type_2d_view& state_vector);

  void getJacMatrixDevice(real_type_3d_view& jac);


};

#endif
