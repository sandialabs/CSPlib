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


#include "interface_TChem.hpp"
#include "tools.hpp"
#include "util.hpp"
#include <Eigen/Dense>



//==========================================================================================

InterfaceTChem::InterfaceTChem(
                const std::string &mech_gas_file     ,
                const std::string &mech_surf_file    ,
                const std::string &thermo_gas_file   ,
                const std::string &thermo_surf_file  )

  {

    _nBatch = 1;
    // TChem::KineticModelDta kmd;
    const bool detail = false;
    TChem::     exec_space::print_configuration(std::cout, detail);
    TChem::host_exec_space::print_configuration(std::cout, detail);

    kmdSurf = TChem::KineticModelData(mech_gas_file,  thermo_gas_file,
                                      mech_surf_file, thermo_surf_file);

    kmcd = kmdSurf.createConstData<TChem::exec_space>(); // data struc with gas phase info
    kmcdSurf = kmdSurf.createConstSurfData<TChem::exec_space>();// data struc with surface phase info
    _Nvars = kmcd.nSpec + 3;// Number of variables in ODE part, Ys (nSpec) + density + Temp + vel
    _Nalge = kmcdSurf.nSpec; //Number of alebraic constrains, Zs
    // _Nelem do we need it?
    _Nspec = kmcd.nSpec;
    _Nreac = kmcd.nReac;
    // _spec_name
}

void InterfaceTChem::InitializeTChem()
{

    printf("Number of gas species: %d\n", kmcd.nSpec );
    printf("Number of surface species: %d\n", kmcdSurf.nSpec );
    printf("Number of gas reactions: %d\n", kmcd.nReac );
    printf("Number of surface reactions: %d\n", kmcdSurf.nReac );

}
//==========================================================================================
// InterfaceTChem::~InterfaceTChem(){
//   // delete &_pars;
// }

//==========================================================================================

void InterfaceTChem::ResetTChem() {
}

//==============================================================================
int  InterfaceTChem::init() {

  if (_Nvars == 0) {
    std::cout << "Model::init has Nvars = 0\n";
    exit(1);
  }

  _state_vec.clear();
  _state_vec.shrink_to_fit();
  _state_vec=std::vector<double>(_Nvars,0.0);

  _source_vec.clear();
  _source_vec.shrink_to_fit();
  _source_vec=std::vector<double>(_Nvars,0.0);

  _jmat.clear();
  _jmat.shrink_to_fit();
  _jmat = std::vector<std::vector<double> >(_Nvars, std::vector<double>(_Nvars,0.0));

  return(0);
}


//==========================================================================================
int InterfaceTChem::NumOfElements() {
  return _Nelem;
}

//==========================================================================================
int InterfaceTChem::NumOfSpecies() {
  return _Nspec;
}

//==========================================================================================
int InterfaceTChem::NumOfReactions() {
  return _Nreac;
}

//==========================================================================================
int InterfaceTChem::NumOfVariables() {
  return _Nvars;
}
int InterfaceTChem::NumOfAlgebraicConstraints() {
  return _Nalge;
}



//==========================================================================================
int InterfaceTChem::loadSpeciesNames() {

  return 0;
}

//==========================================================================================
int InterfaceTChem::getSpeciesNames(std::vector<std::string>& spec_name) {

  if( _spec_name.empty() ) {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"
             << "Vector of for species spe is empty.\n"
             << "Call InterfaceTChem::loadSpeciesNames to fill it out.\n";
    exit(1);
  }

  spec_name = _spec_name;
  return 0;
}

//==========================================================================================
// int InterfaceTChem::setSpeciesNames(std::vector<std::string>& spec_name) {
//   _spec_name=spec_name; // Filling up private array
//   return 0;
// }

#if 1
//==========================================================================================
// Optional tools to generate state vector with random number generator.
// But it should be collected from the users
void InterfaceTChem::genRandStateVector() {

  _state_vec[0] = 700. + (2300.-700.)*rand() / ( (double) RAND_MAX ) ;

  double dsum = 0.0 ;
  for (int i=1; i<_Nvars; i++) {
    _state_vec[i] = rand() / ( (double) RAND_MAX ) ;
    dsum += _state_vec[i];
  }

  for (int i = 1; i<_Nvars; i++ )
    _state_vec[i] /= dsum;

  return;
}

void InterfaceTChem::readfromfileStateVector(const std::string &inputFile,
                                             const std::string &inputFileSurf,
                                             const std::string &inputFilevelocity)
{

  /// input: state vectors: temperature, pressure and mass fraction
  _state = real_type_2d_view("state vector", _nBatch, TChem::Impl::getStateVectorSize(kmcd.nSpec));
  // input :: surface fraction vector, zk
  _siteFraction = real_type_2d_view("SiteFraction", _nBatch, kmcdSurf.nSpec);
  // input :: velocity variables : velocity for  a PRF
  _velocity = real_type_1d_view("Velocity", _nBatch);
  /// create a mirror view to store input from a file
  auto state_host = Kokkos::create_mirror_view(_state);

  /// create a mirror view to store input from a file
  auto siteFraction_host = Kokkos::create_mirror_view(_siteFraction);

  /// create a mirror view to store input from a file
  auto velocity_host = Kokkos::create_mirror_view(_velocity);

  auto state_host_at_0 = Kokkos::subview(state_host, 0, Kokkos::ALL());
  TChem::Test::readStateVector(inputFile, kmcd.nSpec, state_host_at_0);
  TChem::Test::cloneView(state_host);

  // read surface
  auto siteFraction_host_at_0 = Kokkos::subview(siteFraction_host, 0, Kokkos::ALL());
  TChem::Test::readSiteFraction(inputFileSurf, kmcdSurf.nSpec, siteFraction_host_at_0);
  TChem::Test::cloneView(siteFraction_host);

  // read velocity (velocity)
  TChem::Test::readSiteFraction(inputFilevelocity, 1, velocity_host);
  TChem::Test::cloneView(velocity_host);

  Kokkos::deep_copy(_state, state_host);
  Kokkos::deep_copy(_siteFraction, siteFraction_host);
  Kokkos::deep_copy(_velocity, velocity_host);


  const Impl::StateVector<real_type_1d_view_host> sv_at_i(kmcd.nSpec, state_host_at_0);
  const real_type t = sv_at_i.Temperature();
  printf("Temp %e\n", t );
  const real_type p = sv_at_i.Pressure();
  printf("pressure %e\n", p );
  const real_type density = sv_at_i.Density();
  printf("density %e\n", density );
  const real_type_1d_view_host Xc = sv_at_i.MassFractions();
  for (ordinal_type i=0;i<kmcd.nSpec;++i)
    printf("i %d, mass fraction %e\n", i, Xc(i) );
  const real_type vel = velocity_host(0);
  printf("velocity %f \n",  vel );

  for (ordinal_type i=0;i<kmcdSurf.nSpec;++i)
    printf("i %d, site fraction %e\n", i, siteFraction_host_at_0(i) );


  // order T,
  _state_vec.push_back(sv_at_i.Temperature());
  // Ys
  for (auto i = 0; i < kmcd.nSpec; i++) {
    _state_vec.push_back( Xc(i));
  }
  // , density, vel
  _state_vec.push_back(sv_at_i.Density());
  //vel
  _state_vec.push_back(velocity_host(0));

  printf("Done copy state vector\n" );

  for (auto i = 0; i < kmcdSurf.nSpec; i++) {
    _alge_vec.push_back(siteFraction_host_at_0(i));
  }



}

int InterfaceTChem::rhsFunc()
{

  /// output: rhs for Plug Flow rector with surface reactions
  const auto nEqns = kmcd.nSpec + kmcdSurf.nSpec + 3;
  printf("Number of Equation in DAE %d \n", nEqns );
  _rhs = real_type_2d_view("plugflowreactorRHS", _nBatch, nEqns  );

  TChem::PlugFlowReactorRHS::runDeviceBatch(_nBatch,
                                              //inputs
                                              _state, //gas
                                              _siteFraction,//surface
                                              _velocity, // PRF
                                              //ouputs
                                              _rhs,
                                              //data
                                              kmcd,
                                              kmcdSurf);

  const std::string outputFile = "plugflowreactorRHS.dat";

  printf("Write RHS at  %s \n", outputFile.c_str() );

  auto rhs_host = Kokkos::create_mirror_view(_rhs);
  Kokkos::deep_copy(rhs_host, _rhs);

  /// all values are same (print only the first one)
  {
  auto rhs_host_at_0 = Kokkos::subview(rhs_host, 0, Kokkos::ALL());
  TChem::Test::writeReactionRates(outputFile,
                                  nEqns,
                                  rhs_host_at_0);
  }


return(0);
}

int InterfaceTChem::rhsFuncH (std::vector<double>& source_vec)
{

  auto rhs_host = Kokkos::create_mirror_view(_rhs);
  Kokkos::deep_copy(rhs_host, _rhs);

  auto rhs_host_at_0 = Kokkos::subview(rhs_host, 0, Kokkos::ALL());

  TChem::convertToStdVector(source_vec, rhs_host_at_0);


  return(0);
}

int InterfaceTChem::jacFunc()
{
    printf("Number of variables %d\n", _Nvars);
    printf("Number of alegraic constrains %d\n",_Nalge );
    printf("Number of batches %d\n",_nBatch );

    const auto Nvars = kmcd.nSpec + 3;
    const auto Nalge = kmcdSurf.nSpec;

    _Gu = real_type_3d_view("Gu_PlugflowreactorJacobian", _nBatch, Nvars, Nvars );
    _Fu = real_type_3d_view("Fu_PlugflowreactorJacobian", _nBatch, Nalge, Nvars );
    _Gv = real_type_3d_view("Gv_PlugflowreactorJacobian", _nBatch, Nvars, Nalge );
    _Fv = real_type_3d_view("Fv_PlugflowreactorJacobian", _nBatch, Nalge, Nalge );

    TChem::PlugFlowReactorNumJacobian::
    runDeviceBatch(_nBatch,
                   //inputs
                   _state, //gas
                   _siteFraction,//surface
                   _velocity, // PRF
                  /// output
                   _Gu, _Fu, _Gv, _Fv,
                   kmcd,
                   kmcdSurf);

    {
      auto Gu_host = Kokkos::create_mirror_view(_Gu);
      Kokkos::deep_copy(Gu_host, _Gu);
      /// all values are same (print only the first one)
      auto Gu_host_at_0 = Kokkos::subview(Gu_host, 0, Kokkos::ALL(), Kokkos::ALL());

      std::string outputFile = "Gu_jacobian.dat";
      TChem::Test::write2DMatrix(outputFile,
                                 Nvars, Nvars,
                                Gu_host_at_0);
    }

    {
      /// create a mirror view of jacobian (output) to export a file
      auto Fu_host = Kokkos::create_mirror_view(_Fu);
      Kokkos::deep_copy(Fu_host, _Fu);
      /// all values are same (print only the first one)
      auto Fu_host_at_0 = Kokkos::subview(Fu_host, 0, Kokkos::ALL(), Kokkos::ALL());

      std::string outputFile = "Fu_jacobian.dat";
      TChem::Test::write2DMatrix(outputFile,
                                 Nalge, Nvars,
                                 Fu_host_at_0);
    }


   {
     auto Gv_host = Kokkos::create_mirror_view(_Gv);
     Kokkos::deep_copy(Gv_host, _Gv);
     /// all values are same (print only the first one)
     auto Gv_host_at_0 = Kokkos::subview(Gv_host, 0, Kokkos::ALL(), Kokkos::ALL());

     std::string outputFile = "Gv_jacobian.dat";
     TChem::Test::write2DMatrix(outputFile,
                                Nvars, Nalge,
                                Gv_host_at_0);
   }

   {
     auto Fv_host = Kokkos::create_mirror_view(_Fv);
     Kokkos::deep_copy(Fv_host, _Fv);
     /// all values are same (print only the first one)
     auto Fv_host_at_0 = Kokkos::subview(Fv_host, 0, Kokkos::ALL(), Kokkos::ALL());

     std::string outputFile = "Fv_jacobian.dat";
     TChem::Test::write2DMatrix(outputFile,
                                Nalge, Nalge,
                                Fv_host_at_0);
   }




    return(0);
}

void InterfaceTChem::readfromfileStateVector(){

  // Load state vector
  //const int Neq = _NspecSurf + _Nspec + 3;
  double temp;
  std::cout << "\n-- reading State vector from file  --"<<"\n\n";
  std::ifstream ixfs("pfrSolutionOneP.dat");
  std::vector<double> vec ; // tym, T, u_1, ..., u_nspec
  if (ixfs.is_open()){
    while (ixfs >> temp){
      vec.push_back(temp);
    }
  } else {
    std::cerr << " driver_index.cpp: cannot open file state_vec.txt\n";
    exit(-1);
  }
  ixfs.close();

  const int Nvars = vec.size(); // total number of variables incluing surface species
  //printf("Number of variables %e\n",Nvars );
  std::cout << "Read number of variables " << Nvars << std::endl;

  _Nvars = Nvars -  _NspecSurf -1; // fix number of variables in ODE part

  for (int i = 0; i< Nvars -  _NspecSurf; i++ )
    _state_vec[i] = vec[i+1] ;

  for (int i = Nvars -  _NspecSurf; i<Nvars; i++ ){
      _alge_vec.push_back(vec[i+1]);
  }



}

//==============================================================================
int InterfaceTChem::setStateVector(const std::vector<double>& state_vec) {
  _state_vec = state_vec;
  return(0);
}
//==============================================================================
int InterfaceTChem::getStateVector(std::vector<double>& state_vec) {
  if( !_state_vec.empty() ) {
    state_vec = _state_vec;
  } else {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"<<"State Vector is empty!"<<std::endl;
    exit(1);
  }
  return(0);
}
//==============================================================================
int InterfaceTChem::getAlgeVector(std::vector<double>& alge_vec) {
  if( !_alge_vec.empty() ) {
    alge_vec = _alge_vec;
  } else {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"<<"Alge Vector is empty!"<<std::endl;
    exit(1);
  }
  return(0);
}

#endif

//==========================================================================================




int InterfaceTChem::jacFuncH(
    std::vector<std::vector<double> >& jacMat_gu,
    std::vector<std::vector<double> >& jacMat_gv,
    std::vector<std::vector<double> >& jacMat_fu,
    std::vector<std::vector<double> >& jacMat_fv)
    {

    auto Fu_host = Kokkos::create_mirror_view(_Fu);
    Kokkos::deep_copy(Fu_host, _Fu);
    /// all values are same (print only the first one)
    auto Fu_host_at_0 = Kokkos::subview(Fu_host, 0, Kokkos::ALL(), Kokkos::ALL());

    TChem::convertToStdVector(jacMat_fu, Fu_host_at_0);

    auto Fv_host = Kokkos::create_mirror_view(_Fv);
    Kokkos::deep_copy(Fv_host, _Fv);
    /// all values are same (print only the first one)
    auto Fv_host_at_0 = Kokkos::subview(Fv_host, 0, Kokkos::ALL(), Kokkos::ALL());

    TChem::convertToStdVector(jacMat_fv, Fv_host_at_0);

    auto Gv_host = Kokkos::create_mirror_view(_Gv);
    Kokkos::deep_copy(Gv_host, _Gv);
    /// all values are same (print only the first one)
    auto Gv_host_at_0 = Kokkos::subview(Gv_host, 0, Kokkos::ALL(), Kokkos::ALL());

    TChem::convertToStdVector(jacMat_gv, Gv_host_at_0);

    auto Gu_host = Kokkos::create_mirror_view(_Gu);
    Kokkos::deep_copy(Gu_host, _Gu);
    /// all values are same (print only the first one)
    auto Gu_host_at_0 = Kokkos::subview(Gu_host, 0, Kokkos::ALL(), Kokkos::ALL());

    TChem::convertToStdVector(jacMat_gu, Gu_host_at_0);


    return(0);
}

//==========================================================================================
int InterfaceTChem::jacFunc_gu(
    const std::vector<double>& state_vec,
    const std::vector<double>& alge_vec,
    std::vector<std::vector<double> >& jacMat_gu) {

  return(0);
}

//==========================================================================================
int InterfaceTChem::jacFunc_gv(
    const std::vector<double>& state_vec,
    const std::vector<double>& alge_vec,
    std::vector<std::vector<double> >& jacMat_gv) {

  return(0);
}

//==========================================================================================
int InterfaceTChem::jacFunc_fu(
    const std::vector<double>& state_vec,
    const std::vector<double>& alge_vec,
    std::vector<std::vector<double> >& jacMat_fu) {

  return(0);
}

//==========================================================================================
int InterfaceTChem::jacFunc_fv(
    const std::vector<double>& state_vec,
    const std::vector<double>& alge_vec,
    std::vector<std::vector<double> >& jacMat_fv) {

  return(0);
}

#if 0
int InterfaceTChem::linkFunc( ChemicalElementaryDAE &chem_dae  ) {
  ChemicalElementaryDAE ma(
    std::function<int(const std::vector<double>&, const std::vector<double>&, std::vector<double>&)> (std::forward(this->rhsFunc)),
    std::function<int(const std::vector<double>&, const std::vector<double>&, std::vector<std::vector<double> >&, std::vector<std::vector<double> >&
    , std::vector<std::vector<double> >&, std::vector<std::vector<double> >&)> (std::forward(this->jacFunc)),
    std::function<int(const std::vector<double>&, const std::vector<double>&, std::vector<std::vector<double> >&)> (std::forward(this->jacFunc_gu)),
    std::function<int(const std::vector<double>&, const std::vector<double>&, std::vector<std::vector<double> >&)> (std::forward(this->jacFunc_gv)),
    std::function<int(const std::vector<double>&, const std::vector<double>&, std::vector<std::vector<double> >&)> (std::forward(this->jacFunc_fu)),
    std::function<int(const std::vector<double>&, const std::vector<double>&, std::vector<std::vector<double> >&)> (std::forward(this->jacFunc_fv))
  )

  return(0);
}
#endif
