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


#include "chem_elem_DAE_PFR_TChem.hpp"
#include "tools.hpp"
#include "util.hpp"

// Use (void) to silent unused warnings.
#define assertm(exp, msg) assert(((void)msg, exp))

//==========================================================================================
ChemElemDAE_PFR_TChem::ChemElemDAE_PFR_TChem(
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

    kmdSurf = TChem::KineticModelData(mech_gas_file, thermo_gas_file,
                                      mech_surf_file, thermo_surf_file);
    // create tchem object in device
    kmcd = TChem::createGasKineticModelConstData<device_type>(kmdSurf);// data struc with gas phase info
    kmcdSurf = TChem::createSurfaceKineticModelConstData<device_type>(kmdSurf);// data struc with surface phase info

    // create also a copy for host
    kmcd_host = TChem::createGasKineticModelConstData<host_device_type>(kmdSurf); // data struc with gas phase info
    kmcdSurf_host = TChem::createSurfaceKineticModelConstData<host_device_type>(kmdSurf);// data struc with surface phase info

    _Nvars = kmcd.nSpec + 3;// Number of variables in ODE part, Ys (nSpec) + density + Temp + vel
    _Nalge = kmcdSurf.nSpec; //Number of alebraic constrains, Zs
    // _Nelem do we need it?
    _Nspec = kmcd.nSpec;
    _Nreac = kmcd.nReac;
    _Nelem = kmcd.NumberofElementsGas;

    _Nrate_of_processes=  2*kmcd.nReac + 2*kmcdSurf.nReac;
    setNumOfVariables(_Nvars);// in base class

    printf("Number of gas species: %d\n", _Nspec );
    printf("Number of surface species: %d\n", kmcdSurf.nSpec );
    printf("Number of gas reactions: %d\n", _Nreac );
    printf("Number of surface reactions: %d\n", kmcdSurf.nReac );
    printf("Number of elements %d\n", _Nelem);

    // _spec_name
}

int ChemElemDAE_PFR_TChem::getNumofGasSpecies(){
    return(_Nspec);
}
int ChemElemDAE_PFR_TChem::getNumofSurfaceSpecies(){
    return(kmcdSurf.nSpec);
}
int ChemElemDAE_PFR_TChem::getNumOfElements() {
    return _Nelem;
}
void ChemElemDAE_PFR_TChem::readfromfileStateVector(const std::string &inputFile,
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
  // printf("Temp %e\n", t );
  const real_type p = sv_at_i.Pressure();
  // printf("pressure %e\n", p );
  const real_type density = sv_at_i.Density();
  // printf("density %e\n", density );
  const real_type_1d_view_host Xc = sv_at_i.MassFractions();
  // for (ordinal_type i=0;i<kmcd.nSpec;++i)
  //   printf("i %d, mass fraction %e\n", i, Xc(i) );
  // const real_type vel = velocity_host_at_0(0);
  // printf("velocity %f \n",  vel );
  //
  // for (ordinal_type i=0;i<kmcdSurf.nSpec;++i)
  //   printf("i %d, site fraction %e\n", i, siteFraction_host_at_0(i) );


  // order T,
  _state_vec.clear();
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
  _alge_vec.clear();
  for (auto i = 0; i < kmcdSurf.nSpec; i++) {
    _alge_vec.push_back(siteFraction_host_at_0(i));
  }


}
void ChemElemDAE_PFR_TChem::readDataBaseFromFile(const std::string &filename,
                                                std::vector<std::string> &varnames)
{
     //read a solution from a PFR:
     //this data is from TChem++, number of row is position
     //columns solution at each position
     // iter t, dt density, pressure, Temp [K], mass fraction,  site fraction and velocity

     double atposition;
     const int TotalEq = kmcd.nSpec + kmcdSurf.nSpec + 3 + 4;

     std::vector< std::vector<double> > pfrdb;

     std::ifstream ixfs(filename);
     if (ixfs.is_open()) {
       // read header of file and save variable name in vector
       std::string line;
       std::getline(ixfs, line);
       std::istringstream iss(line);
       std::string delimiter = " ";

       size_t pos = 0;
       std::string token;
       while ((pos = line.find(delimiter)) != std::string::npos) {
         varnames.push_back(line.substr(0, pos));
         line.erase(0, pos + delimiter.length());
       }
       varnames.push_back(line);

       while(ixfs >> atposition){
         std::vector<double>vec(TotalEq,0.0); //
         vec[0] = atposition;
         for (int i=1; i<TotalEq; i++)  {
           ixfs >> vec[i];
         }

         pfrdb.push_back(vec);
       }
     } else {
         std::cerr << " chem_elem_DAE_PFR_TChem.cpp: cannot open file "+ filename +"\n";
         exit(-1);
      }

    ixfs.close();

    _nBatch = pfrdb.size();

#if 0
     for (int sp = 0; sp < _nBatch; sp++) {
       for (int i = 1; i < (pfrdb[0]).size() ; i++) {
         printf("value sp %d i %d %f\n", sp, i ,pfrdb[sp][i]   );
       }
     }
#endif

     printf("Reading data from  PFR solution: # state vectors %lu # Variables (including site fraction) %lu\n",pfrdb.size() ,(pfrdb[0]).size());

     /// input: state vectors: temperature, pressure and mass fraction
     const int stateVectorLen = TChem::Impl::getStateVectorSize(kmcd.nSpec);
     printf("len state vector %d\n",stateVectorLen );
     _state = real_type_2d_view("state vector", _nBatch, stateVectorLen );
     // input :: surface fraction vector, zk
     _siteFraction = real_type_2d_view("SiteFraction", _nBatch, kmcdSurf.nSpec);
     // input :: velocity variables : velocity for  a PRF
     _velocity = real_type_1d_view("Velocity", _nBatch);

     /// create a mirror view to store input from a file
     auto state_host = Kokkos::create_mirror_view(_state);
     auto siteFraction_host = Kokkos::create_mirror_view(_siteFraction);
     auto velocity_host = Kokkos::create_mirror_view(_velocity);


     for (int sp = 0; sp < _nBatch; sp++) {
       /* state vector density, pressure, Temp, mass fraction*/
       for (int i = 0; i < stateVectorLen  ; i++) {
         state_host(sp,i) = pfrdb[sp ][i+3];
         // printf("state i %d, %f, %f\n", i, pfrdb[sp + position][i], state_host(sp,i) );
       }
       /* site fraction */
       for (int k = 0; k < kmcdSurf.nSpec ; k++) {
         siteFraction_host(sp,k) = pfrdb[sp ][k + stateVectorLen + 3];
         // printf("site i %d, %f % f\n", k, pfrdb[sp + position][k+ stateVectorLen + 1], siteFraction_host(sp,k) );
       }

       velocity_host(sp) = pfrdb[sp ][stateVectorLen + kmcdSurf.nSpec + 3];
       // printf("vel i %d, %f, %f\n", stateVectorLen + kmcdSurf.nSpec + 1, pfrdb[sp + position][stateVectorLen + kmcdSurf.nSpec + 1],velocity_host(sp,0) );
     }

     Kokkos::deep_copy(_state, state_host);
     Kokkos::deep_copy(_siteFraction, siteFraction_host);
     Kokkos::deep_copy(_velocity, velocity_host);

     _state_db = std::vector<std::vector<double>>(_nBatch,
                 std::vector<double>(_Nvars,0.0));

     for (int i = 0; i < _nBatch; i++) {
       auto state_host_at_i = Kokkos::subview(state_host, i, Kokkos::ALL());

       const Impl::StateVector<real_type_1d_view_host> sv_at_i(kmcd.nSpec, state_host_at_i);
       _state_db[i][0] = sv_at_i.Temperature();
       const real_type_1d_view_host Ys = sv_at_i.MassFractions();

       for (int j = 1; j < _Nvars-2; j++) {
         _state_db[i][j] = Ys(j-1);
       }
       _state_db[i][_Nvars-1] = sv_at_i.Density();
       _state_db[i][_Nvars-2] = _velocity(i);

     }

}
void ChemElemDAE_PFR_TChem::getStateVector(std::vector<double>& state_vec)
{
  auto state_at_0 = Kokkos::subview(_state, 0, Kokkos::ALL());
  auto state_host_at_0 = Kokkos::create_mirror_view(state_at_0);
  Kokkos::deep_copy(state_host_at_0, state_at_0 );
  const Impl::StateVector<real_type_1d_view_host> sv_at_i(kmcd.nSpec, state_host_at_0);

  /// create a mirror view to store input from a file
  auto velocity_host = Kokkos::create_mirror_view(_velocity);
  Kokkos::deep_copy(velocity_host, _velocity );

  // auto siteFraction_at_0  = Kokkos::subview(_siteFraction, 0, Kokkos::ALL());
  // auto siteFraction_host_at_0 = Kokkos::create_mirror_view(siteFraction_at_0);
  // Kokkos::deep_copy(siteFraction_host_at_0, _siteFraction);

  const real_type t = sv_at_i.Temperature();
  const real_type_1d_view_host Ys = sv_at_i.MassFractions();

  _state_vec.clear();
  _state_vec.push_back(sv_at_i.Temperature());

  // Ys
  for (auto i = 0; i < kmcd.nSpec; i++)
   _state_vec.push_back( Ys(i));
  _state_vec.push_back(sv_at_i.Density());
  _state_vec.push_back(velocity_host(0));
  state_vec = _state_vec;

}

int ChemElemDAE_PFR_TChem::getSourceVector(std::vector<double>& source_vec)
{
  _source_vec.clear();
  _source_vec.shrink_to_fit();
  _source_vec=std::vector<double>(_Nvars,0.0);

   auto rhs_at_0 = Kokkos::subview(_rhs, 0, Kokkos::ALL());
   auto rhs_host_at_0 = Kokkos::create_mirror_view(rhs_at_0);
   Kokkos::deep_copy(rhs_host_at_0, rhs_at_0);

   //rhs: include T mass fraction, density, velocity and algebraic constraints
   // std::vector<double> rhs_std(rhs_host.extent(0),0);
   // copy only T, mass fraction, density and velocity
   // cps analysis does not incluide algebraic constrain
   for (int i = 0; i < _source_vec.size(); i++) {
      _source_vec[i] = rhs_host_at_0(i);
  source_vec = _source_vec;
}

  return(0);
}

void ChemElemDAE_PFR_TChem::getStateDBonHost(std::vector<std::vector <double> >& state_db)
{
     state_db.clear();
     state_db = _state_db;
}

void ChemElemDAE_PFR_TChem::getSourceDBonHost(std::vector<std::vector <double> >& source_db)
{

    std::vector<std::vector<double>> source_long;
    source_long = std::vector<std::vector<double>>(_nBatch,
    std::vector<double>(_rhs.extent(1),0.0));
    TChem::convertToStdVector(source_long, _rhs);

    source_db.clear();
    source_db = std::vector<std::vector<double>>(_nBatch,std::vector<double>(_Nvars,0.0));
    for (int i = 0; i < _nBatch; i++) {
      for (int j = 0; j < _Nvars; j++) {
        //copy first _Nvars, rest of values are zero
        source_db[i][j] = source_long[i][j];
      }
    }

}

void ChemElemDAE_PFR_TChem::getJacobianDBonHost(std::vector < std::vector
                                          <std::vector <double> > >& jac_db)
{
    jac_db.clear();
    jac_db = std::vector< std::vector< std::vector< double > > >
    (_jac.extent(0), std::vector<std::vector<double> > (_jac.extent(1),
     std::vector<double>(_jac.extent(2),0)));

    TChem::convertToStdVector(jac_db, _jac);
  }

void ChemElemDAE_PFR_TChem::getRoPDBonHost(std::vector<std::vector <double> >& RoP)
{

     RoP.clear();
     RoP = std::vector<std::vector<double>>(
       _nBatch,std::vector<double>(_Nrate_of_processes,0.0));

     std::vector<std::vector<double > > RoPFor;
     std::vector<std::vector<double > > RoPRev;

     TChem::convertToStdVector(RoPFor, _RoPFor);
     TChem::convertToStdVector(RoPRev, _RoPRev);

     std::vector<std::vector<double > > RoPForSurf;
     std::vector<std::vector<double > > RoPRevSurf;

     TChem::convertToStdVector(RoPForSurf, _RoPForSurf);
     TChem::convertToStdVector(RoPRevSurf, _RoPRevSurf);

     for (int i = 0; i < _nBatch; i++) {

       for (int j = 0; j < _Nreac; j++) {
         RoP[i][j] = RoPFor[i][j];
         RoP[i][j + _Nreac] = -RoPRev[i][j];
       }

       for (int j = 0; j < kmcdSurf.nReac; j++) {
         RoP[i][j + 2*_Nreac] = RoPForSurf[i][j];
         RoP[i][j + 2*_Nreac + kmcdSurf.nReac ] = -RoPRevSurf[i][j];
       }

     }

}

void ChemElemDAE_PFR_TChem::getSmatrixDBonHost(std::vector < std::vector
                                          <std::vector <double> > >& Smatrixdb)
{

     std::vector< std::vector< std::vector<double> > >
     Smat(_Smat.extent(0), std::vector<std::vector<double> > (_Smat.extent(1),
      std::vector<double>(_Smat.extent(2),0)));

     TChem::convertToStdVector(Smat, _Smat);

     std::vector< std::vector< std::vector<double> > >
     Ssmat(_Ssmat.extent(0), std::vector<std::vector<double> > (_Ssmat.extent(1),
      std::vector<double>(_Ssmat.extent(2),0)));

    TChem::convertToStdVector(Ssmat, _Ssmat);

    Smatrixdb.clear();
    Smatrixdb =
    std::vector< std::vector< std::vector<double> > >
    (_nBatch, std::vector<std::vector<double> > (_Nvars,
    std::vector<double>(_Nrate_of_processes,0)));

    Smatrixdb.resize(_nBatch);
    for (int k = 0; k < _nBatch; k++) {
      Smatrixdb[k].resize(_Nvars);
      for (int i=0; i< _Nvars ; i++) {
        Smatrixdb[k][i].resize(_Nrate_of_processes);

         for (int j = 0; j < _Nreac; j++) {
           Smatrixdb[k][i][j] = Smat[k][i][j];
           Smatrixdb[k][i][j + _Nreac] = Smat[k][i][j];
         }

         for (int j = 0; j < kmcdSurf.nReac; j++) {
           Smatrixdb[k][i][j + 2*_Nreac] = Ssmat[k][i][j];
           Smatrixdb[k][i][j + 2*_Nreac + kmcdSurf.nReac ] = Ssmat[k][i][j];
         }

       }
     }


}

int ChemElemDAE_PFR_TChem::evalSourceVector()
{

  /// output: rhs for Plug Flow rector with surface reactions
  // rhs includes alebraic constrains
  const auto nEqns = kmcd.nSpec + kmcdSurf.nSpec + 3;
  // printf("Number of Equation in DAE %d \n", nEqns );
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
                                              kmcdSurf,
                                              _pfrd);
return(0);
}

void ChemElemDAE_PFR_TChem::getSpeciesNames(std::vector<std::string>& spec_name)
{
    spec_name.clear();
    // get species name from device and make a copy in host
    const auto speciesNamesHost =kmcd_host.speciesNames;
    //make a copy in a std vector
    for (int k = 0; k < speciesNamesHost.extent(0); k++)
      spec_name.push_back(&speciesNamesHost(k,0));
}

int ChemElemDAE_PFR_TChem::evalJacMatrix(unsigned int useJacAnl)
{
  if (useJacAnl == 1 ) {
      printf(" Using Numerical Jacobian\n");
      computeNumJac();
  } else {
      printf(" Using Sacado Analytical Jacobian\n");
      computeSacadoJacobian();
  }
  return (0);
}

void ChemElemDAE_PFR_TChem::computeNumJac()
{
  const auto Ntotal_variables =   _Nvars + _Nalge;
  real_type_3d_view jac_long ("whole jacobian", _nBatch, Ntotal_variables, Ntotal_variables );
  real_type_2d_view fac("fac", _nBatch, Ntotal_variables); // this variables needs to be Ntotal length

  TChem::PlugFlowReactorNumJacobian::
                  runDeviceBatch(-1,-1,
                                 _nBatch,//inputs
                                 _state, //gas
                                 _siteFraction,//surface
                                 _velocity,
                                 jac_long,
                                 fac,
                                 kmcd,
                                 kmcdSurf,
                                 _pfrd);

    _jac = real_type_3d_view ("readuced jac", _nBatch, _Nvars, _Nvars );

    ComputeReducedJacobian<TChem::exec_space>
    ::runBatch(jac_long, _jac, _Nvars, _Nalge );
}

void ChemElemDAE_PFR_TChem::computeSacadoJacobian()
{

  const auto Ntotal_variables =   _Nvars + _Nalge;
  real_type_3d_view jac_long ("whole jacobian", _nBatch, Ntotal_variables, Ntotal_variables );

  const auto exec_space_instance = TChem::exec_space();
  using policy_type =
      typename TChem::UseThisTeamPolicy<TChem::exec_space>::type;
  const ordinal_type level = 1;

  const ordinal_type per_team_extent =
  TChem::PlugFlowReactorSacadoJacobian::getWorkSpaceSize(kmcd,kmcdSurf); ///

  const ordinal_type per_team_scratch =
  Scratch<real_type_1d_view>::shmem_size(per_team_extent);

  policy_type policy(exec_space_instance, _nBatch, Kokkos::AUTO());
  policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

  TChem::PlugFlowReactorSacadoJacobian
       ::runDeviceBatch(policy,
                        _state, //gas
                        _siteFraction,//surface
                        _velocity,
                        jac_long,
                        kmcd,
                        kmcdSurf,
                        _pfrd);

  _jac = real_type_3d_view ("readuced jac", _nBatch, _Nvars, _Nvars );

  ComputeReducedJacobian<TChem::exec_space>
  ::runBatch(jac_long, _jac, _Nvars, _Nvars );

}

void ChemElemDAE_PFR_TChem::getJacMatrix(std::vector<std::vector<double> >& jmat){
     _jmat.clear();
     _jmat.shrink_to_fit();
     _jmat = std::vector<std::vector<double>>(_Nvars,std::vector<double>(_Nvars,0.0));

     auto jac_at_0 = Kokkos::subview(_jac, 0, Kokkos::ALL(), Kokkos::ALL());
     auto jac_host_at_0 = Kokkos::create_mirror_view(jac_at_0);
     Kokkos::deep_copy(jac_host_at_0, jac_at_0);

     /// all values are same (print only the first one)
     TChem::convertToStdVector(_jmat, jac_host_at_0);
     jmat = _jmat;
}


//==============================================================================
int ChemElemDAE_PFR_TChem::getAlgeVector(std::vector<double>& alge_vec)
{
  if( !_alge_vec.empty() ) {
    alge_vec = _alge_vec;
  } else {
    std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"<<"Alge Vector is empty!"<<std::endl;
    exit(1);
  }
  return(0);
}

int ChemElemDAE_PFR_TChem::getRoP(std::vector<double>& RoP)
    {
    // gas forward rate of progress
    auto RoPFor_at_0 = Kokkos::subview(_RoPFor, 0, Kokkos::ALL());
    auto RoPFor_host_at_0 = Kokkos::create_mirror_view(RoPFor_at_0);
    Kokkos::deep_copy(RoPFor_host_at_0, RoPFor_at_0);

    // gas reverse rate of progress
    auto RoPRev_at_0 = Kokkos::subview(_RoPRev, 0, Kokkos::ALL());
    auto RoPRev_host_at_0 = Kokkos::create_mirror_view(RoPRev_at_0);
    Kokkos::deep_copy(RoPRev_host_at_0, RoPRev_at_0);

    //surface forward rate of progress
    auto RoPForSurf_at_0 = Kokkos::subview(_RoPForSurf, 0, Kokkos::ALL());
    auto RoPForSurf_host_at_0 = Kokkos::create_mirror_view(RoPForSurf_at_0);
    Kokkos::deep_copy(RoPForSurf_host_at_0, RoPForSurf_at_0);

    auto RoPRevSurf_at_0 = Kokkos::subview(_RoPRevSurf, 0, Kokkos::ALL());
    auto RoPRevSurf_host_at_0 = Kokkos::create_mirror_view(RoPRevSurf_at_0);
    Kokkos::deep_copy(RoPRevSurf_host_at_0, RoPRevSurf_at_0);


    for (int j = 0; j < kmcd.nReac; j++) {
      RoP[j] = RoPFor_host_at_0(j);
      RoP[j + kmcd.nReac] = -RoPRev_host_at_0(j);
    }

    for (int j = 0; j < kmcdSurf.nReac; j++) {
      RoP[j + 2*kmcd.nReac] = RoPForSurf_host_at_0(j);
      RoP[j + 2*kmcd.nReac + kmcdSurf.nReac ] = -RoPRevSurf_host_at_0(j);
    }



    return(0);
    }

int ChemElemDAE_PFR_TChem::evalRoP()
    {
    //compute rate of progress for gas phase
    const auto Nrg   = kmcd.nReac; // number of gas reactions
    const auto Nrs   = kmcdSurf.nReac; // numger of surface reactions

    _RoPFor = real_type_2d_view("Gas_Forward_RateOfProgess", _nBatch, Nrg );
    _RoPRev = real_type_2d_view("Gas_Reverse_RateOfProgess", _nBatch, Nrg);

    _RoPForSurf = real_type_2d_view("Surface_Forward_RateOfProgess", _nBatch, Nrs );
    _RoPRevSurf = real_type_2d_view("Surface_Reverse_RateOfProgess", _nBatch, Nrs);

    TChem::RateOfProgress::
           runDeviceBatch(_nBatch,
                          _state, //gas
                           //output,
                          _RoPFor, // Forward rate of progess
                          _RoPRev, // reverse rate of progess
                          kmcd);

    TChem::RateOfProgressSurface::
           runDeviceBatch(_nBatch,
                          _state, //gas
                          _siteFraction,//surface
                          //output,
                          _RoPForSurf, // Forward rate of progress surface phase
                          _RoPRevSurf, // reverse rate of progess surface phase
                          kmcd,
                          kmcdSurf);



    return(0);
}

int ChemElemDAE_PFR_TChem::getSmatrix(std::vector<std::vector<double> >& Smat)
    {

    auto Smat_at_0 = Kokkos::subview(_Smat, 0, Kokkos::ALL(), Kokkos::ALL());
    auto Smat_host_at_0 = Kokkos::create_mirror_view(Smat_at_0);
    Kokkos::deep_copy(Smat_host_at_0, Smat_at_0);

    auto Ssmat_at_0 = Kokkos::subview(_Ssmat, 0, Kokkos::ALL(), Kokkos::ALL());
    auto Ssmat_host_at_0 = Kokkos::create_mirror_view(Ssmat_at_0);
    Kokkos::deep_copy(Ssmat_host_at_0, Ssmat_at_0);

    for (int i=0; i< _Nvars ; i++) {
      for (int j = 0; j < kmcd.nReac; j++) {
        Smat[i][j] = Smat_host_at_0(i,j);
        Smat[i][j + kmcd.nReac] = Smat_host_at_0(i,j);
      }

      for (int j = 0; j < kmcdSurf.nReac; j++) {
        Smat[i][j + 2*kmcd.nReac] = Ssmat_host_at_0(i,j);
        Smat[i][j + 2*kmcd.nReac+ kmcdSurf.nReac ] = Ssmat_host_at_0(i,j);
      }
    }

    return(0);
    }
//
int ChemElemDAE_PFR_TChem::verifySmatRoP(std::vector<std::vector<double> >&  Smat,
                                        std::vector<double>& RoP, std::vector<double>& rhs)
{

  int nrow_A = (int) Smat.size();
  int ncol_A = (int) Smat[0].size();
  int nrow_B = (int) RoP.size();
  // int ncol_B = 1;

  assertm (ncol_A == nrow_B, "mat times mat inconsistency");

  for (int i=0; i<nrow_A; i++) { // loop over rows
      rhs[i] = 0.0;
      for (int j=0; j<nrow_B; j++){
        rhs[i] += Smat[i][j] * RoP[j];
      }
  }

  return(0);
}


int ChemElemDAE_PFR_TChem::evalSmatrix()
    {

    const auto Nrg   = kmcd.nReac; // number of gas reactions
    const auto Nrs   = kmcdSurf.nReac; // numger of surface reactions

    _Smat  = real_type_3d_view("Smat_PlugflowreactorSmat", _nBatch, _Nvars, Nrg );
    _Ssmat = real_type_3d_view("Smat_PlugflowreactorSmat", _nBatch, _Nvars, Nrs );
    //need to reduce order of velocity in RHS PFR and Jacabian

    TChem::PlugFlowReactorSmat::
           runDeviceBatch(_nBatch,
                          //inputs
                          _state, //gas
                          _siteFraction,//surface
                          _velocity, // PRF
                           /// output
                          _Smat, _Ssmat,
                          kmcd,
                          kmcdSurf,
                          _pfrd);

    return(0);
    }

int ChemElemDAE_PFR_TChem::getNumofGasReactions()
     {
      return (kmcd.nReac);
     }

int ChemElemDAE_PFR_TChem::getNumofSurfaceReactions()
    {
    return (kmcdSurf.nReac);
    }
void ChemElemDAE_PFR_TChem::setPlugFlowReactor(const double Area, const double Pcat )
    {
      _pfrd.Area = Area; // m2
      _pfrd.Pcat = Pcat; //
    }
int ChemElemDAE_PFR_TChem::getNumOfRateOfProcesses(){
    return _Nrate_of_processes;
    }
