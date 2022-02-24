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


#include "chem_elem_TCSTR_TChem.hpp"
#include "tools.hpp"
#include "util.hpp"



// Use (void) to silent unused warnings.
#define assertm(exp, msg) assert(((void)msg, exp))

//==========================================================================================
ChemElemTCSTR_TChem::ChemElemTCSTR_TChem(
                const std::string &mech_gas_file     ,
                const std::string &mech_surf_file    ,
                const std::string &thermo_gas_file   ,
                const std::string &thermo_surf_file ,
                const int& Nalgebraic_constraints )

{

    _nBatch = 1;
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

    _Ngas_species = kmcd.nSpec;
    _Nsurface_species = kmcdSurf.nSpec;
    _Ngas_reactions = kmcd.nReac;
    _Nsurface_reactions = kmcdSurf.nReac;
    _Nelem = kmcd.NumberofElementsGas;


    //rate of progress: gas/surface forward, reverse and conv from T-CSTR
    _Nrate_of_processes = 2*_Ngas_reactions + 2*_Nsurface_reactions + 1;
    _Nspecies_p_temperature    = kmcd.nSpec + 1;

    // only allow surface equations to be alegraic constraints
    if (Nalgebraic_constraints > kmcdSurf.nSpec) {
      _Nalgebraic_constraints =  kmcdSurf.nSpec;
    } else {
      _Nalgebraic_constraints = Nalgebraic_constraints;
    }
    _Nsurface_equations = _Nsurface_species - _Nalgebraic_constraints;

    _Nvars = _Nspecies_p_temperature + _Nsurface_equations; // number of variables on the ode part
    _Ntotal_variables = _Nspecies_p_temperature + _Nsurface_species; // total number of variables

    setNumOfVariables(_Nvars);// in base class

    printf("Number of gas species: %d\n", _Ngas_species );
    printf("Number of surface species: %d\n", _Nsurface_species );
    printf("Number of gas reactions: %d\n", _Ngas_reactions );
    printf("Number of surface reactions: %d\n", _Nsurface_reactions );
    printf("Number of algebraic constraints: %d\n", _Nalgebraic_constraints );
    printf("Number of elements %d\n", _Nelem);
    printf("Number of variables %d\n", _Nvars );
    printf("Total Number of variables %d\n", _Ntotal_variables );
    // _spec_name
}

void ChemElemTCSTR_TChem::readDataBaseFromFile(const std::string &filename,
                                                std::vector<std::string> &varnames)
//
{
     //read a solution from a TCSTR TChem:
     //this data is from TChem++
     //columns solution at each position
     // iter t, dt density, pressure, Temp [K], mass fraction,  site fraction

     double atposition;
     const int TotalEq = _Ngas_species + _Nsurface_species + 3 + 3;

     std::vector< std::vector<double> > cstrdb;

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

         cstrdb.push_back(vec);
       }
     } else {
         std::cerr << " chem_elem_DAE_PFR_TChem.cpp: cannot open file "+ filename +"\n";
         exit(-1);
      }

    ixfs.close();

    _nBatch = cstrdb.size();

#if defined(CSP_PRINT)
     for (int sp = 0; sp < _nBatch; sp++) {
       for (int i = 1; i < (cstrdb[0]).size() ; i++) {
         printf("value sp %d i %d %f\n", sp, i ,cstrdb[sp][i]   );
       }
     }
#endif
     printf("Reading data from  TCSTR solution: # state vectors %lu # Variables (including site fraction) %lu\n",cstrdb.size() ,(cstrdb[0]).size());

     /// input: state vectors: temperature, pressure and mass fraction
     const int stateVectorLen = TChem::Impl::getStateVectorSize(_Ngas_species);

     _state = real_type_2d_view("state vector", _nBatch, stateVectorLen );
     // input :: surface fraction vector, zk
     _siteFraction = real_type_2d_view("SiteFraction", _nBatch, _Nsurface_species);

     /// create a mirror view to store input from a file
     auto state_host = Kokkos::create_mirror_view(_state);
     auto siteFraction_host = Kokkos::create_mirror_view(_siteFraction);

     for (int sp = 0; sp < _nBatch; sp++) {
       /* state vector density, pressure, Temp, mass fraction*/
       for (int i = 0; i < stateVectorLen  ; i++) {
         state_host(sp,i) = cstrdb[sp ][i+3];
         // printf("state i %d, %f, %f\n", i, cstrdb[sp + position][i], state_host(sp,i) );
       }
       /* site fraction */
       for (int k = 0; k < _Nsurface_species ; k++) {
         siteFraction_host(sp,k) = cstrdb[sp ][k + stateVectorLen + 3];
         // printf("site i %d, %f % f\n", k, cstrdb[sp + position][k+ stateVectorLen + 1], siteFraction_host(sp,k) );
       }

     }


     Kokkos::deep_copy(_state, state_host);
     Kokkos::deep_copy(_siteFraction, siteFraction_host);

     _state_db = std::vector<std::vector<double>>(_nBatch,
                 std::vector<double>(_Ntotal_variables,0.0));

     for (int i = 0; i < _nBatch; i++) {
       auto state_host_at_i = Kokkos::subview(state_host, i, Kokkos::ALL());
       auto siteFraction_host_at_i = Kokkos::subview(siteFraction_host, i, Kokkos::ALL());

       const Impl::StateVector<real_type_1d_view_host> sv_at_i(_Ngas_species, state_host_at_i);
       _state_db[i][0] = sv_at_i.Temperature();
       const real_type_1d_view_host Ys = sv_at_i.MassFractions();

       for (int j = 1; j < _Ngas_species; j++) {
         _state_db[i][j] = Ys(j-1);
       }

       for (int j = _Ngas_species + 1; j < _Nvars; j++) {
         _state_db[i][j] = siteFraction_host_at_i(j - _Ngas_species -1 );
       }
     }

}


void ChemElemTCSTR_TChem::getStateDBonHost(std::vector<std::vector <double> >& state_db)
{
     state_db.clear();
     state_db = _state_db;
}

void ChemElemTCSTR_TChem::getSourceDBonHost(std::vector<std::vector <double> >& source_db)
{
    source_db.clear();
    std::vector< std::vector< double> > source_db_temp;
    source_db_temp = std::vector<std::vector<double>>(_rhs.extent(0),std::vector<double>(_rhs.extent(1),0.0));
    TChem::convertToStdVector(source_db_temp, _rhs);
    source_db = std::vector<std::vector<double>>(_nBatch,std::vector<double>(_Nvars,0.0));

    for (int sp = 0; sp < _nBatch; sp++) {
      for (int i = 0; i < _Nvars; i++) {
        source_db[sp][i] = source_db_temp[sp][i];
      }
    }
}

void ChemElemTCSTR_TChem::getStateVector(std::vector<double>& state_vec)
{
  auto state_host = Kokkos::create_mirror_view(_state);
  Kokkos::deep_copy(state_host, _state);
  auto state_host_at_0 = Kokkos::subview(state_host, 0, Kokkos::ALL());
  const Impl::StateVector<real_type_1d_view_host> sv_at_i(_Ngas_species, state_host_at_0);

  const real_type t = sv_at_i.Temperature();
  const real_type_1d_view_host Ys = sv_at_i.MassFractions();

  auto siteFraction_host = Kokkos::create_mirror_view(_siteFraction);
  Kokkos::deep_copy(_siteFraction, siteFraction_host);
  auto siteFraction_host_at_0 = Kokkos::subview(siteFraction_host, 0, Kokkos::ALL());

  _state_vec.clear();
  _state_vec.push_back(sv_at_i.Temperature());

  // Ys
  for (auto i = 0; i < _Ngas_species; i++)
   _state_vec.push_back( Ys(i));

  for (auto i = 0; i < _Nsurface_species; i++)
   _state_vec.push_back( siteFraction_host_at_0(i));

  state_vec = _state_vec;

}


void ChemElemTCSTR_TChem::getSourceVector(std::vector<double>& source_vec)
{
  //  _source_vec.clear();
  _source_vec.shrink_to_fit();
  _source_vec=std::vector<double>(_Ntotal_variables,0.0);
  auto rhs_host = Kokkos::create_mirror_view(_rhs);
  Kokkos::deep_copy(rhs_host, _rhs);
  //
  auto rhs_host_at_0 = Kokkos::subview(rhs_host, 0, Kokkos::ALL());
  TChem::convertToStdVector(_source_vec, rhs_host_at_0);
  source_vec = _source_vec;

}


void ChemElemTCSTR_TChem::getJacobianDBonHost(std::vector < std::vector
                                          <std::vector <double> > >& jac_db)
{
    jac_db.clear();
    jac_db = std::vector< std::vector< std::vector< double > > >
    (_jac.extent(0), std::vector<std::vector<double> > (_jac.extent(1),
     std::vector<double>(_jac.extent(2),0)));

    TChem::convertToStdVector(jac_db, _jac);
  }



void ChemElemTCSTR_TChem::getSpeciesNames(std::vector<std::string>& spec_name)
{
  spec_name.clear();

  const auto speciesNamesHost = kmcd_host.speciesNames;
  //make a copy in a std vector
  for (int k = 0; k < speciesNamesHost.extent(0); k++)
    spec_name.push_back(&speciesNamesHost(k,0));

  const auto speciesSurfNamesHost = kmcdSurf_host.speciesNames;
  //make a copy in a std vector
  for (int k = 0; k < _Nsurface_equations; k++)
    spec_name.push_back(&speciesSurfNamesHost(k,0));

}

void ChemElemTCSTR_TChem::evalRoP()
{
//compute rate of progress for gas phase
_RoPFor = real_type_2d_view("Gas_Forward_RateOfProgess", _nBatch, _Ngas_reactions );
_RoPRev = real_type_2d_view("Gas_Reverse_RateOfProgess", _nBatch, _Ngas_reactions);


TChem::RateOfProgress::
       runDeviceBatch(_nBatch,
                      _state, //gas
                       //output,
                      _RoPFor, // Forward rate of progess
                      _RoPRev, // reverse rate of progess
                      kmcd);

// compute rate of progess surface phase
_RoPForSurf = real_type_2d_view("Surface_Forward_RateOfProgess", _nBatch, _Nsurface_reactions);
_RoPRevSurf = real_type_2d_view("Surface_Reverse_RateOfProgess", _nBatch, _Nsurface_reactions);

TChem::RateOfProgressSurface::
       runDeviceBatch(_nBatch,
                      _state, //gas
                      _siteFraction,//surface
                      //output,
                      _RoPForSurf, // Forward rate of progress surface phase
                      _RoPRevSurf, // reverse rate of progess surface phase
                      kmcd,
                      kmcdSurf);

}

void ChemElemTCSTR_TChem::getRoP(std::vector<double>& RoP)
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


for (int j = 0; j < _Ngas_reactions; j++) {
  RoP[j] = RoPFor_host_at_0(j);
  RoP[j + _Ngas_reactions] = -RoPRev_host_at_0(j);
}

for (int j = 0; j < _Nsurface_reactions; j++) {
  RoP[j + 2*_Ngas_reactions] = RoPForSurf_host_at_0(j);
  RoP[j + 2*_Ngas_reactions + _Nsurface_reactions ] = -RoPRevSurf_host_at_0(j);
}
// add convection term
RoP[ 2*_Ngas_reactions + 2*_Nsurface_reactions ] = 1;


}

//
void ChemElemTCSTR_TChem::getRoPDBonHost(std::vector<std::vector <double> >& RoP)
{
     RoP.clear();
     RoP = std::vector<std::vector<double>>(
       _nBatch,std::vector<double>(_Nrate_of_processes,0.0)); // last

     std::vector<std::vector<double > > RoPFor;
     std::vector<std::vector<double > > RoPRev;

     TChem::convertToStdVector(RoPFor, _RoPFor);
     TChem::convertToStdVector(RoPRev, _RoPRev);

     std::vector<std::vector<double > > RoPForSurf;
     std::vector<std::vector<double > > RoPRevSurf;

     TChem::convertToStdVector(RoPForSurf, _RoPForSurf);
     TChem::convertToStdVector(RoPRevSurf, _RoPRevSurf);

     for (int i = 0; i < _nBatch; i++) {

       for (int j = 0; j < _Ngas_reactions; j++) {
         RoP[i][j] = RoPFor[i][j];
         RoP[i][j + _Ngas_reactions] = -RoPRev[i][j];
       }

       for (int j = 0; j < _Nsurface_reactions; j++) {
         RoP[i][j + 2*_Ngas_reactions] = RoPForSurf[i][j];
         RoP[i][j + 2*_Ngas_reactions + _Nsurface_reactions ] = -RoPRevSurf[i][j];
       }

       RoP[i][2*_Ngas_reactions + 2*_Nsurface_reactions ] = 1; // m*(Y*_k-Y_k) term in CSTR formulation

     }

}


void ChemElemTCSTR_TChem::setCSTR( const std::string& input_condition_file_name,
                                     const double& mdotIn,  const double& Vol,
                                     const double& Acat, const bool& isothermal )
{
  // cstr need initial condition of simulation
  real_type_2d_view_host state_host_initial_condition;

  const auto speciesNamesHost = kmcd_host.speciesNames;

  const ordinal_type stateVecDim = TChem::Impl::getStateVectorSize(_Ngas_species);

  // get species molecular weigths
  const auto SpeciesMolecularWeights = kmcd_host.sMass;

  ordinal_type nBatch(0);
  // use same file that TChem
  TChem::Test::readSample(input_condition_file_name,
                          speciesNamesHost,
                          SpeciesMolecularWeights,
                          _Ngas_species,
                          stateVecDim,
                          state_host_initial_condition,
                          nBatch);

  // works for only one initial condition,
  //we cannot use this code for samples that are produced with different initial condition
  const real_type_1d_view_host state_at_i =
  Kokkos::subview(state_host_initial_condition, 0, Kokkos::ALL());
  const Impl::StateVector<real_type_1d_view_host> sv_at_i(_Ngas_species, state_at_i);

  const auto Ys = sv_at_i.MassFractions();

  _cstr.mdotIn = mdotIn; // inlet mass flow kg/s
  _cstr.Vol    = Vol; // volumen of reactor m3
  _cstr.Acat   = Acat; // Catalytic area m2: chemical active area
  _cstr.isothermal = 1;
  if (isothermal) _cstr.isothermal = 0;
  _cstr.pressure = sv_at_i.Pressure();
  _cstr.Yi = real_type_1d_view("Mass fraction at inlet", _Ngas_species);
  Kokkos::deep_copy(_cstr.Yi, Ys);
  _cstr.number_of_algebraic_constraints = _Nalgebraic_constraints;

  {
    real_type_2d_view_host EnthalpyMass("EnthalpyMass", 1, _Ngas_species);
    real_type_1d_view_host EnthalpyMixMass("EnthalpyMass Mixture", 1);
    // do this computation on the host, because of the size state_host_initial_condition(1)
    const auto host_exec_space_instance = TChem::host_exec_space();
    TChem::EnthalpyMass::runHostBatch(host_exec_space_instance,
                                      -1,
                                      -1,
                                      1,
                                      state_host_initial_condition,
                                      EnthalpyMass,
                                      EnthalpyMixMass,
                                      kmcd_host);
    _cstr.EnthalpyIn = EnthalpyMixMass(0);
  }

}

int ChemElemTCSTR_TChem::evalSourceVector()
{
  // total number of equation including alegraic constraints
  const auto nEqns = _Ngas_species + _Nsurface_species + 1 ;
  // printf("Number of Equation in DAE %d \n", nEqns );
  _rhs = real_type_2d_view("cstr RHS", _nBatch, nEqns  );

  TChem::TransientContStirredTankReactorRHS::runDeviceBatch(_nBatch,                                              //inputs
                                              _state, //gas
                                              _siteFraction,//surface
                                              //ouputs
                                              _rhs,
                                              //data
                                              kmcd,
                                              kmcdSurf,
                                              _cstr);

  return(0);

}

int ChemElemTCSTR_TChem::getNumOfRateOfProcesses(){
    return _Nrate_of_processes;
}


void ChemElemTCSTR_TChem::getSmatrix(std::vector<std::vector<double> >& Smat)
{

auto Smat_at_0 = Kokkos::subview(_Smat, 0, Kokkos::ALL(), Kokkos::ALL());
auto Smat_host_at_0 = Kokkos::create_mirror_view(Smat_at_0);
Kokkos::deep_copy(Smat_host_at_0, Smat_at_0);

auto Ssmat_at_0 = Kokkos::subview(_Ssmat, 0, Kokkos::ALL(), Kokkos::ALL());
auto Ssmat_host_at_0 = Kokkos::create_mirror_view(Ssmat_at_0);
Kokkos::deep_copy(Ssmat_host_at_0, Ssmat_at_0);

auto Sconv_at_0 = Kokkos::subview(_Sconv, 0, Kokkos::ALL());
auto Sconv_host_at_0 = Kokkos::create_mirror_view(Sconv_at_0);
Kokkos::deep_copy(Sconv_host_at_0, Sconv_at_0);


for (int i=0; i< _Nvars ; i++) {
  for (int j = 0; j < _Ngas_reactions; j++) {
    Smat[i][j] = Smat_host_at_0(i,j);
    Smat[i][j + _Ngas_reactions] = Smat_host_at_0(i,j);
  }

  for (int j = 0; j < _Nsurface_reactions; j++) {
    Smat[i][j + 2*_Ngas_reactions] = Ssmat_host_at_0(i,j);
    Smat[i][j + 2*_Ngas_reactions+ _Nsurface_reactions ] = Ssmat_host_at_0(i,j);
  }

  Smat[i][ 2*_Ngas_reactions + 2*_Nsurface_reactions ] = Sconv_host_at_0(i);
}


}

void ChemElemTCSTR_TChem::evalSmatrix()
{

   _Smat  = real_type_3d_view("Smat CSTR", _nBatch,   _Nspecies_p_temperature, _Ngas_reactions );
   _Ssmat = real_type_3d_view("Ssma CSTR", _nBatch,   _Nspecies_p_temperature, _Nsurface_reactions);
   _Sconv  = real_type_2d_view("Sconv CSTR", _nBatch, _Nspecies_p_temperature );
   //need to reduce order of velocity in RHS PFR and Jacabian
   _Cmat = real_type_3d_view("Surface equations matrix CSTR", _nBatch, _Nsurface_equations, _Nsurface_reactions);
   // gas species and temperature
   TChem::TransientContStirredTankReactorSmatrix
        ::runDeviceBatch(-1, //team size
                         -1, //vector_size
                         _nBatch,
                         //inputs
                         _state, //gas
                         _siteFraction,//surface
                          /// output
                         _Smat, _Ssmat, _Sconv,
                         kmcd,
                         kmcdSurf,
                         _cstr);

    // surface equations
   const auto Nsurface_equations(_Nsurface_equations);
   const auto Nsurface_reactions(_Nsurface_reactions);
   const auto Cmat=_Cmat;
   const auto vsurfki=kmcdSurf.vsurfki;
   const auto site_density = kmcdSurf.sitedensity*real_type(10.0);

   Kokkos::parallel_for(
   Kokkos::RangePolicy<TChem::exec_space>(0, _nBatch),
   KOKKOS_LAMBDA(const ordinal_type& i) {
    for (int k = 0; k < Nsurface_equations; k++) {
      for (int l = 0; l < Nsurface_reactions; l++) {
        Cmat(i, k, l) = vsurfki(k,l)/site_density;
      }
    }
  });

}

void ChemElemTCSTR_TChem::getSmatrixDBonHost(std::vector < std::vector
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


    std::vector< std::vector< double> >
    Sconv(_Sconv.extent(0), std::vector< double > (_Sconv.extent(1),0));
    TChem::convertToStdVector(Sconv, _Sconv);

    std::vector< std::vector< std::vector<double> > >
    Cmat(_Cmat.extent(0), std::vector<std::vector<double> > (_Cmat.extent(1),
    std::vector<double>(_Cmat.extent(2),0)));
    TChem::convertToStdVector(Cmat, _Cmat);


    Smatrixdb.clear();
    Smatrixdb =
    std::vector< std::vector< std::vector<double> > >
    (_nBatch, std::vector<std::vector<double> > (_Nvars,
    std::vector<double>(_Nrate_of_processes,0)));

    Smatrixdb.resize(_nBatch);
    for (int k = 0; k < _nBatch; k++) {
      Smatrixdb[k].resize(_Nvars);

      // gas species + temperature
      for (int i=0; i< _Nspecies_p_temperature ; i++) {

         Smatrixdb[k][i].resize(_Nrate_of_processes );

         for (int j = 0; j < _Ngas_reactions; j++) {
           Smatrixdb[k][i][j] = Smat[k][i][j];
           Smatrixdb[k][i][j + _Ngas_reactions] = Smat[k][i][j];
         }

         for (int j = 0; j < _Nsurface_reactions; j++) {
           Smatrixdb[k][i][j + 2*_Ngas_reactions] = Ssmat[k][i][j];
           Smatrixdb[k][i][j + 2*_Ngas_reactions + _Nsurface_reactions ] = Ssmat[k][i][j];
         }

         Smatrixdb[k][i][ 2*_Ngas_reactions + 2*_Nsurface_reactions ] = Sconv[k][i];
      }

      //surface equation in ode form
      for (int i = 0; i< _Nvars - _Nspecies_p_temperature ; i++) {
        Smatrixdb[k][i + _Nspecies_p_temperature].resize(_Nrate_of_processes );
        // gas reactions
        for (int j = 0; j < _Ngas_reactions; j++) {
          Smatrixdb[k][i + _Nspecies_p_temperature][j] = 0.0;
          Smatrixdb[k][i + _Nspecies_p_temperature][j + _Ngas_reactions  ] = 0.0;
        }
        //surface reactions
        for (int j = 0; j < _Nsurface_reactions; j++) {
          Smatrixdb[k][i + _Nspecies_p_temperature][j + 2*_Ngas_reactions] = Cmat[k][i][j];
          Smatrixdb[k][i + _Nspecies_p_temperature][j + 2*_Ngas_reactions + _Nsurface_reactions ] = Cmat[k][i][j];
        }
        //convection
        Smatrixdb[k][i+ _Nspecies_p_temperature][ 2*_Ngas_reactions + 2*_Nsurface_reactions ] = 0.0;

      }


     }




}

int ChemElemTCSTR_TChem::evalJacMatrix(unsigned int useJacAnl)
    {

    if (useJacAnl == 1 ) {
        printf(" Using Numerical Jacobian\n");
        computeNumJac();
    } else {
        printf(" Using Sacado Analytical Jacobian\n");
        computeSacadoJacobian();
    }

  return(0);
    }

void ChemElemTCSTR_TChem::computeNumJac()
{

  real_type_3d_view jac_long ("whole jacobian", _nBatch, _Ntotal_variables, _Ntotal_variables );
  real_type_2d_view fac("fac", _nBatch, _Ntotal_variables); // this variables needs to be Ntotal length

  TChem::TransientContStirredTankReactorNumJacobian::
                  runDeviceBatch(_nBatch,//inputs
                                 _state, //gas
                                 _siteFraction,//surface
                                 jac_long,
                                 fac,
                                 kmcd,
                                 kmcdSurf,
                                 _cstr);

  //
  if (_Nalgebraic_constraints > 0 ){
    _jac = real_type_3d_view ("readuced jac", _nBatch, _Nvars, _Nvars );

    ComputeReducedJacobian<TChem::exec_space>
    ::runBatch(jac_long, _jac, _Nvars, _Nalgebraic_constraints );
  } else{
    _jac = jac_long;
  }



}

void ChemElemTCSTR_TChem::computeSacadoJacobian()
{

  real_type_3d_view jac_long ("whole jacobian", _nBatch, _Ntotal_variables, _Ntotal_variables );

  const auto exec_space_instance = TChem::exec_space();
  using policy_type =
      typename TChem::UseThisTeamPolicy<TChem::exec_space>::type;
  const ordinal_type level = 1;

  const ordinal_type per_team_extent =
  TChem::TransientContStirredTankReactorSacadoJacobian
  ::getWorkSpaceSize(kmcd,kmcdSurf); ///
  const ordinal_type per_team_scratch =

  Scratch<real_type_1d_view>::shmem_size(per_team_extent);
  policy_type policy(exec_space_instance, _nBatch, Kokkos::AUTO());
  policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

  TChem::TransientContStirredTankReactorSacadoJacobian
       ::runDeviceBatch(policy,
                        _state, //gas
                        _siteFraction,//surface
                        jac_long,
                        kmcd,
                        kmcdSurf,
                        _cstr);

//
  if (_Nalgebraic_constraints > 0 ){
    _jac = real_type_3d_view ("readuced jac", _nBatch, _Nvars, _Nvars );

    ComputeReducedJacobian<TChem::exec_space>
    ::runBatch(jac_long, _jac, _Nvars, _Nalgebraic_constraints );
  } else{
    _jac = jac_long;
  }
}

void ChemElemTCSTR_TChem::getJacMatrix(std::vector<std::vector<double> >& jmat){
     _jmat.clear();
     _jmat.shrink_to_fit();

     _jmat = std::vector<std::vector<double>>(_Nvars,std::vector<double>(_Nvars,0.0));

     auto jac_at_0 = Kokkos::subview(_jac, 0, Kokkos::ALL(), Kokkos::ALL());
     auto jac_host_at_0 = Kokkos::create_mirror_view(jac_at_0);
     Kokkos::deep_copy(jac_host_at_0, jac_at_0);

     TChem::convertToStdVector(_jmat, jac_host_at_0);
     jmat = _jmat;
}

//
int ChemElemTCSTR_TChem::verifySmatRoP(std::vector<std::vector<double> >&  Smat,
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

int ChemElemTCSTR_TChem::getNumOfElements() {
    return _Nelem;
}

int ChemElemTCSTR_TChem::getNumofGasReactions()
     {
      return (_Ngas_reactions);
     }

int ChemElemTCSTR_TChem::getNumofSurfaceReactions()
    {
    return (_Nsurface_reactions);
    }
