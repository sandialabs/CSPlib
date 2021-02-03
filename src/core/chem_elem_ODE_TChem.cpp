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


#include "chem_elem_ODE_TChem.hpp"

// Use (void) to silent unused warnings.
#define assertm(exp, msg) assert(((void)msg, exp))

//==========================================================================================
ChemElemODETChem::ChemElemODETChem(
                const std::string &mech_gas_file     ,
                const std::string &thermo_gas_file    )

{

    _nBatch = 1;
    const bool detail = false;
    TChem::     exec_space::print_configuration(std::cout, detail);
    TChem::host_exec_space::print_configuration(std::cout, detail);

    kmd = TChem::KineticModelData(mech_gas_file,thermo_gas_file);

    kmcd = kmd.createConstData<TChem::exec_space>(); // data struc with gas phase info
    // make also a copy on host
    kmcd_host = kmd.createConstData<TChem::host_exec_space>();
    _Nvars = kmcd.nSpec + 1;//
    _Nspec = kmcd.nSpec;
    _Nreac = kmcd.nReac;
    _Nelem = kmcd.NumberofElementsGas;
    setNumOfVariables(_Nvars);// in base class
    printf("Nspec = %-4d\n",_Nspec) ;
    printf("Nvars = %-4d\n",_Nvars) ;
    printf("Nreac = %-4d\n",_Nreac) ;
    printf("Nelem = %-4d\n",_Nelem) ;

    _run_on_device=true;


    // _spec_name
}

void ChemElemODETChem::run_on_host(const bool & run_on_host){

  if (run_on_host){
    _run_on_device = false;
  } else {
    _run_on_device = true;
  }

}


void ChemElemODETChem::readIgnitionZeroDDataBaseFromFile(const std::string& filename,
                                                  std::vector<std::string> &varnames)
   {


  //read a solution from a Ignition Zero D problem:
  //this data is from TChem++, number of row is number of state vector
  // order: iteration,  time,  dt,  density, pressure, Temp [K], mass fraction

  double atposition;
  const int TotalEq = kmcd.nSpec + 3 + 3;
  printf("Reading from Ignition Zero D data base \n ");
  std::vector< std::vector<double> > ingdb;
  std::string line;

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
      for (int i=1; i<TotalEq; i++)
        ixfs >> vec[i];
      ingdb.push_back(vec);
    }
  } else {
      std::cerr << " chem_elem_ODE_TChem.cpp: cannot open file "+ filename +"\n";
      exit(-1);
    }

  ixfs.close();

  printf("data base size %lu %lu \n", ingdb.size(),ingdb[0].size() );
  _nBatch = ingdb.size();

  const ordinal_type stateVectorLen(TChem::Impl::getStateVectorSize(kmcd.nSpec));

  /// input: state vectors: temperature, pressure and mass fraction
  _state = real_type_2d_view("state vector", _nBatch, stateVectorLen);

  /// create a mirror view to store input from a file
  _state_host = Kokkos::create_mirror_view(_state);
  for (int sp = 0; sp < _nBatch; sp++) {
    /* state vector density, pressure, Temp, mass fraction*/
    for (int i = 0; i < kmcd.nSpec + 3  ; i++){
      _state_host(sp,i) = ingdb[sp][i+3];
    }
  }

  Kokkos::deep_copy(_state, _state_host);

  _state_db = std::vector<std::vector<double>>(_nBatch,
              std::vector<double>(_Nvars,0.0));

  for (int i = 0; i < _nBatch; i++) {
    auto state_host_at_i = Kokkos::subview(_state_host, i, Kokkos::ALL());
    const Impl::StateVector<real_type_1d_view_host> sv_at_i(kmcd.nSpec, state_host_at_i);
    _state_db[i][0] = sv_at_i.Temperature();
    const real_type_1d_view_host Ys = sv_at_i.MassFractions();
    for (int j = 1; j < _Nvars; j++) {
      _state_db[i][j] = Ys(j-1);
    }
  }

}

void ChemElemODETChem::setStateVectorDB(std::vector<std::vector <double> >& state_db){
     //
     _nBatch = state_db.size();
     const ordinal_type stateVectorLen(TChem::Impl::getStateVectorSize(kmcd.nSpec));
     _state = real_type_2d_view("state vector", _nBatch, stateVectorLen);
     /// create a mirror view to store input from a file
     _state_host = Kokkos::create_mirror_view(_state);

     _state_db = std::vector<std::vector<double>>(_nBatch,
                 std::vector<double>(_Nvars,0.0));
     //
     printf("Number of samples in the data base %d \n", _nBatch );

     /* state vector density, pressure, Temp, mass fraction*/
     /* time, density, pressure, Temp amd mass fraction */
     for (int i = 0; i < _nBatch; i++) {
       for (int j = 0; j < _Nvars + 3 ; j++) {
         _state_host(i,j) = state_db[i][j + 1];
       }

       for (int j = 0; j < _Nvars; j++) {
         _state_db[i][j] = state_db[i][ j + 3];
       }

     }

     Kokkos::deep_copy(_state, _state_host);

}

void ChemElemODETChem::getStateVector(std::vector<std::vector <double> >& state_db)
{
     state_db.clear();
     state_db = _state_db;
}

void ChemElemODETChem::getSourceVector(std::vector<std::vector <double> >& source_db)
{
    source_db.clear();
    source_db = std::vector<std::vector<double>>(_rhs_host.extent(0),std::vector<double>(_rhs_host.extent(1),0.0));
    TChem::convertToStdVector(source_db, _rhs_host);
}


void ChemElemODETChem::getJacMatrix(std::vector < std::vector
                                          <std::vector <double> > >& jac_db)
{
    jac_db.clear();
    jac_db = std::vector< std::vector< std::vector< double > > >
    (_jac_host.extent(0), std::vector<std::vector<double> > (_jac_host.extent(1),
     std::vector<double>(_jac_host.extent(2),0)));

    TChem::convertToStdVector(jac_db, _jac_host);
  }

void ChemElemODETChem::getRoP(std::vector<std::vector <double> >& RoP)
{

     RoP.clear();
     RoP = std::vector<std::vector<double>>(_nBatch,std::vector<double>(2*_Nreac,0.0));

     std::vector<std::vector<double > > RoPFor;
     std::vector<std::vector<double > > RoPRev;

     TChem::convertToStdVector(RoPFor, _RoPFor_host);
     TChem::convertToStdVector(RoPRev, _RoPRev_host);

     for (int i = 0; i < _nBatch; i++) {
       for (int j = 0; j < _Nreac; j++) {
         RoP[i][j] = RoPFor[i][j];
         RoP[i][j + _Nreac] = -RoPRev[i][j];
       }
     }

}

void ChemElemODETChem::getSmatrix(std::vector < std::vector
                                          <std::vector <double> > >& Smatrixdb)
{

     std::vector< std::vector< std::vector<double> > >
     Smat(_Smat_host.extent(0), std::vector<std::vector<double> > (_Smat_host.extent(1),
      std::vector<double>(_Smat_host.extent(2),0)));

     TChem::convertToStdVector(Smat, _Smat_host);

     Smatrixdb.clear();
     Smatrixdb =
     std::vector< std::vector< std::vector<double> > >
     (_nBatch, std::vector<std::vector<double> > (_Nvars,
      std::vector<double>(2*_Nreac,0)));


     Smatrixdb.resize(_nBatch);
     for (int k = 0; k < _nBatch; k++) {
       Smatrixdb[k].resize(_Nvars);
       for (int i=0; i< _Nvars ; i++) {
         Smatrixdb[k][i].resize(2*_Nreac);
         for (int j = 0; j < _Nreac; j++) {
           Smatrixdb[k][i][j] = Smat[k][i][j];
           Smatrixdb[k][i][j + _Nreac] = Smat[k][i][j];
         }
       }
     }

}

int ChemElemODETChem::NumOfSpecies()
{
  return _Nspec;
}

int ChemElemODETChem::NumOfReactions()
{
  return _Nreac;
}



void ChemElemODETChem::getSpeciesNames(std::vector<std::string>& spec_name)
{
  spec_name.clear();
  // get species name from device and make a copy in host
  const auto speciesNamesHost = Kokkos::create_mirror_view(kmcd.speciesNames);
  Kokkos::deep_copy(speciesNamesHost, kmcd.speciesNames);
  //make a copy in a std vector
  for (int k = 0; k < speciesNamesHost.extent(0); k++)
    spec_name.push_back(&speciesNamesHost(k,0));
}

int ChemElemODETChem::getVarIndex(const std::string & var_name){

  // get species name from device and make a copy in host
  const auto speciesNamesHost = Kokkos::create_mirror_view(kmcd.speciesNames);
  Kokkos::deep_copy(speciesNamesHost, kmcd.speciesNames);

  if (var_name =="Temperature"){
    return 0;
  } else{

    for (int i = 0; i < kmcd.nSpec; i++) {
      if (strncmp(&speciesNamesHost(i, 0),
          (var_name).c_str(),
          LENGTHOFSPECNAME) == 0) {
          return i+1;
        }
   }
  }

  std::cout<< __FILE__<<":"<<__func__<<":"<<__LINE__<<":"
           << var_name<<" does not exist in reaction mechanism\n";
  exit(1);

}

void ChemElemODETChem::evalRoP()
    {
    //compute rate of progress for gas phase
    const auto Nrg   = kmcd.nReac; // number of gas reactions

    if (_run_on_device){

      _RoPFor = real_type_2d_view("Gas_Forward_RateOfProgess", _nBatch, Nrg );
      _RoPRev = real_type_2d_view("Gas_Reverse_RateOfProgess", _nBatch, Nrg);

      RateOfProgress::
             runDeviceBatch(_nBatch,
                            _state, //gas
                             //output,
                            _RoPFor, // Forward rate of progess
                            _RoPRev, // reverse rate of progess
                            kmcd);

      _RoPFor_host = Kokkos::create_mirror_view(_RoPFor);
      Kokkos::deep_copy( _RoPFor_host, _RoPFor);

      _RoPRev_host = Kokkos::create_mirror_view(_RoPRev);
      Kokkos::deep_copy(_RoPRev_host, _RoPRev);

    } else {

      _RoPFor = real_type_2d_view("Gas_Forward_RateOfProgess", _nBatch, Nrg );
      _RoPRev = real_type_2d_view("Gas_Reverse_RateOfProgess", _nBatch, Nrg);

      _RoPFor_host = real_type_2d_view_host("Gas_Forward_RateOfProgess_Host", _nBatch, Nrg );
      _RoPRev_host = real_type_2d_view_host("Gas_Reverse_RateOfProgess_Host", _nBatch, Nrg);

      RateOfProgress::
             runHostBatch(_nBatch,
                            _state_host, //gas
                             //output,
                            _RoPFor_host, // Forward rate of progess
                            _RoPRev_host, // reverse rate of progess
                            kmcd_host);

      Kokkos::deep_copy( _RoPFor, _RoPFor_host);
      Kokkos::deep_copy( _RoPRev, _RoPRev_host);

    }


}


int ChemElemODETChem::evalSourceVector()
{

    if (_run_on_device){

    _rhs = real_type_2d_view("sourcethermgas", _nBatch, _Nvars );
    TChem::SourceTerm::runDeviceBatch(
                                _nBatch,
                                _state, //gas
                                _rhs,
                                kmcd);
    //
    _rhs_host = Kokkos::create_mirror_view(_rhs);
    Kokkos::deep_copy( _rhs_host, _rhs);


    } else{

      _rhs = real_type_2d_view("sourcethermgas", _nBatch, _Nvars );
      _rhs_host = real_type_2d_view_host("sourcethermgas_Host", _nBatch, _Nvars );
      TChem::SourceTerm::runHostBatch(
                                  _nBatch,
                                  _state_host, //gas
                                  _rhs_host,
                                  kmcd_host);

      Kokkos::deep_copy( _rhs, _rhs_host);

    }

    return 0;

}


void ChemElemODETChem::evalSmatrix()
{

  if (_run_on_device){

    _Smat  = real_type_3d_view("Smat_PlugflowreactorSmat", _nBatch, _Nvars, _Nreac );
    TChem::Smatrix::runDeviceBatch(_nBatch,
                                    _state, //gas
                                    _Smat,
                                    kmcd);
    //
    _Smat_host = Kokkos::create_mirror_view(_Smat);
    Kokkos::deep_copy(_Smat_host, _Smat);

  } else {

    _Smat  = real_type_3d_view("Smat_PlugflowreactorSmat_Host", _nBatch, _Nvars, _Nreac );
    _Smat_host  = real_type_3d_view_host("Smat_PlugflowreactorSmat", _nBatch, _Nvars, _Nreac );

    TChem::Smatrix::runHostBatch(_nBatch,
                                   _state_host, //gas
                                    _Smat_host,
                                    kmcd_host);

   //
   Kokkos::deep_copy( _Smat,  _Smat_host);
  }

}

int ChemElemODETChem::evalJacMatrix(unsigned int useJacAnl)
    {
      if (useJacAnl) {
        printf(" Using Numerical Jacobian\n");

        if (_run_on_device){

          _jac = real_type_3d_view("AnalyticalJacIgnition",_nBatch, _Nvars, _Nvars);
          real_type_2d_view fac("fac", _nBatch, _Nvars); // this variables needs to be Ntotal length
          //
          TChem::IgnitionZeroDNumJacobian
               ::runDeviceBatch(_nBatch,
                              _state, //gas
                              _jac,
                              fac,
                              kmcd);
         //
         _jac_host = Kokkos::create_mirror_view(_jac);
         Kokkos::deep_copy( _jac_host, _jac);
         } else {
          _jac = real_type_3d_view("AnalyticalJacIgnition",_nBatch, _Nvars, _Nvars);
          _jac_host = real_type_3d_view_host ("AnalyticalJacIgnitionHost",_nBatch, _Nvars, _Nvars);
          real_type_2d_view_host fac("fac", _nBatch, _Nvars); // this variables needs to be Ntotal length
         //
          TChem::IgnitionZeroDNumJacobian
               ::runHostBatch(_nBatch,
                                _state_host, //gas
                                _jac_host,
                               fac,
                              kmcd_host);

         //
         Kokkos::deep_copy(_jac, _jac_host );

        }

      } else{
        printf(" Using Analytical Jacobian\n");


        if (_run_on_device){

          _jac = real_type_3d_view("AnalyticalJacIgnition",_nBatch, _Nvars, _Nvars);
          //
          TChem::JacobianReduced
               ::runDeviceBatch(_nBatch,
                                _state, //gas
                                _jac,
                                kmcd);

          _jac_host = Kokkos::create_mirror_view(_jac);
          Kokkos::deep_copy( _jac_host, _jac);


      } else {

        _jac = real_type_3d_view("AnalyticalJacIgnition",_nBatch, _Nvars, _Nvars);
        _jac_host = real_type_3d_view_host("AnalyticalJacIgnition_Host",_nBatch, _Nvars, _Nvars);
        //
        TChem::JacobianReduced
             ::runHostBatch(_nBatch,
                              _state_host, //gas
                              _jac_host,
                              kmcd_host);

        Kokkos::deep_copy(_jac, _jac_host );

      }

      }

  return(0);
    }

int ChemElemODETChem::getNumOfElements() {
    return _Nelem;
}

void ChemElemODETChem::getStateVector(std::vector<double>& state_vec)
{

  auto state_host_at_0 = Kokkos::subview(_state_host, 0, Kokkos::ALL());
  const Impl::StateVector<real_type_1d_view_host> sv_at_i(kmcd.nSpec, state_host_at_0);
  const real_type t = sv_at_i.Temperature();
  const real_type_1d_view_host Ys = sv_at_i.MassFractions();

  _state_vec.clear();
  _state_vec.push_back(sv_at_i.Temperature());

  // Ys
  for (auto i = 0; i < kmcd.nSpec; i++)
   _state_vec.push_back( Ys(i));
  state_vec = _state_vec;

}

void ChemElemODETChem::getSourceVector(std::vector<double>& source_vec)
{
  _source_vec.shrink_to_fit();
  _source_vec = std::vector<double>(_Nvars,0.0);
  auto rhs_host_at_0 = Kokkos::subview(_rhs_host, 0, Kokkos::ALL());
  TChem::convertToStdVector(_source_vec, rhs_host_at_0);
  source_vec = _source_vec;

}

void ChemElemODETChem::getJacMatrix(std::vector<std::vector<double> >& jmat)
{
     _jmat.clear();
     _jmat.shrink_to_fit();
     _jmat = std::vector<std::vector<double>>(_Nvars,std::vector<double>(_Nvars,0.0));

     auto jac_host_at_0 = Kokkos::subview(_jac_host, 0, Kokkos::ALL(), Kokkos::ALL());
     TChem::convertToStdVector(_jmat, jac_host_at_0);
     jmat = _jmat;
}

void ChemElemODETChem::getRoP(std::vector<double>& RoP)
{
     //
     // gas forward rate of progress
     auto RoPFor_host_at_0 = Kokkos::subview(_RoPFor_host, 0, Kokkos::ALL());
     // gas reverse rate of progress
     auto RoPRev_host_at_0 = Kokkos::subview(_RoPRev_host, 0, Kokkos::ALL());

     for (int j = 0; j < _Nreac; j++) {
       RoP[j] = RoPFor_host_at_0(j);
       RoP[j + _Nreac] = -RoPRev_host_at_0(j);
     }

}

void ChemElemODETChem::getSmatrix(std::vector<std::vector<double> >& Smat)
{

  auto Smat_host_at_0 = Kokkos::subview(_Smat_host, 0, Kokkos::ALL(), Kokkos::ALL());

  for (int i=0; i< _Nvars ; i++) {
    for (int j = 0; j < _Nreac; j++) {
      Smat[i][j] = Smat_host_at_0(i,j);
      Smat[i][j + _Nreac] = Smat_host_at_0(i,j);
    }
  }

}

void ChemElemODETChem::evalAndGetEigenDecompKokkos(
  std::vector<std::vector <double> >& eig_val_real,
  std::vector<std::vector <double> >& eig_val_imag,
  std::vector < std::vector<std::vector <double> > >& eig_vec_R
)
{
  real_type_2d_view er("real eigen values", _nBatch, _Nvars );
  real_type_2d_view ei("imag eigen values", _nBatch, _Nvars );

  real_type_3d_view V("V", _nBatch, _Nvars, _Nvars);

  //work spaces
  const int wsize = 3*_Nvars*_Nvars + 2*_Nvars;
  real_type_2d_view W ("W",  _nBatch, wsize);

  real_type_3d_view jac("jac", _nBatch, _Nvars, _Nvars);
  //make a copy of _jac, because EigendecompositionKokkos is changing it .
  Kokkos::deep_copy(jac,_jac);

  Tines::SolveEigenvaluesNonSymmetricProblemDevice<exec_space>
    ::invoke(exec_space(), jac, er, ei, V, W);

  eig_val_real =
  std::vector<std::vector<double>>(er.extent(0),
  std::vector<double>(er.extent(1),0.0));

  eig_val_imag =
  std::vector<std::vector<double>>(ei.extent(0),
  std::vector<double>(ei.extent(1),0.0));

  eig_vec_R = std::vector< std::vector< std::vector< double > > >
  (V.extent(0), std::vector<std::vector<double> > (V.extent(1),
  std::vector<double>(V.extent(2),0)));

  TChem::convertToStdVector(eig_val_real, er);
  TChem::convertToStdVector(eig_val_imag, ei);
  TChem::convertToStdVector(eig_vec_R, V);
}
