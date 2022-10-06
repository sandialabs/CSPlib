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
#include "chem_elem_TCSTRI_TChem.hpp"

namespace CSP {

  ChemElemTCSTRI_TChem::
  ChemElemTCSTRI_TChem():
    _n_elem(),
    _n_gas_species(),
    _n_gas_reactions(),
    _n_surface_reactions(),
    _n_surface_species(),
    _n_vars(),
    _n_algebraic_constraints(0),
    _n_total_variables(),
    _n_surface_equations(),
    _n_rate_of_processes(),
    _kmd(),
    _kmcd_gas_device(),
    _kmcd_gas_host(),
    _kmcd_surf_device(),
    _kmcd_surf_host(),
    _cstr_device(),
    _site_fraction(),
    _state(),
    _rop_fwd_gas(),
    _rop_rev_gas(),
    _rop_fwd_surf(),
    _rop_rev_surf(),
    _Smat(),
    // _Cmat_surface_species(),
    _nBatch(0),
    _rhs(),
    _jacobian()
    // _policy()
  {
  }

  ChemElemTCSTRI_TChem::
  ChemElemTCSTRI_TChem(const std::string &mech_gas_file,
                       const std::string &mech_surf_file,
                       const std::string &thermo_gas_file,
                       const std::string &thermo_surf_file,
                       const ordinal_type& n_algebraic_constraints,
                       const bool  use_yaml ) :
    _cstr_device(),
    _site_fraction(),
    _state(),
    _rop_fwd_gas(),
    _rop_rev_gas(),
    _rop_fwd_surf(),
    _rop_rev_surf(),
    _Smat(),
    // _Cmat_surface_species(),
    _rhs(),
    _jacobian()
    // _policy()

  {
    _nBatch = 1;
    const bool detail = false;
    TChem::     exec_space::print_configuration(std::cout, detail);
    TChem::host_exec_space::print_configuration(std::cout, detail);

    if (use_yaml) {
      _kmd = TChem::KineticModelData(mech_gas_file, true);
    } else {
      _kmd = TChem::KineticModelData(mech_gas_file, thermo_gas_file, mech_surf_file, thermo_surf_file);
    }
    
    // create tchem object in device
    _kmcd_gas_device = TChem::createGasKineticModelConstData<interf_device_type>(_kmd);// data struc with gas phase info
    _kmcd_surf_device = TChem::createSurfaceKineticModelConstData<interf_device_type>(_kmd);// data struc with surface phase info

    // create also a copy for host
    _kmcd_gas_host = TChem::createGasKineticModelConstData<interf_host_device_type>(_kmd); // data struc with gas phase info
    _kmcd_surf_host = TChem::createSurfaceKineticModelConstData<interf_host_device_type>(_kmd);// data struc with surface phase info

    _n_gas_species = _kmcd_gas_device.nSpec;
    _n_surface_species = _kmcd_surf_device.nSpec;
    _n_gas_reactions = _kmcd_gas_device.nReac;
    _n_surface_reactions = _kmcd_surf_device.nReac;
    _n_elem = _kmcd_gas_device.NumberofElementsGas;


    //rate of progress: gas/surface forward, reverse and conv from T-CSTR
    _n_rate_of_processes = 2*_n_gas_reactions + 2*_n_surface_reactions + ordinal_type(2);

    // only allow surface equations to be alegraic constraints
    if (n_algebraic_constraints > _kmcd_surf_device.nSpec) {
      _n_algebraic_constraints =  _kmcd_surf_device.nSpec;
    } else {
      _n_algebraic_constraints = n_algebraic_constraints;
    }
    _n_surface_equations = _n_surface_species - _n_algebraic_constraints;

    _n_vars = _n_gas_species + _n_surface_equations; // number of variables on the ode part
    _n_total_variables = _n_gas_species + _n_surface_species; // total number of variables

    printf("Number of gas species: %d\n", _n_gas_species );
    printf("Number of surface species: %d\n", _n_surface_species );
    printf("Number of gas reactions: %d\n", _n_gas_reactions );
    printf("Number of surface reactions: %d\n", _n_surface_reactions );
    printf("Number of algebraic constraints: %d\n", _n_algebraic_constraints );
    printf("Number of elements %d\n", _n_elem);
    printf("Number of variables %d\n", _n_vars );
    printf("Total Number of variables %d\n", _n_total_variables );
  }

  ChemElemTCSTRI_TChem::
  ~ChemElemTCSTRI_TChem()
  {
  }

  void ChemElemTCSTRI_TChem::
  setCSTR( const std::string& input_condition_file_name,
           const double& inlet_mass_flow,  const double& reactor_volume,
           const double& catalytic_area, const int& poisoning_species_idx )
  {
    // cstr need initial condition of simulation
    real_type_2d_view_host state_host_initial_condition;
    const ordinal_type stateVecDim = TChem::Impl::getStateVectorSize(_n_gas_species);

    ordinal_type nBatch(0);
    // use same file that TChem
    TChem::Test::readSample(input_condition_file_name,
                            _kmcd_gas_host.speciesNames,
                            _kmcd_gas_host.sMass,
                            _n_gas_species,
                            stateVecDim,
                            state_host_initial_condition,
                            nBatch);

    // works for only one initial condition,
    //we cannot use this code for samples that are produced with different initial condition
    const real_type_1d_view_host state_at_i =
    Kokkos::subview(state_host_initial_condition, 0, Kokkos::ALL());
    const Impl::StateVector<real_type_1d_view_host> sv_at_i(_n_gas_species, state_at_i);

    const auto Ys = sv_at_i.MassFractions();

    _cstr_device.mdotIn = inlet_mass_flow; // inlet mass flow kg/s
    _cstr_device.Vol    = reactor_volume; // volumen of reactor m3
    _cstr_device.Acat   = catalytic_area; // Catalytic area m2: chemical active area
    _cstr_device.isothermal = 0;
    _cstr_device.pressure = sv_at_i.Pressure();
    _cstr_device.Yi = real_type_1d_view("Mass fraction at inlet", _n_gas_species);
    Kokkos::deep_copy(_cstr_device.Yi, Ys);
    _cstr_device.number_of_algebraic_constraints = _n_algebraic_constraints;
    _cstr_device.poisoning_species_idx=poisoning_species_idx;

    {
      real_type_2d_view_host EnthalpyMass("EnthalpyMass", 1, _n_gas_species);
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
                                        _kmcd_gas_host);
      _cstr_device.EnthalpyIn = EnthalpyMixMass(0);
    }

  }

  void ChemElemTCSTRI_TChem::
  readDataBaseFromFile(const std::string &filename)
  {
    //read a solution from a TCSTR TChem:
    //this data is from TChem++
    //columns solution at each position
    // iter t, dt density, pressure, Temp [K], mass fraction,  site fraction
    std::vector<std::string> varnames;
    double atposition;
    const int TotalEq = _n_gas_species + _n_surface_species + 3 + 3;

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
    }
    else
    {
         std::cerr << " chem_elem_DAE_PFR_TChem.cpp: cannot open file "+ filename +"\n";
          exit(-1);
    }

    ixfs.close();

    _nBatch = cstrdb.size();
    printf("Reading data from  TCSTR solution: # state vectors %lu # Variables (including site fraction) %lu\n",cstrdb.size() ,(cstrdb[0]).size());
    /// input: state vectors: temperature, pressure and mass fraction
    const int stateVectorLen = TChem::Impl::getStateVectorSize(_n_gas_species);
    _state = real_type_2d_view("state vector", _nBatch, stateVectorLen );
    // input :: surface fraction vector, zk
    _site_fraction = real_type_2d_view("SiteFraction", _nBatch, _n_surface_species);
    /// create a mirror view to store input from a file
    auto state_host = Kokkos::create_mirror_view(_state);
    auto site_fraction_host = Kokkos::create_mirror_view(_site_fraction);
    const auto n_surface_species = _n_surface_species;
    // Note that I start the first internal loops at 1 (i=1).
    //Thus, density has a value of 0, and its value will computed by TChem.
    Kokkos::parallel_for(Kokkos::RangePolicy<TChem::host_exec_space>(0, _nBatch),
      [state_host, site_fraction_host, stateVectorLen, n_surface_species, &cstrdb](const ordinal_type& sp) {
        for (int i = 1; i < stateVectorLen  ; i++) {
          state_host(sp,i) = cstrdb[sp ][i+3];
          // printf("state i %d, %f, %f\n", i, cstrdb[sp + position][i], state_host(sp,i) );
        }
        /* site fraction */
        for (int k = 0; k < n_surface_species ; k++) {
          site_fraction_host(sp,k) = cstrdb[sp ][k + stateVectorLen + 3];
          // printf("site i %d, %f % f\n", k, cstrdb[sp + position][k+ stateVectorLen + 1], siteFraction_host(sp,k) );
        }
    });

    Kokkos::deep_copy(_state, state_host);
    Kokkos::deep_copy(_site_fraction, site_fraction_host);

  }

  void ChemElemTCSTRI_TChem::
  evalSourceVector()
  {
    Tines::ProfilingRegionScope region("CSPlib::ChemElemTCSTRI_TChem::evalSourceVector");

    CSPLIB_CHECK_ERROR(_state.span() == 0, " _state is empty");
    CSPLIB_CHECK_ERROR(_site_fraction.span() == 0, " _site_fraction is empty");

    real_type_2d_view rhs_long("whole rhs", _nBatch, _n_total_variables );

    const ordinal_type level = 1;
    const auto exec_space_instance = exec_space();

    const ordinal_type per_team_extent =
    TChem::IsothermalTransientContStirredTankReactorRHS
    ::getWorkSpaceSize(_kmcd_gas_device,_kmcd_surf_device); ///
    const ordinal_type per_team_scratch =
    Scratch<real_type_1d_view>::shmem_size(per_team_extent);
    policy_type policy(exec_space_instance, _nBatch, Kokkos::AUTO());
    policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));
    TChem::IsothermalTransientContStirredTankReactorRHS
         ::runDeviceBatch(policy,
                          _state,
                          _site_fraction,
                          rhs_long,
                          _kmcd_gas_device,
                          _kmcd_surf_device,
                          _cstr_device);
    //
    if (_n_algebraic_constraints > 0 ){

      auto reduced_rhs = Kokkos::subview(rhs_long, Kokkos::ALL(),
      Kokkos::pair<ordinal_type,ordinal_type>(0,_n_vars));
      if (_rhs.span() ==0)
         _rhs = real_type_2d_view("rhs", _nBatch, _n_vars );
      Kokkos::deep_copy(_rhs, reduced_rhs);
    } else{
      _rhs = rhs_long;
    }

  }

  void
  ChemElemTCSTRI_TChem::
  evalSacadoJacobianAndSource()
  {
    Tines::ProfilingRegionScope region("CSPlib::ChemElemTCSTRI_TChem::evalSacadoJacobianAndSource");

    CSPLIB_CHECK_ERROR(_state.span() == 0, " _state is empty");
    CSPLIB_CHECK_ERROR(_site_fraction.span() == 0, " _site_fraction is empty");

    real_type_3d_view jac_long("whole jacobian", _nBatch, _n_total_variables, _n_total_variables );
    real_type_2d_view rhs_long("whole rhs", _nBatch, _n_total_variables );

    const ordinal_type level = 1;
    const auto exec_space_instance = exec_space();

    const ordinal_type per_team_extent =
    TChem::IsothermalTransientContStirredTankReactorSacadoJacobian
    ::getWorkSpaceSize(_kmcd_gas_device,_kmcd_surf_device); ///
    const ordinal_type per_team_scratch =
    Scratch<real_type_1d_view>::shmem_size(per_team_extent);
    policy_type policy(exec_space_instance, _nBatch, Kokkos::AUTO());
    policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

    TChem::IsothermalTransientContStirredTankReactorSacadoJacobian
         ::runDeviceBatch(policy,
                          _state, //gas
                          _site_fraction,//surface
                          jac_long,
                          rhs_long,
                          _kmcd_gas_device,
                          _kmcd_surf_device,
                          _cstr_device);
    // if _n_algebraic_constraints DOE system becomes a DAE system
    if (_n_algebraic_constraints > 0 ){

      _jacobian = real_type_3d_view ("reduced jac", _nBatch, _n_vars, _n_vars );
      ComputeReducedJacobian<exec_space>
      ::runBatch(jac_long, _jacobian, _n_vars, _n_algebraic_constraints );
      // _rhs = real_type_2d_view ("reduced rhs", _nBatch, _n_vars);

      auto reduced_rhs = Kokkos::subview(rhs_long, Kokkos::ALL(),
      Kokkos::pair<ordinal_type,ordinal_type>(0,_n_vars));
      if (_rhs.span() ==0)
         _rhs = real_type_2d_view("rhs", _nBatch, _n_vars );
      // need to make because we need to produce subviews from _rhs, and
      // we cannot make a subview of a subview
      Kokkos::deep_copy(_rhs, reduced_rhs);
    } else{
      _jacobian = jac_long;
      _rhs = rhs_long;
    }

  }

  void
  ChemElemTCSTRI_TChem::
  evalNumericalJacobian()
  {
    Tines::ProfilingRegionScope region("CSPlib::ChemElemTCSTRI_TChem::evalNumericalJacobian");

    CSPLIB_CHECK_ERROR(_state.span() == 0, " _state is empty");
    CSPLIB_CHECK_ERROR(_site_fraction.span() == 0, " _site_fraction is empty");

    real_type_3d_view jac_long("whole jacobian", _nBatch, _n_total_variables, _n_total_variables );
    const ordinal_type level = 1;
    const auto exec_space_instance = exec_space();

    const ordinal_type per_team_extent =
    TChem::IsothermalTransientContStirredTankReactorNumJacobian
    ::getWorkSpaceSize(_kmcd_gas_device,_kmcd_surf_device); ///
    const ordinal_type per_team_scratch =
    Scratch<real_type_1d_view>::shmem_size(per_team_extent);
    policy_type policy(exec_space_instance, _nBatch, Kokkos::AUTO());
    policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

    TChem::IsothermalTransientContStirredTankReactorNumJacobian
         ::runDeviceBatch(policy,
                          _state, //gas
                          _site_fraction,//surface
                          jac_long,
                          _kmcd_gas_device,
                          _kmcd_surf_device,
                          _cstr_device);

    if (_n_algebraic_constraints > 0 )
    {
      _jacobian = real_type_3d_view ("reduced jac", _nBatch, _n_vars, _n_vars );
      ComputeReducedJacobian<exec_space>
      ::runBatch(jac_long, _jacobian, _n_vars, _n_algebraic_constraints );
      _rhs = real_type_2d_view ("reduced rhs", _nBatch, _n_vars);
    } else
    {
      _jacobian = jac_long;
    }

  }



  void ChemElemTCSTRI_TChem::
  getJacMatrixDevice(real_type_3d_view& jac)
  {
    Tines::ProfilingRegionScope region("CSPlib::ChemElemTCSTRI_TChem::getJacMatrixDevice");
    CSPLIB_CHECK_ERROR(_jac.span() == 0, " Jacobian should be computed: run evalSacadoJacobianAndSource()");
    jac = _jacobian;
  }

  void ChemElemTCSTRI_TChem::
  getSourceVectorDevice(real_type_2d_view& rhs)
  {
    Tines::ProfilingRegionScope region("CSPlib::ChemElemTCSTRI_TChem::getSourceVectorDevice");
    CSPLIB_CHECK_ERROR(_rhs.span() == 0, " rhs should be computed: ");
    rhs = _rhs;
  }

  void ChemElemTCSTRI_TChem::
  getStateVectorDevice(real_type_2d_view& state_vector)
  {
    Tines::ProfilingRegionScope region("CSPlib::ChemElemTCSTRI_TChem::getStateVectorDevice");
    // state_vector for csp analysis does not include variables that do not have a differencial equation such as
    // pressure and density
    CSPLIB_CHECK_ERROR(_state.span() == 0, " _state is not allocated in device");
    CSPLIB_CHECK_ERROR(_site_fraction.span() == 0, " _site_fraction is not allocated in device");
    // only mass fraction

    if (state_vector.span() == 0)
      state_vector=real_type_2d_view("x",_nBatch, _n_vars);

    auto Ys=Kokkos::subview(state_vector, Kokkos::ALL(), Kokkos::pair<int,int>(0,_n_gas_species));
    auto Zs=Kokkos::subview(state_vector, Kokkos::ALL(), Kokkos::pair<int,int>(_n_gas_species,_n_vars));
    // variable with differencial equation: mass fraction
    auto mass_fraction = Kokkos::subview(_state, Kokkos::ALL(), Kokkos::pair<int,int>(3,_state.extent(1)));
    // variable with differencial equation: surface fraction with no-gssa
    auto non_qssa_site_fraction = Kokkos::subview(_site_fraction, Kokkos::ALL(), Kokkos::pair<int,int>(0,_n_surface_equations));

    Kokkos::deep_copy(Ys, mass_fraction);
    Kokkos::deep_copy(Zs, non_qssa_site_fraction);

  }
  ordinal_type ChemElemTCSTRI_TChem::
  getNumOfElements()
  {
    return _n_elem;
  }

  void ChemElemTCSTRI_TChem::evalSmatrixDevice(const ordinal_type& team_size,
                                     const ordinal_type& vector_size)
  {
    Tines::ProfilingRegionScope region("CSPlib::ChemElemTCSTRI_TChem::evalSmatrix");
    const auto exec_space_instance = exec_space();
    const ordinal_type level = 1;
    const ordinal_type per_team_extent =
    TChem::IsothermalTransientContStirredTankReactorSmatrix::getWorkSpaceSize(_kmcd_gas_device,
    _kmcd_surf_device);
    const ordinal_type per_team_scratch =
    Scratch<real_type_1d_view>::shmem_size(per_team_extent);

    policy_type policy(exec_space_instance, _nBatch, Kokkos::AUTO());

    if ( team_size > 0 && vector_size > 0) {
        policy = policy_type(exec_space_instance, _nBatch, team_size, vector_size);
    }

    policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));
    const ordinal_type n_processes =  _n_gas_reactions + _n_surface_reactions +ordinal_type(1);
    // this matrix only includes gas species variables, and non-gass surface variables
    if (_Smat.span() == 0)
      _Smat  = real_type_3d_view("_Smat", _nBatch, _n_vars, n_processes );
    TChem::IsothermalTransientContStirredTankReactorSmatrix
         ::runDeviceBatch(policy,
                          _state, //gas
                          _site_fraction,//surface
                          _Smat,
                          // _Cmat_surface_species,
                          _kmcd_gas_device,
                          _kmcd_surf_device,
                          _cstr_device);

  }
  ordinal_type ChemElemTCSTRI_TChem::getNumOfVariables(){
       return _n_vars;
  }

  ordinal_type ChemElemTCSTRI_TChem::getNumOfSamples(){
       return _nBatch;
  }

  ordinal_type ChemElemTCSTRI_TChem::getNumOfGasSpecies(){
       return _n_gas_species;
  }

  ordinal_type ChemElemTCSTRI_TChem::getNumOfSurfaceSpecies(){
       return _n_surface_species;
  }

  ordinal_type ChemElemTCSTRI_TChem::getNumOfSurfaceReactions(){
       return _n_surface_reactions;
  }

  ordinal_type ChemElemTCSTRI_TChem::getNumOfGasReactions(){
       return _n_gas_reactions;
  }

  ordinal_type ChemElemTCSTRI_TChem::getNumOfProcesses(){
       return _n_gas_reactions + _n_surface_reactions + 1;
  }

  ordinal_type ChemElemTCSTRI_TChem::getNumOfProcesseswRevFwd(){
       return _n_rate_of_processes;
  }

  void ChemElemTCSTRI_TChem::getSpeciesNames(std::vector<std::string>& spec_name)
  {
    spec_name.clear();
    //make a copy in a std vector
    for (int k = 0; k < _n_gas_species; k++)
      spec_name.push_back(&_kmcd_gas_host.speciesNames(k,0));
    //make a copy in a std vector
    for (int k = 0; k < _n_surface_equations; k++)
      spec_name.push_back(&_kmcd_surf_host.speciesNames(k,0));
  }

  void ChemElemTCSTRI_TChem::evalRoPDevice(const ordinal_type& team_size,
                                     const ordinal_type& vector_size)
  {
  //compute rate of progress for gas phase
  _rop_fwd_gas = real_type_2d_view("Gas_Forward_RateOfProgess", _nBatch, _n_gas_reactions );
  _rop_rev_gas = real_type_2d_view("Gas_Reverse_RateOfProgess", _nBatch, _n_gas_reactions);

  const auto exec_space_instance = exec_space();
  const ordinal_type level = 1;

  {
    const ordinal_type per_team_extent =
      TChem::RateOfProgress::getWorkSpaceSize(_kmcd_gas_device);
    const ordinal_type per_team_scratch =
    Scratch<real_type_1d_view>::shmem_size(per_team_extent);

    policy_type policy(exec_space_instance, _nBatch, Kokkos::AUTO());

    if ( team_size > 0 && vector_size > 0)
      policy = policy_type(exec_space_instance, _nBatch, team_size, vector_size);

    policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

    TChem::RateOfProgress::
           runDeviceBatch(policy,
                          _state, //gas
                           //output,
                          _rop_fwd_gas, // Forward rate of progress
                          _rop_rev_gas, // reverse rate of progress
                          _kmcd_gas_device);

  }


  // compute rate of progess surface phase
  _rop_fwd_surf = real_type_2d_view("Surface_Forward_RateOfProgess", _nBatch, _n_surface_reactions);
  _rop_rev_surf = real_type_2d_view("Surface_Reverse_RateOfProgess", _nBatch, _n_surface_reactions);

  {
    const ordinal_type per_team_extent =
    TChem::RateOfProgressSurface::getWorkSpaceSize(_kmcd_gas_device, _kmcd_surf_device);
    const ordinal_type per_team_scratch =
    Scratch<real_type_1d_view>::shmem_size(per_team_extent);

    policy_type policy(exec_space_instance, _nBatch, Kokkos::AUTO());

    if ( team_size > 0 && vector_size > 0)
      policy = policy_type(exec_space_instance, _nBatch, team_size, vector_size);

    policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

    TChem::RateOfProgressSurface::
           runDeviceBatch(policy,
                          _state, //gas
                          _site_fraction,//surface
                          //output,
                          _rop_fwd_surf, // Forward rate of progress surface phase
                          _rop_rev_surf, // reverse rate of progess surface phase
                          _kmcd_gas_device,
                          _kmcd_surf_device);

  }


}

  void ChemElemTCSTRI_TChem::getSmatrixDevice(real_type_3d_view& Smatrixdb)
  {

   Tines::ProfilingRegionScope region("CSPlib::ChemElemTCSTRI_TChem::getSmatrixDevice");
   CSPLIB_CHECK_ERROR(_Smat.span() == 0, " S matrix should be computed: run evalSmatrix()");
   Smatrixdb = _Smat;
  }

  void ChemElemTCSTRI_TChem::getRoP_GasDevice(real_type_2d_view& rop_fwd_gas,
                                             real_type_2d_view& rop_rev_gas )
  {
   CSPLIB_CHECK_ERROR(_rop_fwd_gas.span() == 0, " _rop_fwd_gas should be computed: run evalRoP()");
   rop_fwd_gas = _rop_fwd_gas;
   rop_rev_gas = _rop_rev_gas;
  }

  void ChemElemTCSTRI_TChem::getRoP_SurfaceDevice(real_type_2d_view& rop_fwd_surf,
                                                  real_type_2d_view& rop_rev_surf )
  {
   CSPLIB_CHECK_ERROR(rop_fwd_surf.span() == 0, " rop_fwd_surf should be computed: run evalRoP()");
   rop_fwd_surf = _rop_fwd_surf;
   rop_rev_surf = _rop_rev_surf;
  }

  void ChemElemTCSTRI_TChem::getRoPDevice(real_type_2d_view& rop_fwd,
                                          real_type_2d_view& rop_rev )
  {
   Tines::ProfilingRegionScope region("CSPlib::ChemElemTCSTRI_TChem::getRoPDevice");
   CSPLIB_CHECK_ERROR(_rop_fwd_gas.span() == 0, " _rop_fwd_gas should be computed: run evalRoP()");

   // includes adiational process, inlet terms.
   const ordinal_type n_processes =  _n_gas_reactions + _n_surface_reactions +ordinal_type(1);
   if (rop_fwd.span() == 0)
     rop_fwd= real_type_2d_view("rop fwd", _nBatch, n_processes);
   if (rop_rev.span() == 0)
     rop_rev= real_type_2d_view("rop rev", _nBatch, n_processes);

   const real_type n_gas_surface_reactions = _n_gas_reactions + _n_surface_reactions;
   using range_type = Kokkos::pair<ordinal_type, ordinal_type>;
   real_type_2d_view rop_fwd_gas = Kokkos::subview(rop_fwd, Kokkos::ALL(), range_type(0, _n_gas_reactions));
   real_type_2d_view rop_fwd_surf = Kokkos::subview(rop_fwd, Kokkos::ALL(), range_type(_n_gas_reactions, n_gas_surface_reactions));

   real_type_2d_view rop_rev_gas = Kokkos::subview(rop_rev, Kokkos::ALL(), range_type(0, _n_gas_reactions));
   real_type_2d_view rop_rev_surf = Kokkos::subview(rop_rev, Kokkos::ALL(), range_type(_n_gas_reactions, n_gas_surface_reactions));

   Kokkos::deep_copy( rop_fwd_gas, _rop_fwd_gas);
   Kokkos::deep_copy( rop_fwd_surf, _rop_fwd_surf);
   Kokkos::deep_copy( rop_rev_gas, _rop_rev_gas);
   Kokkos::deep_copy( rop_rev_surf, _rop_rev_surf);

   Kokkos::parallel_for(
     Kokkos::RangePolicy<exec_space>(0, _nBatch),
     KOKKOS_LAMBDA(const ordinal_type& i) {
       rop_fwd(i,n_processes-1) = real_type(1);
  });


  }

}
