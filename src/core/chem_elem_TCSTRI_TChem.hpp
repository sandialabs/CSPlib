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


#ifndef MODEL_CHEM_TCSTRI_TCHEM
#define MODEL_CHEM_TCSTRI_TCHEM

#include "TChem.hpp"
#include "Tines.hpp"
#include "tools_tines.hpp"
#include "util.hpp"

using ordinal_type = TChem::ordinal_type;
using real_type = TChem::real_type;
using real_type_1d_view = TChem::real_type_1d_view;
using real_type_2d_view = TChem::real_type_2d_view;
using real_type_3d_view = TChem::real_type_3d_view;

using real_type_0d_view_host = TChem::real_type_0d_view_host;
using real_type_1d_view_host = TChem::real_type_1d_view_host;
using real_type_2d_view_host = TChem::real_type_2d_view_host;

using interf_device_type = TChem::interf_device_type;
using interf_host_device_type = TChem::interf_host_device_type;

namespace CSP {

struct ChemElemTCSTRI_TChem {
  public:
    /// this policy is for time integration which can be repeatedly reused
    using policy_type = typename TChem::UseThisTeamPolicy<exec_space>::type;
  private:
    ordinal_type _n_elem, _n_gas_species, _n_gas_reactions, _n_surface_reactions, _n_surface_species;
    ordinal_type _n_vars, _n_algebraic_constraints, _n_total_variables;
    ordinal_type _n_surface_equations, _n_rate_of_processes;

    TChem::KineticModelData _kmd;
    // device TChem object
    KineticSurfModelConstData<interf_device_type> _kmcd_surf_device;
    KineticModelConstData<interf_device_type> _kmcd_gas_device;

    //host TChem object
    KineticSurfModelConstData<interf_host_device_type> _kmcd_surf_host;
    KineticModelConstData<interf_host_device_type> _kmcd_gas_host;

    TransientContStirredTankReactorData<interf_device_type> _cstr_device;

    real_type_2d_view _site_fraction;

    real_type_2d_view _state;
    real_type_2d_view _rop_fwd_gas, _rop_rev_gas, _rop_fwd_surf, _rop_rev_surf;
    real_type_3d_view _Smat/*,_Cmat_surface_species*/;
    // real_type_2d_view _s_inlet;
    ordinal_type _nBatch;
    real_type_2d_view _rhs;
    real_type_3d_view _jacobian;

  public:
    ChemElemTCSTRI_TChem();
    ChemElemTCSTRI_TChem(const std::string &mech_gas_file,
                         const std::string &mech_surf_file,
                         const std::string &thermo_gas_file,
                         const std::string &thermo_surf_file,
                         const ordinal_type& Nalgebraic_constraints,
                         const bool  use_yaml=false );
    ~ChemElemTCSTRI_TChem();
    // when using sacado, tchem computes both jacobian and source term
    void evalSacadoJacobianAndSource();
    void getJacMatrixDevice(real_type_3d_view& jac);
    ordinal_type getNumOfElements();
    void getSourceVectorDevice(real_type_2d_view& rhs);
    void readDataBaseFromFile(const std::string &filename);
    void getStateVectorDevice(real_type_2d_view& state_vector);
    void setCSTR( const std::string& input_condition_file_name,
             const double& inlet_mass_flow,  const double& reactor_volume,
             const double& catalytic_area, const int& poisoning_species_idx );
    void evalSmatrixDevice(const ordinal_type& team_size=-1,
                           const ordinal_type& vector_size=-1);
    void evalRoPDevice(const ordinal_type& team_size=-1,
                 const ordinal_type& vector_size=-1);
    void getSmatrixDevice(real_type_3d_view& Smatrixdb);
    void getRoPDevice(real_type_2d_view& rop_fwd, real_type_2d_view& rop_rev );
    ordinal_type getNumOfVariables();
    ordinal_type getNumOfSamples();
    ordinal_type getNumOfGasSpecies();
    ordinal_type getNumOfSurfaceSpecies();
    ordinal_type getNumOfSurfaceReactions();
    ordinal_type getNumOfGasReactions();
    ordinal_type getNumOfProcesses();
    ordinal_type getNumOfProcesseswRevFwd();
    void getSpeciesNames(std::vector<std::string>& spec_name);
    void evalSourceVector();
    void evalNumericalJacobian();
    void getRoP_SurfaceDevice(real_type_2d_view& rop_fwd_surf,
                              real_type_2d_view& rop_rev_surf );
    void getRoP_GasDevice(real_type_2d_view& rop_fwd_gas,
                          real_type_2d_view& rop_rev_gas );
};
}

#endif
