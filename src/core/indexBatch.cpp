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


#include "indexBatch.hpp"

CSPIndexBatch::~CSPIndexBatch()
{
  freeBetaView();
  freeAlphaView();
  freeGammaView();
  freeParticipationIndexView();
  freeWorkView();
  freeSlowImportanceIndexView();
  freeFastImportantIndexView();
}

void CSPIndexBatch::createBetaView(){
     if (_Beta.span() == 0)
        _Beta = real_type_3d_view("Beta",_nBatch, _n_variables, _n_processes );
}

void CSPIndexBatch::evalBeta(const ordinal_type& team_size,
                             const ordinal_type& vector_size)
{
  Tines::ProfilingRegionScope region("CSPlib::evalBeta");
  createBetaView();
  using policy_type = Kokkos::TeamPolicy<exec_space>;
  policy_type policy(_nBatch, Kokkos::AUTO()); // fine
  if ( team_size > 0 && vector_size > 0) {
   policy = policy_type(exec_space(), _nBatch, team_size, vector_size);
  }
  Kokkos::fence();

  index_csplib_device::evalBetaBatch("CSPlib::evalBeta::runDeviceBatch",
                policy,
                _B, // input
                _S, // input
                _Beta);

} // end of evalBeta


void CSPIndexBatch::createAlphaView(){
     if (_Alpha.span() == 0)
        _Alpha = real_type_3d_view("Alpha",_nBatch, _n_variables, _n_processes );
}

void CSPIndexBatch::freeAlphaView(){
   if (_Alpha.span() > 0)
     _Alpha = real_type_3d_view();
}

void CSPIndexBatch::evalAlpha(const ordinal_type& team_size,
                              const ordinal_type& vector_size)
{

  Tines::ProfilingRegionScope region("CSPlib::evalAlpha");
  if (_Beta.span() == 0)
      evalBeta(team_size, vector_size);
  createAlphaView();
  Kokkos::fence();

  using policy_type = Kokkos::TeamPolicy<exec_space>;
  policy_type policy(_nBatch, Kokkos::AUTO()); // fine
  if ( team_size > 0 && vector_size > 0) {
   policy = policy_type(exec_space(), _nBatch, team_size, vector_size);
  }

  index_csplib_device::evalAlphaBatch("CSPlib::evalAlpha::runDeviceBatch",
                policy,
                _Beta, // input
                _A, // input
                _M,
                _Alpha);


} // end of evalBeta

void CSPIndexBatch::createGammaView(){
     if (_Gamma.span() == 0)
       _Gamma = real_type_3d_view("Gamma",_nBatch, _n_variables, _n_processes );
}


void CSPIndexBatch::freeGammaView(){
  if (_Gamma.span() > 0)
    _Gamma = real_type_3d_view();
}

void CSPIndexBatch::evalGamma(const ordinal_type& team_size,
                              const ordinal_type& vector_size)
{
  Tines::ProfilingRegionScope region("CSPlib::evalGamma");
  if (_Beta.span() == 0)
      evalBeta(team_size,vector_size);
  createGammaView();


  using policy_type = Kokkos::TeamPolicy<exec_space>;
  policy_type policy(_nBatch, Kokkos::AUTO()); // fine
  if ( team_size > 0 && vector_size > 0) {
   policy = policy_type(exec_space(), _nBatch, team_size, vector_size);
  }

  Kokkos::fence();


  index_csplib_device::evalGammaBatch("CSPlib::evalGamma::runDeviceBatch",
                policy,
                _Beta, // input
                _A, // input
                _M,
                _Gamma);


} // end of evalBeta

void CSPIndexBatch::createParticipationIndexView(){
     _ParticipationIndex._dev = real_type_3d_view("ParticipationIndex",_nBatch, _n_variables, _n_total_processes);
     _ParticipationIndex._host = Kokkos::create_mirror_view(Kokkos::HostSpace(), _ParticipationIndex._dev);
     _ParticipationIndex_need_sync = NoNeedSync;
}

void CSPIndexBatch::freeParticipationIndexView(){
     if (_ParticipationIndex._dev.span() > 0)
       _ParticipationIndex._dev = real_type_3d_view();
     if (_ParticipationIndex._host.span() > 0)
       _ParticipationIndex._host = real_type_3d_view_host();
}

void CSPIndexBatch::createWorkView(){
     if (_work.span() == 0)
       _work = real_type_2d_view("work",_nBatch, _n_variables );
}



void CSPIndexBatch::freeWorkView(){
  if (_work.span() > 0)
    _work = real_type_2d_view();
}

void CSPIndexBatch::freeBetaView(){
  if (_Beta.span() > 0)
    _Beta = real_type_3d_view();
}


void CSPIndexBatch::evalParticipationIndex(const ordinal_type& team_size,
                                           const ordinal_type& vector_size)
{
    Tines::ProfilingRegionScope region("CSPlib::evalParticipationIndex");

     policy_type policy(exec_space(), _nBatch, Kokkos::AUTO());
     if ( team_size > 0 && vector_size > 0) {
      policy = policy_type(exec_space(), _nBatch, team_size, vector_size);
     }

     if (_Beta.span() == 0)
         evalBeta(team_size, vector_size);
    if (_ParticipationIndex._dev.span() == 0){
      createParticipationIndexView();
    }

     createWorkView();

     Kokkos::fence();

     index_csplib_device::evalCSPIndexBatch("CSPlib::evalParticipationIndex::runDeviceBatch",
                                            policy,
                                            _Beta, // input
                                            _RoP, // input
                                            _ParticipationIndex._dev, //output
                                            _work);  // work

    _ParticipationIndex_need_sync = NeedSyncToHost;
}

void CSPIndexBatch::evalParticipationIndexFwdAndRev(const ordinal_type& team_size,
                                           const ordinal_type& vector_size)
{
    Tines::ProfilingRegionScope region("CSPlib::evalParticipationIndex");

     policy_type policy(exec_space(), _nBatch, Kokkos::AUTO());
     if ( team_size > 0 && vector_size > 0) {
      policy = policy_type(exec_space(), _nBatch, team_size, vector_size);
     }

     if (_Beta.span() == 0)
         evalBeta(team_size, vector_size);
     if (_ParticipationIndex._dev.span() == 0)
        createParticipationIndexView();

     createWorkView();

     Kokkos::fence();

     // negative because _RoP_rev is not negative
     const real_type factor(-1);
     index_csplib_device::evalCSPIndexBatch("CSPlib::evalParticipationIndex::runDeviceBatch",
                                            policy,
                                            _Beta, // input
                                            _RoP_fwd, // input
                                            _RoP_rev, // input
                                            factor,
                                            _ParticipationIndex._dev, //output
                                            _work);  // work


    _ParticipationIndex_need_sync = NeedSyncToHost;
}

void CSPIndexBatch::getParticipationIndex(
       std::vector<std::vector< std::vector< double > > > &PartIndex ) {
       CSPLIB_CHECK_ERROR(_ParticipationIndex._dev.span() == 0, " Partocipation index should be computed: run evalParticipationIndex()");

       if (_ParticipationIndex_need_sync == NeedSyncToHost) {
         Kokkos::deep_copy(_ParticipationIndex._host, _ParticipationIndex._dev);
         _ParticipationIndex_need_sync = NoNeedSync;
       }

       PartIndex.clear();
       PartIndex = std::vector< std::vector< std::vector< double > > >
       (_ParticipationIndex._dev.extent(0), std::vector<std::vector<double> > (_ParticipationIndex._dev.extent(1),
        std::vector<double>(_ParticipationIndex._dev.extent(2),0)));
       Tines::convertToStdVector(PartIndex, _ParticipationIndex._dev);
}

CSPIndexBatch::real_type_3d_view_host CSPIndexBatch::getParticipationIndex(){
  Tines::ProfilingRegionScope region("CSPlib::getParticipationIndex");
  CSPLIB_CHECK_ERROR(_ParticipationIndex._dev.span() == 0, " Partocipation index should be computed: run evalParticipationIndex()");

  if (_ParticipationIndex_need_sync == NeedSyncToHost) {
    Kokkos::deep_copy(_ParticipationIndex._host, _ParticipationIndex._dev);
    _ParticipationIndex_need_sync = NoNeedSync;
  }

  return _ParticipationIndex._host;

}

void CSPIndexBatch::createSlowImportanceIndexView(){
     _SlowImportanceIndex._dev = real_type_3d_view("_SlowImportanceIndex",_nBatch, _n_variables, _n_total_processes);
     _SlowImportanceIndex._host = Kokkos::create_mirror_view(Kokkos::HostSpace(), _SlowImportanceIndex._dev);
     _SlowImportanceIndex_need_sync = NoNeedSync;
}



void CSPIndexBatch::freeSlowImportanceIndexView(){
    if (_SlowImportanceIndex._dev.span() > 0)
      _SlowImportanceIndex._dev = real_type_3d_view();
    if (_SlowImportanceIndex._host.span() > 0)
      _SlowImportanceIndex._host = real_type_3d_view_host();
     _SlowImportanceIndex_need_sync = NoNeedSync;
}

void CSPIndexBatch::evalImportanceIndexSlow(const ordinal_type& team_size,
                                            const ordinal_type& vector_size)
     {
     Tines::ProfilingRegionScope region("CSPlib::evalImportanceIndexSlow");
     policy_type policy(exec_space(), _nBatch, Kokkos::AUTO());
     if ( team_size > 0 && vector_size > 0) {
       policy = policy_type(exec_space(), _nBatch, team_size, vector_size);
     }

     if (_Alpha.span() == 0)
         evalAlpha(team_size, vector_size);
     if (_SlowImportanceIndex._dev.span() == 0){
       createSlowImportanceIndexView();
     }

     createWorkView();
     Kokkos::fence();


     index_csplib_device::evalCSPIndexBatch("CSPlib::evalSlowImportanceIndex::runDeviceBatch",
                                            policy,
                                            _Alpha, // input
                                            _RoP, // input
                                            _SlowImportanceIndex._dev, //output
                                            _work);  // work


    _SlowImportanceIndex_need_sync = NeedSyncToHost;
}

void CSPIndexBatch::evalImportanceIndexSlowFwdAndRev(const ordinal_type& team_size,
                                            const ordinal_type& vector_size)
{
     Tines::ProfilingRegionScope region("CSPlib::evalImportanceIndexSlow");
     policy_type policy(exec_space(), _nBatch, Kokkos::AUTO());
     if ( team_size > 0 && vector_size > 0) {
       policy = policy_type(exec_space(), _nBatch, team_size, vector_size);
     }

     if (_Alpha.span() == 0)
         evalAlpha(team_size, vector_size);
     if (_SlowImportanceIndex._dev.span() == 0){
       createSlowImportanceIndexView();
     }

     createWorkView();
     Kokkos::fence();

     // negative because _RoP_rev is not negative
     const real_type factor(-1);
     index_csplib_device::evalCSPIndexBatch("CSPlib::evalSlowImportanceIndex::runDeviceBatch",
                                            policy,
                                            _Alpha, // input
                                            _RoP_fwd, // input
                                            _RoP_rev,
                                            factor,
                                            _SlowImportanceIndex._dev, //output
                                            _work);  // work

     _SlowImportanceIndex_need_sync = NeedSyncToHost;
}

void CSPIndexBatch::getImportanceIndexSlow(
       std::vector<std::vector< std::vector< double > > > &Islow ) {
       CSPLIB_CHECK_ERROR(_SlowImportanceIndex._dev.span() == 0, " Slow importance index should be computed: run evalImportanceIndexSlow()");

       if (_SlowImportanceIndex_need_sync == NeedSyncToHost) {
         Kokkos::deep_copy(_SlowImportanceIndex._host, _SlowImportanceIndex._dev);
         _SlowImportanceIndex_need_sync = NoNeedSync;
       }

       Islow.clear();
       Islow = std::vector< std::vector< std::vector< double > > >
       (_SlowImportanceIndex._dev.extent(0), std::vector<std::vector<double> > (_SlowImportanceIndex._dev.extent(1),
        std::vector<double>(_SlowImportanceIndex._dev.extent(2),0)));
       Tines::convertToStdVector(Islow, _SlowImportanceIndex._dev);
}

CSPIndexBatch::real_type_3d_view_host CSPIndexBatch::getImportanceIndexSlow(){
  Tines::ProfilingRegionScope region("CSPlib::getImportanceIndexSlow");
  CSPLIB_CHECK_ERROR(_SlowImportanceIndex._dev.span() == 0, " Slow importance index should be computed: run evalImportanceIndexSlow()");

  if (_SlowImportanceIndex_need_sync == NeedSyncToHost) {
    Kokkos::deep_copy(_SlowImportanceIndex._host, _SlowImportanceIndex._dev);
    _SlowImportanceIndex_need_sync = NoNeedSync;
  }

  return _SlowImportanceIndex._host;
}

void CSPIndexBatch::createFastImportantIndexView(){
     _FastImportanceIndex._dev = real_type_3d_view("FastImportanceIndex",_nBatch, _n_variables, _n_total_processes);
     _FastImportanceIndex._host = Kokkos::create_mirror_view(Kokkos::HostSpace(), _FastImportanceIndex._dev);
     _FastImportanceIndex_need_sync = NoNeedSync;

}

void CSPIndexBatch::freeFastImportantIndexView(){
     if (_FastImportanceIndex._dev.span() > 0)
       _FastImportanceIndex._dev = real_type_3d_view();
    if (_FastImportanceIndex._host.span() > 0)
     _FastImportanceIndex._host = real_type_3d_view_host();
}

void CSPIndexBatch::evalImportanceIndexFast(const ordinal_type& team_size,
                                            const ordinal_type& vector_size)
{
     Tines::ProfilingRegionScope region("CSPlib::evalImportanceIndexFast");
     policy_type policy(exec_space(), _nBatch, Kokkos::AUTO());
     if ( team_size > 0 && vector_size > 0) {
      policy = policy_type(exec_space(), _nBatch, team_size, vector_size);
     }

     if (_Gamma.span() == 0)
         evalGamma(team_size, vector_size);

     if (_FastImportanceIndex._dev.span() == 0) {
       createFastImportantIndexView();
     }

     createWorkView();
     Kokkos::fence();

     index_csplib_device::evalCSPIndexBatch("CSPlib::evalFastImportantIndex::runDeviceBatch",
                                            policy,
                                            _Gamma, // input
                                            _RoP, // input
                                            _FastImportanceIndex._dev, //output
                                            _work);  // work

    _FastImportanceIndex_need_sync = NeedSyncToHost;
}

void CSPIndexBatch::evalImportanceIndexFastFwdAndRev(const ordinal_type& team_size,
                                            const ordinal_type& vector_size)
{
     Tines::ProfilingRegionScope region("CSPlib::evalImportanceIndexFast");
     policy_type policy(exec_space(), _nBatch, Kokkos::AUTO());
     if ( team_size > 0 && vector_size > 0) {
      policy = policy_type(exec_space(), _nBatch, team_size, vector_size);
     }

     if (_Gamma.span() == 0)
         evalGamma(team_size, vector_size);

     if (_FastImportanceIndex._dev.span() == 0) {
       createFastImportantIndexView();
     }

     createWorkView();
     Kokkos::fence();

    // negative because _RoP_rev is not negative
     const real_type factor(-1);
     index_csplib_device::evalCSPIndexBatch("CSPlib::evalFastImportantIndex::runDeviceBatch",
                                            policy,
                                            _Gamma, // input
                                            _RoP_fwd, // input
                                            _RoP_rev,
                                            factor,
                                            _FastImportanceIndex._dev, //output
                                            _work);  // work

    _FastImportanceIndex_need_sync = NeedSyncToHost;
}

void CSPIndexBatch::getImportanceIndexFast(
       std::vector<std::vector< std::vector< double > > > &Ifast )
{

       CSPLIB_CHECK_ERROR(_FastImportanceIndex._dev.span() == 0, " Fast importance index should be computed: run evalImportanceIndexFast()");

       if (_FastImportanceIndex_need_sync == NeedSyncToHost) {
         Kokkos::deep_copy(_FastImportanceIndex._host, _FastImportanceIndex._dev);
         _FastImportanceIndex_need_sync = NoNeedSync;
       }

       Ifast.clear();

       Ifast = std::vector< std::vector< std::vector< double > > >
       (_FastImportanceIndex._host.extent(0), std::vector<std::vector<double> > (_FastImportanceIndex._host.extent(1),
        std::vector<double>(_FastImportanceIndex._host.extent(2),0)));
       Tines::convertToStdVector(Ifast, _FastImportanceIndex._host);
}

CSPIndexBatch::real_type_3d_view_host CSPIndexBatch::getImportanceIndexFast(){
  Tines::ProfilingRegionScope region("CSPlib::evalImportanceIndexFast");
  CSPLIB_CHECK_ERROR(_FastImportanceIndex._dev.span() == 0, " Fast importance index should be computed: run evalImportanceIndexFast()");

  if (_FastImportanceIndex_need_sync == NeedSyncToHost) {
    Kokkos::deep_copy(_FastImportanceIndex._host, _FastImportanceIndex._dev);
    _FastImportanceIndex_need_sync = NoNeedSync;
  }

  return _FastImportanceIndex._host;
}
