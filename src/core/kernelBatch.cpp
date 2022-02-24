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


#include "kernelBatch.hpp"

CSPKernelBatch::~CSPKernelBatch()
{
  freeA_View();
  freeEigenvaluesRealPartView();
  freeEigenvaluesImagPartView();
  freeB_View();
  freeCSP_PointersView();
  freeTimeScalesView();
  freeModalAmpView();
  freeM_View();
}
void CSPKernelBatch::createA_View(){
     if (_A._dev.span() == 0){
       _A._dev = real_type_3d_view("right CSP basis vectors ", _nBatch, _n_variables, _n_variables );
       _A._host = Kokkos::create_mirror_view(Kokkos::HostSpace(), _A._dev);
       _A_need_sync = NoNeedSync;
     }
}
void CSPKernelBatch::freeA_View(){
  _A._dev = real_type_3d_view();
  _A._host = real_type_3d_view_host();
  _A_need_sync = NoNeedSync;

}

void CSPKernelBatch::createEigenvaluesRealPartView(){
     if (_eigenvalues_real_part._dev.span() == 0){
       _eigenvalues_real_part._dev = real_type_2d_view("eigenvalues real part ", _nBatch, _n_variables );
       _eigenvalues_real_part._host = Kokkos::create_mirror_view(Kokkos::HostSpace(), _eigenvalues_real_part._dev);
       _eigenvalues_real_part_need_sync = NoNeedSync;
     }
}

void CSPKernelBatch::freeEigenvaluesRealPartView(){
  _eigenvalues_real_part._dev = real_type_2d_view();
  _eigenvalues_real_part._host = real_type_2d_view_host();
  _eigenvalues_real_part_need_sync = NoNeedSync;
}

void CSPKernelBatch::createEigenvaluesImagPartView(){
     if (_eigenvalues_imag_part._dev.span() == 0){
       _eigenvalues_imag_part._dev = real_type_2d_view("eigenvalues imag part ", _nBatch, _n_variables );
       _eigenvalues_imag_part._host = Kokkos::create_mirror_view(Kokkos::HostSpace(), _eigenvalues_imag_part._dev);
       _eigenvalues_imag_part_need_sync = NoNeedSync;
     }
}

void CSPKernelBatch::freeEigenvaluesImagPartView(){
  _eigenvalues_imag_part._dev = real_type_2d_view();
  _eigenvalues_imag_part._host = real_type_2d_view_host();
  _eigenvalues_imag_part_need_sync = NoNeedSync;
}

void CSPKernelBatch::evalEigenSolution(const ordinal_type& team_size,
                                       const ordinal_type& vector_size)
{
  Tines::ProfilingRegionScope region("CSPlib::evalEigenSolution");
  const int wlen = 3 * _n_variables * _n_variables + 2 * _n_variables;
  real_type_2d_view work_eigensolver("work eigen solver", _nBatch, wlen);

  const bool use_tpl_if_avail(true);
  const bool compute_and_sort_eigenpairs = true;

  createA_View();
  createEigenvaluesRealPartView();
  createEigenvaluesImagPartView();
  Kokkos::fence();

  Tines::control_type control;
  /// tpl use
  control["Bool:UseTPL"].bool_value = use_tpl_if_avail;
  /// eigen solve
  if ( team_size > 0 && vector_size > 0) {
    control["Bool:SolveEigenvaluesNonSymmetricProblem:Sort"].bool_value = compute_and_sort_eigenpairs;
    control["IntPair:Hessenberg:TeamSize"].int_pair_value = std::pair<int,int>(team_size,vector_size);
    control["IntPair:RightEigenvectorSchur:TeamSize"].int_pair_value = std::pair<int,int>(team_size,vector_size);
    control["IntPair:Gemm:TeamSize"].int_pair_value = std::pair<int,int>(team_size,vector_size);
    control["IntPair:SortRightEigenPairs:TeamSize"].int_pair_value = std::pair<int,int>(team_size,vector_size);
  } // else use tines default values


  Tines::SolveEigenvaluesNonSymmetricProblemDevice<exec_space>
       ::invoke( exec_space(), _Jacobian, _eigenvalues_real_part._dev,
                _eigenvalues_imag_part._dev, _A._dev, work_eigensolver,
                 control);

  _A_need_sync = NeedSyncToHost;
  _eigenvalues_real_part_need_sync = NeedSyncToHost;
  _eigenvalues_imag_part_need_sync = NeedSyncToHost;

  work_eigensolver =  real_type_2d_view();

}

void CSPKernelBatch::sortEigenSolution(const ordinal_type& team_size,
                                       const ordinal_type& vector_size)
{
  Tines::ProfilingRegionScope region("CSPlib::sortEigenSolution");
  CSPLIB_CHECK_ERROR(A._dev.span() == 0, " Right CSP basis vectors should be computed and sorted: run evalEigenSolution()");

  const int wlen = 3 * _n_variables * _n_variables + 2 * _n_variables;
  real_type_2d_view work_eigensolver("work eigen solver", _nBatch, wlen);
  Kokkos::fence();


  Tines::SortRightEigenPairsDevice<exec_space>
       ::invoke(exec_space(), _eigenvalues_real_part._dev,
                _eigenvalues_imag_part._dev, _A._dev, work_eigensolver);


  work_eigensolver =  real_type_2d_view();
}

void CSPKernelBatch::createB_View(){
     if (_B._dev.span() == 0){
       _B._dev = real_type_3d_view("left CSP basis vectors ", _nBatch, _n_variables, _n_variables );
       _B._host = Kokkos::create_mirror_view(Kokkos::HostSpace(), _B._dev);
       _B_need_sync = NoNeedSync;
     }
}

void CSPKernelBatch::freeB_View(){
  _B._dev = real_type_3d_view();
  _B._host = real_type_3d_view_host();
  _B_need_sync = NoNeedSync;
}

void CSPKernelBatch::evalCSPbasisVectors(const ordinal_type& team_size,
                                         const ordinal_type& vector_size)
{
  Tines::ProfilingRegionScope region("CSPlib::evalCSPbasisVectors");
  CSPLIB_CHECK_ERROR(A._dev.span() == 0, " Right CSP basis vectors should be computed and sorted: run evalEigenSolution() and sortEigenSolution() ");
  createB_View();


  real_type_2d_view work(" work ", _nBatch, 2*_n_variables + _n_variables * _n_variables );
  policy_type policy(exec_space(), _nBatch, Kokkos::AUTO());
  if ( team_size > 0 && vector_size > 0) {
    policy = policy_type(exec_space(), _nBatch, team_size, vector_size);
  }
  Kokkos::fence();

  kernel_csplib_device::evalLeftCSP_BasisVectorsBatch(
                        "CSPlib::computeCSPbasisVectors::runDeviceBatch",
                        policy, _A._dev, _B._dev, work);


  _B_need_sync = NeedSyncToHost;
  work = real_type_2d_view();
}

void CSPKernelBatch::createCSP_PointersView(){
     if (_CSP_pointers._dev.span() == 0){
       _CSP_pointers._dev = real_type_3d_view("CSP pointers ", _nBatch, _n_variables, _n_variables );
       _CSP_pointers._host = Kokkos::create_mirror_view(Kokkos::HostSpace(), _CSP_pointers._dev);
       _CSP_pointers_need_sync = NoNeedSync;
     }
}


void CSPKernelBatch::freeCSP_PointersView(){
  _CSP_pointers._dev = real_type_3d_view();
  _CSP_pointers._host = real_type_3d_view_host();
  _CSP_pointers_need_sync = NoNeedSync;
}

void CSPKernelBatch::evalCSP_Pointers(const ordinal_type& team_size,
                                      const ordinal_type& vector_size)
{
  Tines::ProfilingRegionScope region("CSPlib::evalCSP_Pointers");

  CSPLIB_CHECK_ERROR(A._dev.span() == 0, " Right CSP basis vectors should be computed and sorted: run evalEigenSolution() and sortEigenSolution() ");

  policy_type policy(exec_space(), _nBatch, Kokkos::AUTO());
  if ( team_size > 0 && vector_size > 0) {
    policy = policy_type(exec_space(), _nBatch, team_size, vector_size);
  }

  createCSP_PointersView();
  Kokkos::fence();

  kernel_csplib_device
  ::evalCSPPointersBatch("CSPlib::evalCSPPointers::runDeviceBatch",
                         policy,
                         _A._dev, // input : right CSP basis vectors
                         _B._dev,// input : letf CSP basis vectors
                         _CSP_pointers._dev);


  _CSP_pointers_need_sync = NeedSyncToHost;
}

void CSPKernelBatch::createTimeScalesView()
{
     if (_time_scales._dev.span() == 0){
       _time_scales._dev = real_type_2d_view("Time scales", _nBatch, _n_variables);
       _time_scales._host = Kokkos::create_mirror_view(Kokkos::HostSpace(), _time_scales._dev);
       _time_scales_need_sync = NoNeedSync;
     }
}


void CSPKernelBatch::freeTimeScalesView()
{
  _time_scales._dev = real_type_2d_view();
  _time_scales._host = real_type_2d_view_host();
  _time_scales_need_sync = NoNeedSync;
}

void CSPKernelBatch::evalTimeScales(const ordinal_type& team_size,
                                    const ordinal_type& vector_size)
{
  Tines::ProfilingRegionScope region("CSPlib::evalTimeScales");
  policy_type policy(exec_space(), _nBatch, Kokkos::AUTO());
  if ( team_size > 0 && vector_size > 0) {
    policy = policy_type(exec_space(), _nBatch, team_size, vector_size);
  }
  createTimeScalesView();
  Kokkos::fence();

  kernel_csplib_device
  ::evalTimeScalesBatch("CSPlib::evalTimeScales::runDeviceBatch",
                        policy,
                        _eigenvalues_real_part._dev, // input
                        _eigenvalues_imag_part._dev, //input
                        _time_scales._dev); // output


  _time_scales_need_sync = NeedSyncToHost;

}

void CSPKernelBatch::createModalAmpView()
{
     if (_modal_amplitude._dev.span() == 0){
       _modal_amplitude._dev = real_type_2d_view("modal amplitude", _nBatch, _n_variables);
       _modal_amplitude._host = Kokkos::create_mirror_view(Kokkos::HostSpace(), _modal_amplitude._dev);
       _modal_amplitude_need_sync = NoNeedSync;
     }
}



void CSPKernelBatch::freeModalAmpView()
{
  _modal_amplitude._dev = real_type_2d_view();
  _modal_amplitude._host = real_type_2d_view_host();
  _modal_amplitude_need_sync = NoNeedSync;
}

void CSPKernelBatch::evalModalAmp(const ordinal_type& team_size,
                                  const ordinal_type& vector_size)
{
  Tines::ProfilingRegionScope region("CSPlib::evalModalAmp");

  policy_type policy(exec_space(), _nBatch, Kokkos::AUTO());
  if ( team_size > 0 && vector_size > 0) {
    policy = policy_type(exec_space(), _nBatch, team_size, vector_size);
  }
  createModalAmpView();
  Kokkos::fence();

  kernel_csplib_device
  ::evalModalAmpBatch("CSPlib::evalModalAmp::runDeviceBatch",
                     policy,
                     _B._dev,// input : letf CSP basis vectors
                     _rhs,// input
                     _modal_amplitude._dev);

  _modal_amplitude_need_sync = NeedSyncToHost;

}


void CSPKernelBatch::createM_View()
{
     if (_M._dev.span() == 0){
       _M._dev = ordinal_type_1d_view("M", _nBatch);
       _M._host = Kokkos::create_mirror_view(Kokkos::HostSpace(), _M._dev);
       _M_need_sync = NoNeedSync;
     }
}

void CSPKernelBatch::freeM_View()
{
  _M._dev = ordinal_type_1d_view();
  _M._host = ordinal_type_1d_view_host();
  _M_need_sync = NoNeedSync;
}

void CSPKernelBatch::evalM(const ordinal_type& team_size,
                           const ordinal_type& vector_size)
{
  Tines::ProfilingRegionScope region("CSPlib::evalM");

  CSPLIB_CHECK_ERROR(_modal_amplitude._dev.span() == 0, " modal amplitude should be computed: run evalModalAmp()");
  CSPLIB_CHECK_ERROR(_time_scales._dev.span() == 0, " Time scales should be computed: run evalTimeScales()");
  CSPLIB_CHECK_ERROR(A._dev.span() == 0,
   " Right CSP basis vectors should be computed and sorted: run evalEigenSolution() and sortEigenSolution() ");
  real_type_2d_view error_csp(" error csp vector", _nBatch, _n_variables);
  real_type_2d_view delta_yfast("delta yfast", _nBatch, _n_variables);

  createM_View();

  policy_type policy(exec_space(), _nBatch, Kokkos::AUTO());
  if ( team_size > 0 && vector_size > 0) {
    policy = policy_type(exec_space(), _nBatch, team_size, vector_size);
  }
  Kokkos::fence();

  kernel_csplib_device
  ::evalMBatch("CSPlib::evalM::runDeviceBatch",
               policy,
               _A._dev, // input : right CSP basis vectors
               _eigenvalues_real_part._dev, // input
               _modal_amplitude._dev, // input
               _time_scales._dev,
               _state_vector,
               _nElem, //input
               _csp_rel_tol, // input relative tolerance
               _csp_abs_tol, // input absolute tolerance
               error_csp, // work
               delta_yfast,
               _M._dev);


  _M_need_sync = NeedSyncToHost;

  // free work space
  error_csp =  real_type_2d_view();
}

CSPKernelBatch::real_type_3d_view CSPKernelBatch::getLeftCSPVecDevice()
{
 CSPLIB_CHECK_ERROR(B._dev.span() == 0, " Left CSP basis vectors should be computed: run evalCSPbasisVectors()");
 return _B._dev;
}

CSPKernelBatch::real_type_3d_view CSPKernelBatch::getRightCSPVecDevice()
{
 CSPLIB_CHECK_ERROR(A._dev.span() == 0,
   " Right CSP basis vectors should be computed and sorted: run evalEigenSolution() and sortEigenSolution() ");
 return _A._dev;
}

CSPKernelBatch::real_type_3d_view CSPKernelBatch::getCSPPointersDevice()
{
  Tines::ProfilingRegionScope region("CSPlib::getCSPPointersDevice");
  return _CSP_pointers._dev;
}

CSPKernelBatch::real_type_3d_view_host CSPKernelBatch::getCSPPointers()
{
  CSPLIB_CHECK_ERROR(_CSP_pointers._dev.span() == 0, " CSP pointers should be computed: run evalCSP_Pointers()");

  if (_CSP_pointers_need_sync == NeedSyncToHost) {
    Kokkos::deep_copy(_CSP_pointers._host, _CSP_pointers._dev);
    _CSP_pointers_need_sync = NoNeedSync;
  }
  return _CSP_pointers._host;
}

CSPKernelBatch::real_type_2d_view_host CSPKernelBatch::getTimeScales()
{
  Tines::ProfilingRegionScope region("CSPlib::evalTimeScales");
  CSPLIB_CHECK_ERROR(_time_scales._dev.span() == 0, " Time scales should be computed: run evalTimeScales()");

  if (_time_scales_need_sync == NeedSyncToHost) {
    Kokkos::deep_copy(_time_scales._host, _time_scales._dev);
    _time_scales_need_sync = NoNeedSync;
  }
  return _time_scales._host;
}

CSPKernelBatch::real_type_2d_view_host CSPKernelBatch::getModalAmp()
{
  Tines::ProfilingRegionScope region("CSPlib::getModalAmp");
  CSPLIB_CHECK_ERROR(_modal_amplitude._dev.span() == 0, " modal amplitude should be computed: run evalModalAmp()");

  if (_modal_amplitude_need_sync == NeedSyncToHost) {
    Kokkos::deep_copy(_modal_amplitude._host, _modal_amplitude._dev);
    _modal_amplitude_need_sync = NoNeedSync;
  }
  return _modal_amplitude._host;
}

CSPKernelBatch::ordinal_type_1d_view CSPKernelBatch::getMDevice()
{
  Tines::ProfilingRegionScope region("CSPlib::getMDevice");
  CSPLIB_CHECK_ERROR(_M._dev.span() == 0, " M, number of exhausted modes, should be computed: run evalM()");
  return _M._dev;
}

CSPKernelBatch::ordinal_type_1d_view_host CSPKernelBatch::getM()
{
  Tines::ProfilingRegionScope region("CSPlib::evalM");
  CSPLIB_CHECK_ERROR(_M._dev.span() == 0, "M, number of exhausted modes, should be computed: run evalM()");

  if (_M_need_sync == NeedSyncToHost) {
    Kokkos::deep_copy(_M._host, _M._dev);
    _M_need_sync = NoNeedSync;
  }
  return _M._host;
}

CSPKernelBatch::real_type_2d_view_host CSPKernelBatch::getEigenValuesRealPart()
{
  CSPLIB_CHECK_ERROR(_eigenvalues_real_part._dev.span() == 0, "eigen values real part should be computed: run evalEigenSolution()");

  if (_eigenvalues_real_part_need_sync == NeedSyncToHost) {
    Kokkos::deep_copy(_eigenvalues_real_part._host, _eigenvalues_real_part._dev);
    _eigenvalues_real_part_need_sync = NoNeedSync;
  }
  return _eigenvalues_real_part._host;
}

CSPKernelBatch::real_type_2d_view_host CSPKernelBatch::getEigenValuesImagPart()
{
  CSPLIB_CHECK_ERROR(_eigenvalues_imag_part._dev.span() == 0, "eigen values imag part should be computed: run evalEigenSolution()");

  if (_eigenvalues_imag_part_need_sync == NeedSyncToHost) {
    Kokkos::deep_copy(_eigenvalues_imag_part._host, _eigenvalues_imag_part._dev);
    _eigenvalues_imag_part_need_sync = NoNeedSync;
  }
  return _eigenvalues_imag_part._host;
}
