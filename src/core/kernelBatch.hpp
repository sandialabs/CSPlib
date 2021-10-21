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


#ifndef KERNEL_BATCH_CSP
#define KERNEL_BATCH_CSP
#include "Tines.hpp"
#include "util.hpp"
#include "kernel_kokkos.hpp"
#include "CSPlib_ConfigDefs.h"

// #define CSPLIB_MESUARE_WALL_TIME

class CSPKernelBatch
{

 private:
   using real_type = typename Tines::ats<double>::scalar_type;

   using host_exec_space = Kokkos::DefaultHostExecutionSpace;
   using exec_space = Kokkos::DefaultExecutionSpace;
   using host_device_type = typename Tines::UseThisDevice<host_exec_space>::type;
   using device_type      = typename Tines::UseThisDevice<exec_space>::type;

   using kernel_csplib_host = CSP::KernelComputation<host_device_type>;
   using kernel_csplib_device = CSP::KernelComputation<device_type>;

   using policy_type = Kokkos::TeamPolicy<exec_space>;

   using real_type_1d_view = Tines::value_type_1d_view<real_type,device_type>;
   using real_type_2d_view = Tines::value_type_2d_view<real_type,device_type>;
   using real_type_3d_view = Tines::value_type_3d_view<real_type,device_type>;

   using ordinal_type_1d_view = Tines::value_type_1d_view<int,device_type>;

   using real_type_1d_view_host = typename real_type_1d_view::HostMirror;
   using real_type_2d_view_host = typename real_type_2d_view::HostMirror;
   using real_type_3d_view_host = typename real_type_3d_view::HostMirror;

   using ordinal_type_1d_view_host = typename ordinal_type_1d_view::HostMirror;

   template<typename ViewType>
    struct DualViewType {
      ViewType _dev;
      typename ViewType::HostMirror _host;
    };

    using real_type_1d_dual_view = DualViewType<real_type_1d_view>;
    using real_type_2d_dual_view = DualViewType<real_type_2d_view>;
    using real_type_3d_dual_view = DualViewType<real_type_3d_view>;

    using ordinal_type_1d_dual_view = DualViewType<ordinal_type_1d_view>;

    real_type_3d_view _Jacobian; //inputs sort-right eigenvectors or right CSP basis vectors
    real_type_2d_view _rhs, _state_vector; // input rhs, state vector

    int _nElem; // number of elements
    real_type _csp_rel_tol; // input relative tolerance
    real_type _csp_abs_tol; // input absolute tolerance


    real_type_3d_dual_view _B, _A; // left CSP basis vectors, rigth CSP basis vectors
    real_type_3d_dual_view _CSP_pointers;
    real_type_2d_dual_view _modal_amplitude, _time_scales;
    real_type_2d_dual_view _eigenvalues_real_part, _eigenvalues_imag_part;

    ordinal_type_1d_dual_view _M; // number of exhausted modes

    int _M_need_sync, _B_need_sync, _CSP_pointers_need_sync, _time_scales_need_sync;
    int _modal_amplitude_need_sync, _A_need_sync;
    int _eigenvalues_real_part_need_sync,  _eigenvalues_imag_part_need_sync;
    int _nBatch, _n_variables;

#if defined(CSPLIB_MESUARE_WALL_TIME)
    Kokkos::Impl::Timer timer;
    FILE* fs;
#endif

    enum {
       NeedSyncToDevice = 1,
       NeedSyncToHost = -1,
       NoNeedSync = 0
     };

  public:
   CSPKernelBatch( const real_type_3d_view& Jacobian,
                  const real_type_2d_view& rhs,
                  const real_type_2d_view& state_vector,
                  const int& nElem,
                  const real_type& csp_rel_tol,
                  const real_type& csp_abs_tol) :
                  _Jacobian(Jacobian),
                  _rhs(rhs),
                  _state_vector(state_vector),
                  _nElem(nElem),
                  _csp_rel_tol(csp_rel_tol),
                  _csp_abs_tol(csp_abs_tol),
                  _nBatch(Jacobian.extent(0)),
                  _n_variables(Jacobian.extent(1))
          {
            printf("kernel: _nBatch %d, _n_variables %d\n", _nBatch, _n_variables);
#if defined(CSPLIB_MESUARE_WALL_TIME)
            fs = fopen("CSPLIB_KernelKokkos.out", "a+");
            fprintf(fs, "%s, %d \n","Number of state vectors  ", _nBatch);
#endif
          }

   ~CSPKernelBatch();

   void evalCSPbasisVectors(const ordinal_type& team_size=ordinal_type(-1),
                            const ordinal_type& vector_size=ordinal_type(-1));
   void createB_View();
   void createCSP_PointersView();
   void evalCSP_Pointers(const ordinal_type& team_size=ordinal_type(-1),
                         const ordinal_type& vector_size=ordinal_type(-1));
   void createTimeScalesView();
   void evalTimeScales(const ordinal_type& team_size=ordinal_type(-1),
                       const ordinal_type& vector_size=ordinal_type(-1));
   void createModalAmpView();
   void evalModalAmp(const ordinal_type& team_size=ordinal_type(-1),
                     const ordinal_type& vector_size=ordinal_type(-1));
   void createM_View();
   void evalM(const ordinal_type& team_size=ordinal_type(-1),
              const ordinal_type& vector_size=ordinal_type(-1));
   real_type_3d_view getLeftCSPVecDevice();
   real_type_3d_view getCSPPointersDevice();
   real_type_3d_view_host getCSPPointers();
   real_type_2d_view_host getTimeScales();
   real_type_2d_view_host getModalAmp();
   ordinal_type_1d_view_host getM();
   ordinal_type_1d_view getMDevice();
   void evalEigenSolution(const ordinal_type& team_size=ordinal_type(-1),
                          const ordinal_type& vector_size=ordinal_type(-1));
   void sortEigenSolution(const ordinal_type& team_size=ordinal_type(-1),
                          const ordinal_type& vector_size=ordinal_type(-1));
   void createEigenvaluesImagPartView();
   void createEigenvaluesRealPartView();
   void createA_View();
   real_type_3d_view getRightCSPVecDevice();
   real_type_2d_view_host getEigenValuesRealPart();
   real_type_2d_view_host getEigenValuesImagPart();
   void freeA_View();
   void freeEigenvaluesRealPartView();
   void freeEigenvaluesImagPartView();
   void freeB_View();
   void freeCSP_PointersView();
   void freeTimeScalesView();
   void freeModalAmpView();
   void freeM_View();





  }; // end of CSPKernel
  #endif  //end of header guard
