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


#ifndef __TEST_MKL_HPP__
#define __TEST_MKL_HPP__
#include "mkl.h"

#include "Kokkos_Core.hpp"
#include "Kokkos_Timer.hpp"

/// SpT, HpT, value_type are defined in the main-csp.cpp
namespace TestCSP {

  struct TestMKL {
    using A_value_type_3d_view = Kokkos::View<value_type***, Kokkos::LayoutRight,HpT>;
    using E_value_type_3d_view = Kokkos::View<value_type***, Kokkos::LayoutRight,HpT>;
    using V_value_type_4d_view = Kokkos::View<value_type****,Kokkos::LayoutRight,HpT>;
    using W_value_type_2d_view = Kokkos::View<value_type**,  Kokkos::LayoutRight,HpT>;

    using E_complex_value_type_2d_view = Kokkos::View<std::complex<value_type>**  ,Kokkos::LayoutRight,HpT>;
    using V_complex_value_type_4d_view = Kokkos::View<std::complex<value_type>****,Kokkos::LayoutRight,HpT>;
    using A_complex_value_type_3d_view = Kokkos::View<std::complex<value_type>*** ,Kokkos::LayoutRight,HpT>;

    int _N, _Blk;
    
    A_value_type_3d_view _A;
    E_value_type_3d_view _E;
    V_value_type_4d_view _V;
    W_value_type_2d_view _W;

    int getWorkSpaceSize() {
      int lwork_mkl = -1;
      {    
        double work_query;
        LAPACKE_dgeev_work(LAPACK_ROW_MAJOR, 
                           //'N', 'V',
                           'V', 'V', 
                           _Blk, 
                           NULL, _Blk,
                           NULL, NULL,
                           NULL, _Blk,
                           NULL, _Blk,
                           &work_query,
                           lwork_mkl);
        lwork_mkl = int(work_query);
      }
      return lwork_mkl;
    }

    template<typename ArgViewType>
    void setProblem(const ArgViewType &A) {
      const value_type zero(0);
      Kokkos::deep_copy(_A, A);
      Kokkos::deep_copy(_E, zero);
      Kokkos::deep_copy(_V, zero);
      Kokkos::deep_copy(_W, zero);
    }

    struct RunTestTag   {};

    inline
    void operator()(const RunTestTag &, const int &i) const {
      LAPACKE_dgeev_work(LAPACK_ROW_MAJOR, 
                         //'N', 'V', 
                         'V', 'V', 
                         _Blk, 
                         (value_type*)&_A(i,0,0), int(_A.stride(1)),
                         &_E(i,0,0), &_E(i,1,0),
                         &_V(i,0,0,0), int(_V.stride(2)),
                         &_V(i,1,0,0), int(_V.stride(2)),
                         &_W(i,0), int(_W.extent(1)));
    }

    double runTest() {
      Kokkos::Timer timer;      
      timer.reset();
      {
        Kokkos::RangePolicy<HpT,RunTestTag> policy(0, _N);      
        Kokkos::parallel_for(policy, *this);
        Kokkos::fence();
      }
      const double t = timer.seconds();
      return t;
    }

    TestMKL(const int N, const int Blk) 
      : _N(N),
        _Blk(Blk),
        _A("A_mkl", N, Blk, Blk),
        _E("E_mkl", N, 2, Blk),
        _V("V_mkl", N, 2, Blk, Blk),
        _W("W_mkl", N, getWorkSpaceSize()) {}

  };
}

#endif
