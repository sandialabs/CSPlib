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


#ifndef __TEST_MAGMA_HPP__
#define __TEST_MAGMA_HPP__

#include "cublas_v2.h"
#include "magma_v2.h"
#include "magma_lapack.h"

#include "Kokkos_Core.hpp"
#include "Kokkos_Timer.hpp"

/// SpT, HpT, value_type are defined in the main-csp.cpp
namespace TestCSP {

  struct TestMagma {
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

    A_value_type_3d_view _A_problem;

    int getWorkSpaceSize() {
      int lwork_magma = -1;
      {    
        int info;
        double work_query;
        magma_dgeev(MagmaVec, MagmaVec,
                    //MagmaNoVec, MagmaVec,
                    _Blk, 
                    NULL, _Blk,
                    NULL, NULL,
                    NULL, _Blk,
                    NULL, _Blk,
                    &work_query,
                    lwork_magma,
                    &info);
        lwork_magma = int(work_query);
      }
      return lwork_magma;
    }

    template<typename ArgViewType>
    void setProblem(const ArgViewType &A_problem) {
      // change the matrix into column major 
      for (int i=0;i<_N;++i) {
        auto B = Kokkos::subview(A_problem, i, Kokkos::ALL(), Kokkos::ALL());
        auto A = Kokkos::subview(_A, i, Kokkos::ALL(), Kokkos::ALL());
        
        // transpose copy
        for (int k0=0;k0<_Blk;++k0)
          for (int k1=0;k1<_Blk;++k1)
            A(k0,k1) = B(k1,k0);
      }
      
      const value_type zero(0);
      Kokkos::deep_copy(_E, zero);
      Kokkos::deep_copy(_V, zero);
      Kokkos::deep_copy(_W, zero);
    }

    inline
    void runMagma(const int &i) const {
      int info;
      magma_dgeev(MagmaVec, MagmaVec,
                  //MagmaNoVec, MagmaVec,
                  _Blk, 
                  &_A(i,0,0), _Blk,
                  &_E(i,0,0), &_E(i,1,0),
                  &_V(i,0,0,0), _Blk,
                  &_V(i,1,0,0), _Blk,
                  &_W(i,0), int(_W.extent(1)),
                  &info);
      Kokkos::fence();
      if (info) {
        printf("Error: magma_dgeev returns info %d\n", info);
      }
    }

    
    double runTest() {
      Kokkos::Timer timer;      
      timer.reset();
      {
        for (int i=0;i<_N;++i)
          runMagma(i);
      }
      const double t = timer.seconds();
      return t;
    }

    void postUpdate() {
      Kokkos::View<value_type**,HpT> T("Temp", _Blk, _Blk);
      for (int i=0;i<_N;++i) {
        // copy VL into T
        for (int k0=0;k0<_Blk;++k0)
          for (int k1=0;k1<_Blk;++k1)
            T(k0,k1) = _V(i,0,k0,k1);
        // transpose copy T into VL
        for (int k0=0;k0<_Blk;++k0)
          for (int k1=0;k1<_Blk;++k1)
            _V(i,0,k0,k1) = T(k1,k0);

        // copy VR into T
        for (int k0=0;k0<_Blk;++k0)
          for (int k1=0;k1<_Blk;++k1)
            T(k0,k1) = _V(i,1,k0,k1);
        // transpose copy T into VR
        for (int k0=0;k0<_Blk;++k0)
          for (int k1=0;k1<_Blk;++k1)
            _V(i,1,k0,k1) = T(k1,k0);
      }
    }

    TestMagma(const int N, const int Blk) 
      : _N(N),
        _Blk(Blk),
        _A("A_magma", N, Blk, Blk),
        _E("E_magma", N, 2, Blk),
        _V("V_magma", N, 2, Blk, Blk),
        _W("W_magma", N, getWorkSpaceSize()) {}

  };
}

#endif
