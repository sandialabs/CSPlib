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


#ifndef __TEST_CHECK_HPP__
#define __TEST_CHECK_HPP__

#include "Kokkos_Core.hpp"
#include "Kokkos_Timer.hpp"

/// SpT, HpT, value_type are defined in the main-csp.cpp
namespace TestCSP {

  struct TestCheck {
    using A_value_type_3d_view = Kokkos::View<value_type***, Kokkos::LayoutRight,HpT>;
    using E_value_type_3d_view = Kokkos::View<value_type***, Kokkos::LayoutRight,HpT>;
    using V_value_type_4d_view = Kokkos::View<value_type****,Kokkos::LayoutRight,HpT>;

    using E_complex_value_type_2d_view = Kokkos::View<std::complex<value_type>**  ,Kokkos::LayoutRight,HpT>;
    using V_complex_value_type_4d_view = Kokkos::View<std::complex<value_type>****,Kokkos::LayoutRight,HpT>;
    using A_complex_value_type_3d_view = Kokkos::View<std::complex<value_type>*** ,Kokkos::LayoutRight,HpT>;

    int _N, _Blk;
    bool _vl_stores_col_vectors;

    A_value_type_3d_view _A_problem;    
    E_value_type_3d_view _E;
    V_value_type_4d_view _V;

    E_complex_value_type_2d_view _Ec;
    V_complex_value_type_4d_view _Vc;
    A_complex_value_type_3d_view _Ac;

    struct ConvertToComplexTag {};
    struct CheckLeftEigenvectorTag {};
    struct CheckRightEigenvectorTag {};

    inline 
    void operator()(const ConvertToComplexTag &, const int &i) const {
      const value_type zero(0);

      // for convenience, create a complex eigenvalues and eigenvectors
      auto er = Kokkos::subview(_E, i, 0, Kokkos::ALL());
      auto ei = Kokkos::subview(_E, i, 1, Kokkos::ALL());
      auto VL = Kokkos::subview(_V, i, 0, Kokkos::ALL(), Kokkos::ALL());
      auto VR = Kokkos::subview(_V, i, 1, Kokkos::ALL(), Kokkos::ALL());
      
      for (int l=0;l<_Blk;) {
        auto e  = Kokkos::subview(_Ec, i, l);
        auto vl = Kokkos::subview(_Vc, i, 0, Kokkos::ALL(), l);
        auto vr = Kokkos::subview(_Vc, i, 1, Kokkos::ALL(), l);
        
        if (ei(l) == zero) {
          // real eigenvalue
          e() = std::complex<value_type>(er(l), ei(l));
          for (int k=0;k<_Blk;++k) {
            vl(k) = _vl_stores_col_vectors ? VL(k,l) : VL(l,k);
            vr(k) = VR(k,l);
          }
          l += 1;
        } else {
          // complex eigenvalues
          auto ep0 = e;
          auto ep1 = Kokkos::subview(_Ec, i, l+1);
          
          ep0() = std::complex<value_type>(er(l  ), ei(l  ));
          ep1() = std::complex<value_type>(er(l+1), ei(l+1));
          
          auto vl0 = vl;
          auto vr0 = vr;
          auto vl1 = Kokkos::subview(_Vc, i, 0, Kokkos::ALL(), l+1);
          auto vr1 = Kokkos::subview(_Vc, i, 1, Kokkos::ALL(), l+1); 
          
          for (int k=0;k<_Blk;++k) {
            const value_type vl_kl  = _vl_stores_col_vectors ? VL(k,l  ) :  VL(l  ,k);
            const value_type vl_klp = _vl_stores_col_vectors ? VL(k,l+1) : -VL(l+1,k);
            vl0(k) = std::complex<value_type>(vl_kl,  vl_klp);
            vl1(k) = std::complex<value_type>(vl_kl, -vl_klp);
            vr0(k) = std::complex<value_type>(VR(k,l),  VR(k,l+1));
            vr1(k) = std::complex<value_type>(VR(k,l), -VR(k,l+1));
          }
          l += 2;
        }
      }
    }

    inline 
    void operator()(const CheckLeftEigenvectorTag &, const int &i) const {
      auto Ac = Kokkos::subview(_Ac,        i,    Kokkos::ALL(), Kokkos::ALL());
      auto Ap = Kokkos::subview(_A_problem, i,    Kokkos::ALL(), Kokkos::ALL());
      
      auto e  = Kokkos::subview(_Ec       , i,    Kokkos::ALL());
      auto VL = Kokkos::subview(_Vc       , i, 0, Kokkos::ALL(), Kokkos::ALL());
      
      // set Ac = VL'*A
      for (int k0=0;k0<_Blk;++k0)
        for (int k1=0;k1<_Blk;++k1) {
          std::complex<value_type> tmp(0);            
          for (int p=0;p<_Blk;++p) 
            tmp += std::conj(VL(p,k0))*Ap(p,k1);
          Ac(k0,k1) = tmp;
        }
      
      // check Ac - E VL' = 0
      for (int k0=0;k0<_Blk;++k0)
        for (int k1=0;k1<_Blk;++k1) 
          Ac(k0,k1) -= e(k0)*std::conj(VL(k1,k0));

#if 0
      printf("e \n");
      for (int k0=0;k0<_Blk;++k0) {
        const auto val = e(k0);
        printf(" %e+%ei \n", std::real(val), std::imag(val));
      }

      printf("VL \n");
      for (int k0=0;k0<_Blk;++k0) {
        for (int k1=0;k1<_Blk;++k1) {
          const auto val = VL(k0,k1);
          printf(" %e+%ei ", std::real(val), std::imag(val));
        }
        printf("\n");
      }
      
      printf("Ac \n");
      for (int k0=0;k0<_Blk;++k0) {
        for (int k1=0;k1<_Blk;++k1) {
          const auto val = Ac(k0,k1);
          printf(" %e+%ei ", std::real(val), std::imag(val));
        }
        printf("\n");
      }
#endif
    }

    inline 
    void operator()(const CheckRightEigenvectorTag &, const int &i) const {
      auto Ac = Kokkos::subview(_Ac,        i,    Kokkos::ALL(), Kokkos::ALL());
      auto Ap = Kokkos::subview(_A_problem, i,    Kokkos::ALL(), Kokkos::ALL());
      
      auto e  = Kokkos::subview(_Ec       , i,    Kokkos::ALL());
      auto VR = Kokkos::subview(_Vc       , i, 1, Kokkos::ALL(), Kokkos::ALL());
      
      // set Ac = A*VR
      for (int k0=0;k0<_Blk;++k0)
        for (int k1=0;k1<_Blk;++k1) {
          std::complex<value_type> tmp(0);            
          for (int p=0;p<_Blk;++p) 
            tmp += Ap(k0,p)*VR(p,k1);
          Ac(k0,k1) = tmp;
        }
      
      // check Ac - VR E   = 0
      for (int k0=0;k0<_Blk;++k0)
        for (int k1=0;k1<_Blk;++k1) 
          Ac(k0,k1) -= VR(k0,k1)*e(k1);
    }

    template<typename MViewType>
    double computeNormSquared(const MViewType &M) {
      double norm = 0;
      for (int k=0;k<_N;++k) 
        for (int i=0;i<_Blk;++i)
          for (int j=0;j<_Blk;++j) {
            const auto val = std::abs(M(k,i,j));
            norm += val*val;
          }
      return norm;
    }

    std::pair<bool,bool> checkTest(double tol = 1e-6) {
      // reconstruct matrix and compute diff
      Kokkos::parallel_for(Kokkos::RangePolicy<HpT,ConvertToComplexTag>(0, _N), *this);
      Kokkos::fence();
      
      const double q = _N;

      const double norm_ref = computeNormSquared(_A_problem);

      Kokkos::parallel_for(Kokkos::RangePolicy<HpT,CheckLeftEigenvectorTag>(0, _N), *this);
      Kokkos::fence();

      const double norm_left = computeNormSquared(_Ac);
      
      Kokkos::parallel_for(Kokkos::RangePolicy<HpT,CheckRightEigenvectorTag>(0, _N), *this);
      Kokkos::fence();

      const double norm_right = computeNormSquared(_Ac);
      
      const bool left_pass  = std::sqrt(norm_left /norm_ref/q) < tol;
      const bool right_pass = std::sqrt(norm_right/norm_ref/q) < tol;

      printf(" --- VL^H*A - E*VL^H: ref norm %e, diff %e\n", norm_ref, norm_left);
      printf(" --- A*VR - VR*E    : ref norm %e, diff %e\n", norm_ref, norm_right);

      return std::pair<bool,bool>(left_pass, right_pass);
    }

    template<typename AViewType,
             typename EViewType,
             typename VViewType>
    TestCheck(const int N, const int Blk,
              const AViewType &A_problem,
              const EViewType &E,
              const VViewType &V,
              const bool vl_stores_col_vectors)
      : _N(N),
        _Blk(Blk),
        _A_problem("A_problem_check", N, Blk, Blk),
        _E("E_check", N, 2, Blk),
        _V("V_check", N, 2, Blk, Blk),
        _Ec("Ec_mkl", N, Blk),
        _Vc("Vc_mkl", N, 2, Blk, Blk),
        _Ac("Ac_mkl", N, Blk, Blk),
        _vl_stores_col_vectors(vl_stores_col_vectors) {
      auto A_tmp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A_problem);
      auto E_tmp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), E);
      auto V_tmp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), V);

      Kokkos::deep_copy(_A_problem, A_tmp);
      Kokkos::deep_copy(_E, E_tmp);
      Kokkos::deep_copy(_V, V_tmp);
    }

  };
}

#endif
