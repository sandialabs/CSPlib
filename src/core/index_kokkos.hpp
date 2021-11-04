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


#ifndef INDEX_KOKKOS_CSP
#define INDEX_KOKKOS_CSP

#include "Tines.hpp"
#include "eigendecomposition_kokkos.hpp"
namespace CSP {

template <typename DeviceType>
struct IndexComputation {
  using device_type = DeviceType;
  using real_type = typename Tines::ats<double>::scalar_type;
  using ordinal_type = int;

  using real_type_0d_view_type = Tines::value_type_0d_view<real_type, device_type>;
  using real_type_1d_view_type = Tines::value_type_1d_view<real_type, device_type>;
  using real_type_2d_view_type = Tines::value_type_2d_view<real_type, device_type>;
  using real_type_3d_view_type = Tines::value_type_3d_view<real_type, device_type>;
  using ordinal_type_1d_view_type = Tines::value_type_1d_view<ordinal_type, device_type>;

  template <typename MemberType>
  KOKKOS_INLINE_FUNCTION static void
  evalBeta(const MemberType &member,
              const real_type_2d_view_type& B,
              const real_type_2d_view_type& S,
              const real_type_2d_view_type& Beta)
  {
    //Beta = B * S
    const real_type one(1), zero(0);
    Tines::Gemm<Tines::Trans::NoTranspose, Tines::Trans::NoTranspose>::invoke(member, one, B, S,
                                                           zero, Beta);
    member.team_barrier();
  }

  template <typename MemberType>
  KOKKOS_INLINE_FUNCTION static void
  evalCSPIndex(const MemberType &member,
               const real_type_2d_view_type& Beta, // input
               const real_type_1d_view_type& RoP, // input
               const real_type_2d_view_type& CPSIndex, //output
               const real_type_1d_view_type& deno)
  {
    /*
    participation index for all modes
                          (b^m . S_k) . r^k
    P(m,k)           = -----------------------------         m=1,N
                       sum_kk | (b^m . S_kk) . r^kk |
    */

    const ordinal_type n_variables(Beta.extent(0));
    const ordinal_type n_processes(Beta.extent(1));

    Kokkos::parallel_for(
      Kokkos::TeamThreadRange(member, n_variables), [&](const int &i) {
        real_type suma(0);
        Kokkos::parallel_reduce(
          Kokkos::ThreadVectorRange(member, n_processes), [&](const int &j, real_type& update) {
            update += Tines::ats<real_type>::abs( Beta(i,j) * RoP(j) );
      }, suma );
      deno(i) = suma ;
    });

    member.team_barrier();

    Kokkos::parallel_for(
      Kokkos::TeamThreadRange(member, n_variables), [&](const int &i) {
        if (deno(i) != 0) {
          Kokkos::parallel_for(
            Kokkos::ThreadVectorRange(member, n_processes), [&](const int &j) {
              CPSIndex(i,j) = Beta(i,j) * RoP(j) / deno(i) ;
          });
        } else {
            Kokkos::parallel_for(
              Kokkos::ThreadVectorRange(member, n_processes), [&](const int &j) {
                CPSIndex(i,j) = 0;
            });
        }
      });
      member.team_barrier();


  }

  template <typename MemberType>
  KOKKOS_INLINE_FUNCTION static void
  evalAlpha(const MemberType &member,
                         const real_type_2d_view_type& Beta, // input
                         const real_type_2d_view_type& A, // input
                         const ordinal_type& M,
                         const real_type_2d_view_type& Alpha //output
                         )
  {
    // KK: internal compiler error on cuda; change const to non-const variable
    ordinal_type n_processes(Beta.extent(1));
    ordinal_type n_variables(Beta.extent(0));

    // alpha = A* B
    Kokkos::parallel_for(
      Kokkos::TeamThreadRange(member, n_processes), [=](const int &k) {
      Kokkos::parallel_for(
        Kokkos::ThreadVectorRange(member, n_variables), [=](const int &j) {
          Alpha(j,k) = 0;
          for (int i=M; i<n_variables; i++) {
            Alpha(j,k) += A(j,i) * Beta(i,k);
          }
      });
    });
    member.team_barrier();

  }

  template <typename MemberType>
  KOKKOS_INLINE_FUNCTION static void
  evalGamma(const MemberType &member,
                         const real_type_2d_view_type& Beta, // input
                         const real_type_2d_view_type& A, // input
                         const ordinal_type& M,
                         const real_type_2d_view_type& Gamma //output
                         )
  {
    const ordinal_type n_processes(Beta.extent(1));
    const ordinal_type n_variables(Beta.extent(0));

    // alpha = A* B
    Kokkos::parallel_for(
      Kokkos::TeamThreadRange(member, n_processes), [&](const int &k) {
      Kokkos::parallel_for(
        Kokkos::ThreadVectorRange(member, n_variables), [&](const int &j) {
          Gamma(j,k) = 0;
          for (int i=0; i<M; i++) {
            Gamma(j,k) += A(j,i) * Beta(i,k);
          }
      });
    });
    member.team_barrier();

  }

  template<typename PolicyType>
  static void
  evalBetaBatch(const std::string& profile_name,
                const PolicyType& policy,
                const real_type_3d_view_type& B, // input
                const real_type_3d_view_type& S, // input
                const real_type_3d_view_type& Beta) // output
  {
    Tines::ProfilingRegionScope region(profile_name);
    using policy_type = PolicyType;

    const real_type one(1), zero(0);
    using exec_space = typename policy_type::execution_space;
    Tines::GemmDevice<Tines::Trans::NoTranspose,Tines::Trans::NoTranspose,exec_space>
        ::invoke(exec_space(), one, B, S, zero, Beta);

    // Kokkos::parallel_for(
    //  profile_name,
    //  policy,
    //  KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
    //    const ordinal_type i = member.league_rank();
    //    const real_type_2d_view_type B_at_i = Kokkos::subview(B, i, Kokkos::ALL(), Kokkos::ALL());
    //    const real_type_2d_view_type Beta_at_i = Kokkos::subview(Beta, i, Kokkos::ALL(), Kokkos::ALL());
    //    const real_type_2d_view_type S_at_i = Kokkos::subview(S, i, Kokkos::ALL(), Kokkos::ALL());
    //    evalBeta(member, B_at_i, S_at_i, Beta_at_i);
    //  });

  }
  //
  template<typename PolicyType>
  static void
  evalAlphaBatch(const std::string& profile_name,
                const PolicyType& policy,
                const real_type_3d_view_type& Beta, // input
                const real_type_3d_view_type& A, // input
                const ordinal_type_1d_view_type& M, // input
                const real_type_3d_view_type& Alpha) // output
  {
    Tines::ProfilingRegionScope region(profile_name);
    using policy_type = PolicyType;

    Kokkos::parallel_for(
     profile_name,
     policy,
     KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
       const ordinal_type i = member.league_rank();
       const real_type_2d_view_type A_at_i = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
       const real_type_2d_view_type Beta_at_i = Kokkos::subview(Beta, i, Kokkos::ALL(), Kokkos::ALL());
       const ordinal_type M_at_i = M(i);
       const real_type_2d_view_type Alpha_at_i = Kokkos::subview(Alpha, i, Kokkos::ALL(), Kokkos::ALL());

       evalAlpha(member, Beta_at_i, A_at_i, M_at_i, Alpha_at_i);
     });
  }
  //
  template<typename PolicyType>
  static void
  evalGammaBatch(const std::string& profile_name,
                const PolicyType& policy,
                const real_type_3d_view_type& Beta, // input
                const real_type_3d_view_type& A, // input
                const ordinal_type_1d_view_type& M, // input
                const real_type_3d_view_type& Gamma) // output
  {
    Tines::ProfilingRegionScope region(profile_name);
    using policy_type = PolicyType;

    Kokkos::parallel_for(
     profile_name,
     policy,
     KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
       const ordinal_type i = member.league_rank();
       const real_type_2d_view_type A_at_i =     Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
       const real_type_2d_view_type Beta_at_i = Kokkos::subview(Beta, i, Kokkos::ALL(), Kokkos::ALL());
       const ordinal_type M_at_i = M(i);
       const real_type_2d_view_type Gamma_at_i = Kokkos::subview(Gamma, i, Kokkos::ALL(), Kokkos::ALL());

       evalGamma(member, Beta_at_i, A_at_i, M_at_i, Gamma_at_i);
     });

  }
  //
  template<typename PolicyType>
  static void
  evalCSPIndexBatch(const std::string& profile_name,
                const PolicyType& policy,
                const real_type_3d_view_type& Beta, // input
                const real_type_2d_view_type& RoP, // input
                const real_type_3d_view_type& CSPIndex, //output
                const real_type_2d_view_type& deno) // work
  {
    Tines::ProfilingRegionScope region(profile_name);
    using policy_type = PolicyType;

    Kokkos::parallel_for(
     profile_name,
     policy,
     KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
       const ordinal_type i = member.league_rank();
       const real_type_1d_view_type RoP_at_i = Kokkos::subview(RoP, i, Kokkos::ALL());
       const real_type_1d_view_type deno_at_i = Kokkos::subview(deno, i, Kokkos::ALL());
       const real_type_2d_view_type Beta_at_i = Kokkos::subview(Beta, i, Kokkos::ALL(), Kokkos::ALL());
       const real_type_2d_view_type CSPIndex_at_i = Kokkos::subview(CSPIndex, i, Kokkos::ALL(), Kokkos::ALL());

       evalCSPIndex(member, Beta_at_i, RoP_at_i, CSPIndex_at_i, deno_at_i);
     });
  }

};

}// namespace csplib

#endif  //end of header guard
