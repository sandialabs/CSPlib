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


#ifndef KERNEL_KOKKOS_CSP
#define KERNEL_KOKKOS_CSP

#include "Tines.hpp"
#include "eigendecomposition_kokkos.hpp"

namespace CSP {

//
template <typename DeviceType>
struct KernelComputation {
  using device_type = DeviceType;
  using real_type = typename Tines::ats<double>::scalar_type;
  using ordinal_type = int;

  using real_type_0d_view_type = Tines::value_type_0d_view<real_type, device_type>;
  using real_type_1d_view_type = Tines::value_type_1d_view<real_type, device_type>;
  using real_type_2d_view_type = Tines::value_type_2d_view<real_type, device_type>;
  using real_type_3d_view_type = Tines::value_type_3d_view<real_type, device_type>;
  using ordinal_type_1d_view_type = Tines::value_type_1d_view<ordinal_type, device_type>;
  using ordinal_type_0d_view_type = Tines::value_type_0d_view<ordinal_type, device_type>;


  template <typename MemberType>
  KOKKOS_INLINE_FUNCTION static void
  evalLeftCSP_BasisVectors(const MemberType &member,
                          const real_type_2d_view_type& A, // input : right CSP basis vectors
                          const real_type_2d_view_type& B,// output : letf CSP basis vectors
                          const real_type_1d_view_type& work)// work space
  {
    auto wprt = work.data();
    const ordinal_type n_variables(A.extent(0));

    const real_type_1d_view_type w = real_type_1d_view_type(wprt, 2*n_variables);
    wprt += 2*n_variables;
    const real_type_2d_view_type Ac = real_type_2d_view_type(wprt, n_variables, n_variables );
    wprt += n_variables * n_variables;
    /// we do not want to touch the given matrix
    member.team_barrier();

    Tines::Copy::invoke(member, A, Ac);
    member.team_barrier();
    // B = inv(A)
    Tines::InvertMatrix::invoke(member, Ac, B, w);
    member.team_barrier();
  }

  template <typename MemberType>
  KOKKOS_INLINE_FUNCTION static void
  evalCSPPointers(const MemberType &member,
                  const real_type_2d_view_type& A, // input : right CSP basis vectors
                  const real_type_2d_view_type& B,// input : letf CSP basis vectors
                  const real_type_2d_view_type& CSP_pointers)
  {

    Kokkos::parallel_for(
      Kokkos::TeamThreadRange(member, B.extent(0)), [&](const int &i) {
      Kokkos::parallel_for(
        Kokkos::ThreadVectorRange(member, B.extent(1)), [&](const int &j) {
          CSP_pointers(i,j) = A(j,i)*B(i,j);
      });
    });
    member.team_barrier();
  }

  template <typename MemberType>
  KOKKOS_INLINE_FUNCTION static void
  evalTimeScales(const MemberType &member,
                 const real_type_1d_view_type& eigenvalues_real_part, // input
                 const real_type_1d_view_type& eigenvalues_imag_part, //input
                 const real_type_1d_view_type& time_scales) // output
{
  Kokkos::parallel_for(
    Kokkos::TeamThreadRange(member, eigenvalues_imag_part.extent(0)), [&](const int &i) {
      time_scales(i) = 1.0;
      const real_type evamp = Tines::ats<real_type>::sqrt( eigenvalues_real_part(i) * eigenvalues_real_part(i)
                   + eigenvalues_imag_part(i) * eigenvalues_imag_part(i) );
      //
      if (evamp != 0.0 ){
         time_scales(i) =  real_type(1.0) / evamp;
      }

  });
  member.team_barrier();

}

template <typename MemberType>
KOKKOS_INLINE_FUNCTION static void
evalModalAmp(const MemberType &member,
             const real_type_2d_view_type& B,// input : letf CSP basis vectors
             const real_type_1d_view_type& rhs,// input
             const real_type_1d_view_type& modal_amplitude)
{
  const real_type one(1), zero(0);
  //modal_amplitude = B * rhs
  Tines::Gemv<Tines::Trans::NoTranspose>::invoke(member, one, B, rhs, zero, modal_amplitude);
  member.team_barrier();
}


template <typename MemberType>
KOKKOS_INLINE_FUNCTION static void
evalM(const MemberType &member,
      const real_type_2d_view_type& A, // input : right CSP basis vectors
      const real_type_1d_view_type& eigenvalues_real_part, // input
      const real_type_1d_view_type& modal_amplitude, // input
      const real_type_1d_view_type& time_scales,
      const real_type_1d_view_type& state_vector,
      const ordinal_type& nElem, //input
      const real_type& rel_tol, // input relative tolerance
      const real_type& abs_tol, // input absolute tolerance
      const real_type_1d_view_type& error_csp, // work
      const real_type_1d_view_type& delta_yfast,
      ordinal_type& NofDM) // output
{
  // Maximun number of exhausted modes:
  // nElem number of elements constrains, conservation of mass
  // 1  conservation of enthalpy for adiabatic systems
  const ordinal_type nvars(state_vector.extent(0));
  const real_type one(1.);
  ordinal_type MaxExM = nElem > 0 ? nvars - nElem -1 : nvars ;

  NofDM=-1;

  // negative eigenvalues: exhausted modes cannot be bigger than number of eigen values
  ordinal_type no_of_neg_eig_val_real(0);
  Kokkos::parallel_reduce(
    Kokkos::TeamVectorRange(member, nvars), [&](const int &i, int& update) {
      if (eigenvalues_real_part(i) < 0.0)
        update++;
      // scalar error
      error_csp(i) = Tines::ats<real_type>::abs(state_vector(i)) * rel_tol + abs_tol;
  }, no_of_neg_eig_val_real);

  member.team_barrier();
  MaxExM =  MaxExM < no_of_neg_eig_val_real ?  MaxExM : no_of_neg_eig_val_real;

  ordinal_type exM(0);

  for ( int M = 0; M < nvars; M++) { // searching M in each variable
    // M is element position
    //check if next eigenvalue is a complex conjugate
    // we do not want to have an exhausted model between two equal eigen values
    // with an equal real part
    exM = M;
    if (M < nvars-1) {
      if  (eigenvalues_real_part(M)==eigenvalues_real_part(M+1) ) {
        M += 1;
          // if I am in the last model, I must quit because there is no \tau[_nvars+1]
        if (M == nvars){
            NofDM = nvars < MaxExM ? nvars : MaxExM;
            // M=nvars+1; // exit loop
            return;
        }
      }
    }//if M< nvars-1

    const ordinal_type Mp1 = M+1 < nvars-1 ? M+1 : nvars-1;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(member, nvars), [&](const ordinal_type &i) {
      real_type d_yfast(0);
      Kokkos::parallel_reduce(
        Kokkos::ThreadVectorRange(member, Mp1),
        [&](const ordinal_type &r, real_type& update) {
          update += modal_amplitude(r) * A(i,r) *
                ( -one + Tines::ats<real_type>::exp(eigenvalues_real_part(r) *
                 time_scales(Mp1) ) ) / eigenvalues_real_part(r);
      }, d_yfast);
      delta_yfast(i) = d_yfast;

    });

    member.team_barrier();
    ordinal_type count_yfast(0);
    Kokkos::parallel_reduce(
      Kokkos::TeamVectorRange(member, nvars),
      [&](const ordinal_type &i, ordinal_type& update) {
        if (Tines::ats<real_type>::abs(delta_yfast(i)) > error_csp(i))
          update ++;
    }, count_yfast);

    if (count_yfast > 0) {
      // at least one delta_yfast is bigger than error_csp
      //       //check elements
      NofDM = exM < MaxExM ? exM : MaxExM;
      return; // exit loop
    }


  } // end searching M in each variable
  
  // if searching loop does not find M
  if (NofDM == -1 )
    NofDM = nvars < MaxExM ? nvars : MaxExM ;


}

template<typename PolicyType>
static void
evalLeftCSP_BasisVectorsBatch(const std::string& profile_name,
              const PolicyType& policy,
              const real_type_3d_view_type& A, // input : right CSP basis vectors
              const real_type_3d_view_type& B,// output : letf CSP basis vectors
              const real_type_2d_view_type& work) // work
{
  Tines::ProfilingRegionScope region(profile_name);
  using policy_type = PolicyType;

  Kokkos::parallel_for(
   profile_name,
   policy,
   KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
     const ordinal_type i = member.league_rank();
     const real_type_2d_view_type A_at_i = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
     const real_type_2d_view_type B_at_i = Kokkos::subview(B, i, Kokkos::ALL(), Kokkos::ALL());
     const real_type_1d_view_type work_at_i = Kokkos::subview(work, i, Kokkos::ALL());
     evalLeftCSP_BasisVectors(member, A_at_i, B_at_i, work_at_i);
  });

}

template<typename PolicyType>
static void
evalCSPPointersBatch(const std::string& profile_name,
              const PolicyType& policy,
              const real_type_3d_view_type& A, // input : right CSP basis vectors
              const real_type_3d_view_type& B,// input : letf CSP basis vectors
              const real_type_3d_view_type& CSP_pointers) // output
{
  Tines::ProfilingRegionScope region(profile_name);
  using policy_type = PolicyType;

  Kokkos::parallel_for(
   profile_name,
   policy,
   KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
     const ordinal_type i = member.league_rank();
     const real_type_2d_view_type A_at_i = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
     const real_type_2d_view_type B_at_i = Kokkos::subview(B, i, Kokkos::ALL(), Kokkos::ALL());
     const real_type_2d_view_type CSP_pointers_at_i = Kokkos::subview(CSP_pointers, i, Kokkos::ALL(), Kokkos::ALL());
     evalCSPPointers(member, A_at_i, B_at_i, CSP_pointers_at_i);
  });

}

template<typename PolicyType>
static void
evalTimeScalesBatch(const std::string& profile_name,
              const PolicyType& policy,
              const real_type_2d_view_type& eigenvalues_real_part, // input
              const real_type_2d_view_type& eigenvalues_imag_part, //input
              const real_type_2d_view_type& time_scales) // output
{
  Tines::ProfilingRegionScope region(profile_name);
  using policy_type = PolicyType;

  Kokkos::parallel_for(
   profile_name,
   policy,
   KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
     const ordinal_type i = member.league_rank();
     const real_type_1d_view_type eigenvalues_real_part_at_i
     = Kokkos::subview(eigenvalues_real_part, i, Kokkos::ALL());

     const real_type_1d_view_type eigenvalues_imag_part_at_i
     = Kokkos::subview(eigenvalues_imag_part, i, Kokkos::ALL());

     const real_type_1d_view_type time_scales_at_i = Kokkos::subview(time_scales, i, Kokkos::ALL());
     evalTimeScales(member, eigenvalues_real_part_at_i, eigenvalues_imag_part_at_i, time_scales_at_i);
  });

}

template<typename PolicyType>
static void
evalModalAmpBatch(const std::string& profile_name,
              const PolicyType& policy,
              const real_type_3d_view_type& B,// input : letf CSP basis vectors
              const real_type_2d_view_type& rhs,// input
              const real_type_2d_view_type& modal_amplitude) // output
{
  Tines::ProfilingRegionScope region(profile_name);
  using policy_type = PolicyType;

  Kokkos::parallel_for(
   profile_name,
   policy,
   KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
     const ordinal_type i = member.league_rank();
     const real_type_2d_view_type B_at_i = Kokkos::subview(B, i, Kokkos::ALL(), Kokkos::ALL());
     const real_type_1d_view_type rhs_at_i = Kokkos::subview(rhs, i, Kokkos::ALL());
     const real_type_1d_view_type modal_amplitude_at_i = Kokkos::subview(modal_amplitude, i, Kokkos::ALL());

     evalModalAmp(member, B_at_i, rhs_at_i, modal_amplitude_at_i);
  });

}

template<typename PolicyType>
static void
evalMBatch(const std::string& profile_name,
              const PolicyType& policy,
              const real_type_3d_view_type& A, // input : right CSP basis vectors
              const real_type_2d_view_type& eigenvalues_real_part, // input
              const real_type_2d_view_type& modal_amplitude, // input
              const real_type_2d_view_type& time_scales,
              const real_type_2d_view_type& state_vector,
              const ordinal_type& nElem, //input
              const real_type& rel_tol, // input relative tolerance
              const real_type& abs_tol, // input absolute tolerance
              const real_type_2d_view_type& error_csp, // work
              const real_type_2d_view_type& delta_yfast, // work
              const ordinal_type_1d_view_type& NofDM) // output
{
  Tines::ProfilingRegionScope region(profile_name);
  using policy_type = PolicyType;

  Kokkos::parallel_for(
   profile_name,
   policy,
   KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
     const ordinal_type i = member.league_rank();
     const real_type_2d_view_type A_at_i = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
     const real_type_1d_view_type eigenvalues_real_part_at_i
     = Kokkos::subview(eigenvalues_real_part, i, Kokkos::ALL());
     const real_type_1d_view_type modal_amplitude_at_i = Kokkos::subview(modal_amplitude, i, Kokkos::ALL());
     const real_type_1d_view_type time_scales_at_i = Kokkos::subview(time_scales, i, Kokkos::ALL());
     const real_type_1d_view_type error_csp_at_i = Kokkos::subview(error_csp, i, Kokkos::ALL());
     const real_type_1d_view_type state_vector_at_i = Kokkos::subview(state_vector, i, Kokkos::ALL());
     const ordinal_type_0d_view_type NofDM_at_i = Kokkos::subview(NofDM, i);
     const real_type_1d_view_type delta_yfast_at_i = Kokkos::subview(delta_yfast, i, Kokkos::ALL());



     evalM(member, A_at_i, eigenvalues_real_part_at_i, modal_amplitude_at_i,
                          time_scales_at_i, state_vector_at_i, nElem, rel_tol, abs_tol,
                          error_csp_at_i, delta_yfast_at_i,  NofDM_at_i());
  });

}



};
}// namespace csplib

#endif  //end of header guard
