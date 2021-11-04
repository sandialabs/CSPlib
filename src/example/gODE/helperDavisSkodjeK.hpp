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


#ifndef HELPER_DAVISKODJEK
#define HELPER_DAVISKODJEK

#include "Tines.hpp"

template <typename SpT>
struct Davis_Skodje {
  using device_type = typename Tines::UseThisDevice<SpT>::type;
  using real_type_1d_view_type = Tines::value_type_1d_view<double, device_type>;
  using real_type_2d_view_type = Tines::value_type_2d_view<double, device_type>;
  using real_type_3d_view_type = Tines::value_type_3d_view<double, device_type>;

  static void
  rhs(const real_type_2d_view_type& state,
      const real_type_2d_view_type& source,
      const double epsilon )
  {

    const double nBatch(state.extent(0));

    Kokkos::parallel_for(Kokkos::RangePolicy<SpT>(0, nBatch ), KOKKOS_LAMBDA(const int &i) {
      const double one(1.0);
      source(i, 0) = (-state(i, 0) + state(i, 1)/(one + state(i, 1)))/epsilon
                   - state(i, 1)/(one + state(i, 1))/(one + state(i, 1));
      source(i, 1) = -state(i, 1);
    });

  }

  static void
  state_vector(const real_type_2d_view_type& state,
               const double tbegin,
               const double dt,
               const int nBatch,
               const double y0,
               const double z0,
               const double epsilon )
  {

    const double one(1.0);
    Kokkos::parallel_for(Kokkos::RangePolicy<SpT>(0, nBatch ), KOKKOS_LAMBDA(const int &i) {
      const double one(1.0);
      const double t = tbegin + dt*i;
      state(i,0) = (y0-z0/(1.+z0)) * Tines::ats<double>::exp(-t/epsilon) + z0*Tines::ats<double>::exp(-t)/(one+z0*Tines::ats<double>::exp(-t));
      state(i,1) = z0*Tines::ats<double>::exp(-t);
    });

  }




  static void
  jacobian(const real_type_2d_view_type& state,
           const real_type_3d_view_type& jac,
           const double epsilon )
  {

    const double nBatch(state.extent(0));
    const double one(1.0);
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::OpenMP>(0, nBatch ), KOKKOS_LAMBDA(const int &i) {
      const double one(1.0);
      jac(i, 0,0) = -one/epsilon;
      jac(i, 1,0) = double(0);
      jac(i, 0,1) = double(2.) * state(i, 1) / Tines::ats<double>::pow( state(i, 1) + one , 3.) - one / Tines::ats<double>::pow( state(i, 1) + one, 2) +
                    ( - state(i, 1) / Tines::ats<double>::pow( state(i, 1) + one , 2.) + one / ( state(i, 1) + one ) ) / epsilon;
      jac(i, 1,1) = -one;

    });

  }

};

#if defined(KOKKOS_ENABLE_OPENMP)
static
void rhs_Davis_Skodje(const Tines::value_type_2d_view<double,typename Tines::UseThisDevice<Kokkos::OpenMP>::type>& state,
                      const Tines::value_type_2d_view<double,typename Tines::UseThisDevice<Kokkos::OpenMP>::type>& source,
                      const double epsilon){
    //
  std::string profile_name = "CSPlib::OpenMP::Davis_Skodje::RHSs";
  Kokkos::Profiling::pushRegion(profile_name);
  Davis_Skodje<Kokkos::OpenMP>::rhs(state, source, epsilon );
  Kokkos::Profiling::popRegion();
  return;
}

static
void state_Davis_Skodje(const Tines::value_type_2d_view<double,typename Tines::UseThisDevice<Kokkos::OpenMP>::type>& state,
                        const double tbegin, const double dt, const int nBatch,
                        const double y0, const double z0,
                        const double epsilon){

   std::string profile_name = "CSPlib::OpenMP::Davis_Skodje::StateVectors";
   Kokkos::Profiling::pushRegion(profile_name);
   Davis_Skodje<Kokkos::OpenMP>
   ::state_vector(state, tbegin, dt, nBatch, y0, z0, epsilon );
   Kokkos::Profiling::popRegion();

  return;
}

static
void jac_Davis_Skodje(const Tines::value_type_2d_view<double,typename Tines::UseThisDevice<Kokkos::OpenMP>::type>& state,
                      const Tines::value_type_3d_view<double,typename Tines::UseThisDevice<Kokkos::OpenMP>::type>& jac,
                      const double epsilon){

  std::string profile_name = "CSPlib::OpenMP::Davis_Skodje::Jacobians";
  Kokkos::Profiling::pushRegion(profile_name);
  Davis_Skodje<Kokkos::OpenMP>::jacobian(state, jac, epsilon );
  Kokkos::Profiling::popRegion();
  return;
}
#endif

#if defined(KOKKOS_ENABLE_CUDA)
static
void rhs_Davis_Skodje(const Tines::value_type_2d_view<double,typename Tines::UseThisDevice<Kokkos::Cuda>::type>& state,
                      const Tines::value_type_2d_view<double,typename Tines::UseThisDevice<Kokkos::Cuda>::type>& source,
                      const double epsilon){
    //
  std::string profile_name = "CSPlib::CUDA::Davis_Skodje::RHSs";
  Kokkos::Profiling::pushRegion(profile_name);
  Davis_Skodje<Kokkos::Cuda>::rhs(state, source, epsilon );
  Kokkos::Profiling::popRegion();
  return;
}

static
void state_Davis_Skodje(const Tines::value_type_2d_view<double,typename Tines::UseThisDevice<Kokkos::Cuda>::type>& state,
                        const double tbegin, const double dt, const int nBatch,
                        const double y0, const double z0,
                        const double epsilon){

   std::string profile_name = "CSPlib::OpenMP::Davis_Skodje::StateVectors";
   Kokkos::Profiling::pushRegion(profile_name);
   Davis_Skodje<Kokkos::Cuda>
   ::state_vector(state, tbegin, dt, nBatch, y0, z0, epsilon );
   Kokkos::Profiling::popRegion();

  return;
}

static
void jac_Davis_Skodje(const Tines::value_type_2d_view<double,typename Tines::UseThisDevice<Kokkos::Cuda>::type>& state,
                      const Tines::value_type_3d_view<double,typename Tines::UseThisDevice<Kokkos::Cuda>::type>& jac,
                      const double epsilon){

  std::string profile_name = "CSPlib::OpenMP::Davis_Skodje::Jacobians";
  Kokkos::Profiling::pushRegion(profile_name);
  Davis_Skodje<Kokkos::Cuda>::jacobian(state, jac, epsilon );
  Kokkos::Profiling::popRegion();
  return;
}
#endif

#endif  //end of header guard
