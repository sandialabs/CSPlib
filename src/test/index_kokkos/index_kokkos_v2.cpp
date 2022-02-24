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



#include "index_kokkos.hpp"
#include "Tines.hpp"
#include "index.hpp"

int main(int argc, char* argv[])
{

  {
    CSP::ScopeGuard guard(argc, argv);
    using real_type = double;
    using host_exec_space = Kokkos::DefaultHostExecutionSpace;
    using host_device_type = typename Tines::UseThisDevice<host_exec_space>::type;
    using real_type_2d_view_type = Tines::value_type_2d_view<real_type,host_device_type>;
    using real_type_1d_view_type = Tines::value_type_1d_view<real_type,host_device_type>;

    using index_csplib = CSP::IndexComputation<host_device_type>;
    using ats = Tines::ats<real_type>;

    const real_type n_variables = 5;
    const real_type n_reactions = 10;
    const real_type n_processes = 2 * n_reactions;
    real_type_2d_view_type B("B", n_variables, n_variables);
    real_type_2d_view_type Q("Q", n_variables, n_reactions);
    real_type_1d_view_type RoP_fwd("RoP_fwd", n_reactions);
    real_type_1d_view_type RoP_rev("RoP_rev", n_reactions);

    Kokkos::Random_XorShift64_Pool<host_device_type> random(13718);
    Kokkos::fill_random(B, random, real_type(1.0));
    Kokkos::fill_random(Q, random, real_type(1.0));
    Kokkos::fill_random(RoP_fwd, random, real_type(1.0));
    Kokkos::fill_random(RoP_rev, random, real_type(1.0));

    const auto member = Tines::HostSerialTeamMember();
    // A = inv(B)
    real_type_1d_view_type w("w", 2 * n_variables);
    real_type_2d_view_type A("A", n_variables, n_variables);
    Tines::InvertMatrix::invoke(member, B, A, w);

    real_type_2d_view_type Beta("Beta", n_variables, n_reactions);
    index_csplib::evalBeta(member, B, Q, Beta);
    Tines::showMatrix("Beta", Beta);

    // slow importance index
    const real_type M(n_variables-2);
    real_type_2d_view_type Alpha("Alpha", n_variables, n_reactions);
    index_csplib::evalAlpha(member, Beta, A, M, Alpha);

    real_type_2d_view_type SlowImpIndex("SlowImpIndex", n_variables, n_processes);
    real_type_1d_view_type deno("deno", n_variables);
    index_csplib::evalCSPIndex(member, Alpha, RoP_fwd, RoP_rev, -1, SlowImpIndex, deno);
    Tines::showMatrix("SlowImpIndex", SlowImpIndex);

    // participation index
    real_type_2d_view_type ParIndex("ParIndex", n_variables, n_processes);
    index_csplib::evalCSPIndex(member, Beta, RoP_fwd, RoP_rev, -1,  ParIndex, deno);
    Tines::showMatrix("ParIndex", ParIndex);

    // fast important index
    real_type_2d_view_type Gamma("Gamma", n_variables, n_reactions);
    real_type_2d_view_type FastImpIndex("FastImpIndex", n_variables, n_processes);

    index_csplib::evalGamma(member, Beta, A, M, Gamma);
    index_csplib::evalCSPIndex(member, Gamma, RoP_fwd, RoP_rev, -1, FastImpIndex, deno);
    Tines::showMatrix("FastImpIndex", FastImpIndex);

    std::vector<double> eig_val_real, eig_val_imag; // we do not need eigen solution to compute the Participation, and Importance indecies

    std::vector< std::vector<  double  > > csp_vec_R_2d
    (A.extent(0), std::vector< double > (A.extent(1),0));

    Tines::convertToStdVector(csp_vec_R_2d, A);

    std::vector< std::vector<  double  > > csp_vec_L_2d
    (B.extent(0), std::vector<double  > (B.extent(1),0));

    Tines::convertToStdVector(csp_vec_L_2d, B);

    real_type_2d_view_type S("S", n_variables, n_processes);

    auto S_A = Kokkos::subview(S, Kokkos::ALL(), Kokkos::pair<int,int>(0,n_reactions));
    auto S_B = Kokkos::subview(S, Kokkos::ALL(), Kokkos::pair<int,int>(n_reactions,2*n_reactions));

    Kokkos::deep_copy(S_A, Q);
    Kokkos::deep_copy(S_B, Q);

    std::vector< std::vector<  double  > > Smat
    (S.extent(0), std::vector< double > (S.extent(1),0));

    Tines::convertToStdVector(Smat, S);

    std::vector< double > RoPv (2*n_reactions,0);

    for (int i = 0; i < n_reactions; i++) {
      const ordinal_type idx = i+n_reactions;
      RoPv[i] = RoP_fwd(i);
      RoPv[idx] = -RoP_rev(i);
    }

    CSPIndex idx(n_processes, n_variables,
                 M, eig_val_real, eig_val_imag,
                 csp_vec_R_2d, csp_vec_L_2d, Smat, RoPv );

    idx.evalParticipationIndex();
    idx.evalImportanceIndexSlow();
    idx.evalImportanceIndexFast();

    std::vector<std::vector<double> > P_ik    ;
    std::vector<std::vector<double> > Islow_jk;
    std::vector<std::vector<double> > Ifast_jk;

    idx.getParticipationIndex ( P_ik     );
    idx.getImportanceIndexSlow( Islow_jk );
    idx.getImportanceIndexFast( Ifast_jk );


    auto compareCSPIndex = [](const std::string &label, std::vector<std::vector<double> >  &A, real_type_2d_view_type &B) {
      int m (B.extent(0));
      int n (B.extent(1));
      real_type err(0), norm(0);
      real_type max_diff(0);
      int max_i(0);
      int max_j(0);
      for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
          const real_type diff = ats::abs(A[i][j] - B(i, j));
          const real_type val = ats::abs(A[i][j]);
          if (max_diff < diff)
          {
            max_diff = diff;
            max_i = i;
            max_j = j;
          }
          norm += val * val;
          err += diff * diff;
        }
      const real_type rel_err = ats::sqrt(err / norm);

      const real_type max_rel_err = (A[max_i][max_j]-B(max_i, max_j))/A[max_i][max_j];

      // Tines::showMatrix(label, B);
      const real_type margin = 1e2, threshold = ats::epsilon() * margin;
      if (rel_err < threshold)
        std::cout << "PASS ";
      else
        std::cout << "FAIL ";
      std::cout << label << " relative error : " << rel_err
                << " within threshold : " << threshold <<"\n\n";

      std::cout  << " row idx : " << max_i   << " column idx : " << max_j
                 << " max absolute error : " << max_diff
                 << " std::vector version : "      << A[max_i][max_j] << " kokkos version : "    << B(max_i, max_j)
                 << " max relavite error :"  << max_rel_err <<"\n\n";
    };

    compareCSPIndex(std::string("Participation Index "), P_ik, ParIndex);

    compareCSPIndex(std::string("Slow Importance Index "), Islow_jk, SlowImpIndex);

    compareCSPIndex(std::string("Fast Importance Index "), Ifast_jk, FastImpIndex);




  }

return 0;
}
