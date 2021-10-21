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


#include "kernel_kokkos.hpp"
#include "Tines.hpp"
#include "kernel.hpp"


int main(int argc, char* argv[])
{
  {
    CSP::ScopeGuard guard(argc, argv);

    using real_type = double;
    using host_exec_space = Kokkos::DefaultHostExecutionSpace;
    using host_device_type = typename Tines::UseThisDevice<host_exec_space>::type;
    using real_type_2d_view_type = Tines::value_type_2d_view<real_type,host_device_type>;
    using real_type_1d_view_type = Tines::value_type_1d_view<real_type,host_device_type>;

    using ordinal_type_0d_view_type = Tines::value_type_0d_view<int,host_device_type>;

    using kernel_csplib = CSP::KernelComputation<host_device_type>;
    using ats = Tines::ats<real_type>;

    const real_type n_variables = 10;
    const int nElem = 1;

    const real_type csp_rtolvar(1e-8);
    const real_type csp_atolvar(1e-12);

    real_type_1d_view_type state("state", n_variables);
    real_type_1d_view_type source("source", n_variables);
    real_type_2d_view_type jac("jac", n_variables, n_variables);

    Kokkos::Random_XorShift64_Pool<host_device_type> random(13718);
    Kokkos::fill_random(state, random, real_type(1.0));
    Kokkos::fill_random(source, random, real_type(1.0));
    Kokkos::fill_random(jac, random, real_type(1.0));


    std::vector< double > state_std (std::vector< double > (state.extent(1),0));
    Tines::convertToStdVector(state_std, state);

    std::vector< double > source_std (std::vector< double > (source.extent(1),0));
    Tines::convertToStdVector(source_std, source);

    std::vector< std::vector<  double  > > jac_std
    (jac.extent(0), std::vector< double > (jac.extent(1),0));
    Tines::convertToStdVector(jac_std, jac);

    std::vector<double> eig_val_real;
    std::vector<double> eig_val_imag;
    std::vector<double> eig_vec_L;
    std::vector<double> eig_vec_R;
    std::vector<double> tau_vec;
    std::vector<double> f_vec;

    std::vector<double> csp_vec_R(n_variables*n_variables);
    std::vector<double> csp_vec_L(n_variables*n_variables);
    std::vector<std::vector<double> > csp_vec_R_2d;
    std::vector<std::vector<double> > csp_vec_L_2d;
    std::vector<std::vector<double> > cspp_ij;

    Kernel ker(n_variables, state_std, source_std, jac_std);

    // Eigen solution:
    // Sorting eigen values and vectors in ascending order
    // of, sign(eig_val_real)*Mod(eig_val_real + i * eig_val_imag)
    ker.evalEigenValVec();
    
    ker.getEigenValVec(eig_val_real, eig_val_imag, eig_vec_L, eig_vec_R);

    ker.setCSPVec(); // A = eig_vec_R and B = A^{-1}
    ker.getCSPVec(csp_vec_L, csp_vec_R);

    ker.evalCSPPointers();
    ker.getCSPPointers( cspp_ij );

    // Time scales:
    ker.evalTau();
    ker.getTau(tau_vec);

    ker.evalModalAmp( );
    ker.getModalAmp( f_vec );

    ker.setCSPerr(csp_rtolvar, csp_atolvar);

    // Exhausted mode
    int NofDM = 0;
    ker.evalM(nElem);
    ker.getM(NofDM);

    const auto member = Tines::HostSerialTeamMember();
    // A = inv(B)
    real_type_1d_view_type w("w", 2 * n_variables + n_variables*n_variables );
    real_type_2d_view_type B("B", n_variables, n_variables);
    real_type_2d_view_type CSP_pointers("CSP_pointers",n_variables, n_variables );
    real_type_2d_view_type A;
    real_type_1d_view_type eigenvalues_real_part;
    real_type_1d_view_type eigenvalues_imag_part;
    real_type_1d_view_type time_scales("time_scales", n_variables);
    real_type_1d_view_type modal_amplitude("modal_amplitude", n_variables);
    real_type_1d_view_type error_csp("error_csp", n_variables);
    ordinal_type NofDM_kokkos;

    CSP::construct_2D_from_1D(n_variables, n_variables, csp_vec_R, csp_vec_R_2d);
    CSP::construct_2D_from_1D(n_variables, n_variables, csp_vec_L, csp_vec_L_2d);

    Tines::convertToKokkos(A, csp_vec_R_2d);
    Tines::convertToKokkos(eigenvalues_real_part, eig_val_real);
    Tines::convertToKokkos(eigenvalues_imag_part, eig_val_imag);

    kernel_csplib::evalLeftCSP_BasisVectors(member, A, B, w);

    kernel_csplib::evalCSPPointers(member, A, B, CSP_pointers);
    kernel_csplib::evalTimeScales(member, eigenvalues_real_part,
                                          eigenvalues_imag_part, time_scales);

    kernel_csplib::evalModalAmp(member, B, source, modal_amplitude);

    kernel_csplib::evalM(member, A, eigenvalues_real_part, modal_amplitude,
                         time_scales, source, nElem, csp_rtolvar, csp_atolvar,
                         error_csp, NofDM_kokkos);

    //
    auto compareKernel = [](const std::string &label, std::vector<std::vector<double> >  &A, real_type_2d_view_type &B) {
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

    auto compareKernel1D = [](const std::string &label, std::vector<double >  &A, real_type_1d_view_type &B) {
      int m (B.extent(0));
      real_type err(0), norm(0);
      real_type max_diff(0);
      int max_j(0);
      for (int j = 0; j < m; ++j) {
        const real_type diff = ats::abs(A[j] - B(j));
        const real_type val = ats::abs(A[j]);
        if (max_diff < diff)
          {
            max_diff = diff;
            max_j = j;
          }
          norm += val * val;
          err += diff * diff;
        }
      const real_type rel_err = ats::sqrt(err / norm);

      const real_type max_rel_err = (A[max_j] - B(max_j))/A[max_j];

      // Tines::showMatrix(label, B);
      const real_type margin = 1e2, threshold = ats::epsilon() * margin;
      if (rel_err < threshold)
        std::cout << "PASS ";
      else
        std::cout << "FAIL ";
      std::cout << label << " relative error : " << rel_err
                << " within threshold : " << threshold <<"\n\n";

      std::cout  << " row idx : " << max_j
                 << " max absolute error : " << max_diff
                 << " std::vector version : "      << A[max_j] << " kokkos version : "    << B(max_j)
                 << " max relavite error :"  << max_rel_err <<"\n\n";
    };

    compareKernel(std::string("Left CSP basis vectors "), csp_vec_L_2d, B);
    compareKernel(std::string("Rigth CSP basis vectors "), csp_vec_R_2d, A);
    compareKernel(std::string("CSP pointers "), cspp_ij, CSP_pointers);
    compareKernel1D(std::string("times scales  "), tau_vec, time_scales);
    compareKernel1D(std::string("modal_amplitude  "), f_vec, modal_amplitude);
    printf(" M std::vector %d  kokkos %d \n", NofDM,   NofDM_kokkos);





  }

  return 0;

}
