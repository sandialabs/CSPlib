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
#include "Tines.hpp"
#include "CSPlib_CommandLineParser.hpp"
#include "eigendecomposition_kokkos.hpp"
#include "helperDavisSkodjeK.hpp"
#include "kernelBatch.hpp"

int main(int argc, char *argv[]) {

  double csp_rtolvar(1.e-3); //1.e-7; // 1.e+3; //
  double csp_atolvar(1.e-14); //1.e-8; // 1.e+3; //
  double y0(2);
  double z0(1e-2);
  double tend(4);
  int nPoints(2000);
  double epsilon(0.01);

  CSP::CommandLineParser opts("This example Number of exhausted and time scale for mDavis Skodje problem");
  opts.set_option<double>("rtol", "relative tolerance for csp analysis e.g., 1e-2 ", &csp_rtolvar);
  opts.set_option<double>("atol", "absolute tolerance for csp analysis e.g., 1e-8 ", &csp_atolvar);
  opts.set_option<double>("y0", "initial value for y e.g., 2 ", &y0);
  opts.set_option<double>("z0", "initial value for y e.g., 1e-2 ", &z0);
  opts.set_option<double>("tend", "time end e.g., 4 ", &tend);
  opts.set_option<int>("nPoints", "number of points  e.g., 2000 ", &nPoints);

  const bool r_parse = opts.parse(argc, argv);
  if (r_parse) return 0; // print help return



  {
    CSP::ScopeGuard guard(argc, argv);

    const double dt = tend/nPoints;
    double tbegin(0);
    const int nElem(0);
    const int team_size(-1), vector_size(-1);

    using exec_space  = Kokkos::DefaultExecutionSpace;
    using host_exec_space = Kokkos::DefaultHostExecutionSpace;
    using device_type = typename Tines::UseThisDevice<exec_space>::type;
    using host_device_type = typename Tines::UseThisDevice<host_exec_space>::type;

    const auto exec_space_instance = exec_space();

    Tines::value_type_2d_view<double, device_type > state(" state vector", nPoints, 2 );
    Tines::value_type_2d_view<double, device_type > source(" source", nPoints, 2 );
    Tines::value_type_3d_view<double, device_type > jac(" jacobian", nPoints, 2, 2 );
    state_Davis_Skodje(state, tbegin, dt, nPoints, y0, z0, epsilon);
    exec_space_instance.fence();
    rhs_Davis_Skodje(state, source, epsilon);
    exec_space_instance.fence();
    jac_Davis_Skodje(state, jac, epsilon);
    exec_space_instance.fence();

    CSPKernelBatch kernelBatch(jac, source, state, nElem, csp_rtolvar, csp_atolvar);
    exec_space_instance.fence();
    // compute eigensolution and sort eigensolution w.r.t magnitude of eigenvalues
    kernelBatch.evalEigenSolution(team_size, vector_size);
    exec_space_instance.fence();

    // Setting CSP vectors:
    kernelBatch.evalCSPbasisVectors(team_size, vector_size); // A = eig_vec_R and B = A^{-1}
    exec_space_instance.fence();

    kernelBatch.evalCSP_Pointers(team_size, vector_size);
    exec_space_instance.fence();

    Tines::value_type_3d_view<double, host_device_type >  csp_pointers_host  = kernelBatch.getCSPPointers();
    exec_space_instance.fence();

    kernelBatch.evalTimeScales(team_size, vector_size);
    exec_space_instance.fence();

    Tines::value_type_2d_view<double, host_device_type >  time_scales_host = kernelBatch.getTimeScales();
    exec_space_instance.fence();

    kernelBatch.evalModalAmp(team_size, vector_size);
    exec_space_instance.fence();

    Tines::value_type_2d_view<double, host_device_type >  modal_ampl_host = kernelBatch.getModalAmp();
    exec_space_instance.fence();

    kernelBatch.evalM(team_size, vector_size);
    exec_space_instance.fence();

    Tines::value_type_1d_view<int, host_device_type > M_host = kernelBatch.getM();
    exec_space_instance.fence();

    Tines::value_type_1d_view<int, device_type > M = kernelBatch.getMDevice();

    std::string firstname("");

    std::string mNew_file_name = firstname + "_m.dat";
    FILE *fout = fopen ( (mNew_file_name).c_str(), "w" );

    std::string tau_file_name = firstname + "_tau.dat";
    FILE *fout_tau = fopen ( (tau_file_name).c_str(), "w" );

    // std::string num_rank_file_name = firstname + "_jac_numerical_rank.dat";
    // FILE *fout_num_rank = fopen ( (num_rank_file_name).c_str(), "w" );

    std::string magMode_file_name = firstname + "_magMode.dat";
    FILE *fout_magMode = fopen ( (magMode_file_name).c_str(), "w" );

    std::string state_file_name = firstname + "_state.dat";
    FILE *fout_state = fopen ( (state_file_name).c_str(), "w" );

    std::string time_file_name = firstname + "_time.dat";
    FILE *fout_time = fopen ( (time_file_name).c_str(), "w" );

    std::string cspp_ij_name = firstname + "_cspPointers.dat";
    FILE *fout_cspP = fopen ( (cspp_ij_name).c_str(), "w" );


    auto state_host = Kokkos::create_mirror_view(state);
    Kokkos::deep_copy(state_host, state);
    double t (0);

    for (int sp = 0; sp < nPoints; sp++) {

      fprintf(fout_time," %e \n", t);
      t +=dt;

      //state
      for (int k = 0; k<2; k++ ) {
            fprintf(fout_state,"%20.14e \t", state_host(sp,k));
      }
      fprintf(fout_state,"\n");

      fprintf(fout," %d \n", M_host(sp));

      // mode amplitud
      for (int k = 0; k<2; k++ ) {
            fprintf(fout_magMode,"%20.14e \t", modal_ampl_host(sp,k));
      }
      fprintf(fout_magMode,"\n");

      //tau
      for (int k = 0; k<2; k++ ) {
            fprintf(fout_tau,"%20.14e \t", time_scales_host(sp,k));
      }
      fprintf(fout_tau,"\n");

      // csp pointer
        for (int k = 0; k<2; k++ ) {
          for (int j = 0; j<2; j++ ) {
            fprintf(fout_cspP,"%15.10e \t", csp_pointers_host(sp,k,j));
          }
          fprintf(fout_cspP,"\n");
        }

    }


    fclose(fout_magMode);
    // fclose(fout_num_rank);
    fclose(fout_tau);
    fclose(fout);
    fclose(fout_state);
    fclose(fout_time);
    fclose(fout_cspP);

  }


  //====================================================================
    return 0;
  }
