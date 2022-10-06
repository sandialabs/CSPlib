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
#include "kernelBatch.hpp"

template <typename ViewType>
void readInputs2D(const std::string& filename, ViewType &A, int& nsamples, int& nvars)
{
  std::ifstream file(filename);
  int one(1), nrows(0), ncols(0); // number of samples, number of rows, number of columns
  if (file.is_open()) {
    file >> nsamples;
    file >> nrows;
    file >> ncols;
    // double value;
    A = ViewType("A", nsamples, nrows, ncols);

    for (int i = 0; i < nsamples; ++i) // samples
      for (int j = 0; j < nrows; ++j) // rows
        for (int k = 0; k < ncols; ++k)  // columns
          file >> A(i, j, k);
    } else {
      std::cerr << " readInputs: cannot open file "+ filename +"\n";
      exit(-1);
  }
  nvars = ncols;
  file.close();
}

template <typename ViewType>
void readInputs1D(const std::string& filename, ViewType &A, int& nsamples, int& nvars)
{
  std::ifstream file(filename);
  int one(1), nrows(0), ncols(0); // number of samples, number of rows, number of columns
  if (file.is_open()) {
    file >> nsamples;
    file >> nrows;
    file >> ncols;
      /// read 2d
    A = ViewType("A", nsamples, ncols);
    for (int i = 0; i < nsamples; ++i) // samples
      for (int k = 0; k < ncols; ++k) // columns
        file >> A(i, k);
    } else {
      std::cerr << " readInputs: cannot open file "+ filename +"\n";
      exit(-1);
  }
  nvars = ncols;
  file.close();

}

int main(int argc, char *argv[]) {

  double csp_rtolvar(1.e-3); //1.e-7; // 1.e+3; //
  double csp_atolvar(1.e-14); //1.e-8; // 1.e+3; //
  bool verbose(false);

  std::string x_filename ="x.txt";
  std::string f_filename ="f.txt";
  std::string jac_filename ="J.txt";

  CSP::CommandLineParser opts("This example Number of exhausted and time scale for mDavis Skodje problem");
  opts.set_option<double>("rtol", "relative tolerance for csp analysis e.g., 1e-2 ", &csp_rtolvar);
  opts.set_option<double>("atol", "absolute tolerance for csp analysis e.g., 1e-8 ", &csp_atolvar);
  opts.set_option<std::string>("x-filename", "x file name e.g., x.txt", &x_filename);
  opts.set_option<std::string>("f-filename", "f file name e.g., f.txt", &f_filename);
  opts.set_option<std::string>("J-filename", "J file name e.g., J.txt", &jac_filename);
  opts.set_option<bool>(
      "verbose", "If true, printout state vector, jac ...", &verbose);

  const bool r_parse = opts.parse(argc, argv);
  if (r_parse) return 0; // print help return

  {
    CSP::ScopeGuard guard(argc, argv);

    const int nElem(0);
    const int team_size(-1), vector_size(-1);

    using exec_space  = Kokkos::DefaultExecutionSpace;
    using host_exec_space = Kokkos::DefaultHostExecutionSpace;
    using device_type = typename Tines::UseThisDevice<exec_space>::type;
    using host_device_type = typename Tines::UseThisDevice<host_exec_space>::type;

    const auto exec_space_instance = exec_space();
    int nsamples(1), nvars(1);
    Tines::value_type_2d_view<double, device_type > state_host("x", nsamples, nvars);
    Tines::value_type_2d_view<double, device_type > source_host("f", nsamples, nvars);
    Tines::value_type_3d_view<double, device_type > jac_host("x", nsamples, nvars, nvars);

    readInputs1D(x_filename, state_host, nsamples, nvars);
    printf("filename x %s number of samples %d, number of variables %d \n", x_filename.c_str(), nsamples, nvars );
    readInputs1D(f_filename, source_host, nsamples, nvars);
    printf("filename f %s number of samples %d, number of variables %d \n", f_filename.c_str(), nsamples, nvars );
    readInputs2D(jac_filename, jac_host, nsamples, nvars);
    printf("filename f %s number of samples %d, number of variables %d \n", jac_filename.c_str(), nsamples, nvars );

    if (verbose) {
      for (size_t isp = 0; isp < nsamples; isp++){
        printf("sample No %d \n", isp);
        for (int i = 0; i < nvars; i++)
          printf("x(%d):%e, f(%d):(%e)  \n", i,  state_host(isp,i), i, source_host(isp,i) );
      }

      for (size_t isp = 0; isp < nsamples; isp++){
        printf("sample No %d \n", isp);
        for (int i = 0; i < nvars; i++) {
          for (int j = 0; j < nvars; j++)
            printf(" J(%d,%d):%e", i, j,  jac_host(isp,i,j));
          printf("\n");
        } // end nvars
      } // end samples

    }

    // move inputs from host to device
    auto jac = Kokkos::create_mirror_view(jac_host);
    Kokkos::deep_copy(jac, jac_host);

    auto state = Kokkos::create_mirror_view(state_host);
    Kokkos::deep_copy(state, state_host);

    auto source = Kokkos::create_mirror_view(source_host);
    Kokkos::deep_copy(source, source_host);

    CSPKernelBatch kernelBatch(jac, source, state, nElem, csp_rtolvar, csp_atolvar);
    exec_space_instance.fence();
    // compute eigensolution and sort eigensolution w.r.t magnitude of eigenvalues
    kernelBatch.evalEigenSolution(team_size, vector_size);
    exec_space_instance.fence();

    // Setting CSP vectors:
    kernelBatch.evalCSPbasisVectors(team_size, vector_size); // A = eig_vec_R and B = A^{-1}
    exec_space_instance.fence();

    kernelBatch.evalTimeScales(team_size, vector_size);
    exec_space_instance.fence();

    Tines::value_type_2d_view<double, host_device_type >  time_scales_host = kernelBatch.getTimeScales();
    exec_space_instance.fence();

    kernelBatch.evalModalAmp(team_size, vector_size);
    exec_space_instance.fence();

    kernelBatch.evalM(team_size, vector_size);
    exec_space_instance.fence();

    Tines::value_type_1d_view<int, host_device_type > M_host = kernelBatch.getM();
    exec_space_instance.fence();

    std::string firstname("");

    std::string m_file_name = firstname + "_m.dat";
    FILE *fout = fopen ( (m_file_name).c_str(), "w" );

    std::string tau_file_name = firstname + "_tau.dat";
    FILE *fout_tau = fopen ( (tau_file_name).c_str(), "w" );

    for (int sp = 0; sp < nsamples; sp++) {

      fprintf(fout," %d \n", M_host(sp));
      //tau
      for (int k = 0; k<nvars; k++ ) {
            fprintf(fout_tau,"%20.14e \t", time_scales_host(sp,k));
      }
      fprintf(fout_tau,"\n");

    }


    fclose(fout_tau);
    fclose(fout);

  }


  //====================================================================
    return 0;
  }
