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
#include "indexBatch.hpp"
#include "kernelBatch.hpp"
#include "CSPlib_CommandLineParser.hpp"
#include "TChem.hpp"
#include "kernel.hpp"

int main(int argc, char* argv[])
{

  std::string firstname("");
  CSP::CommandLineParser opts("This example carries out a csp analysis with TChem model class");
  opts.set_option<std::string>("prefix", "prefix to save output files e.g., pos_ ", &firstname);

  const bool r_parse = opts.parse(argc, argv);
  if (r_parse) return 0; // print help return

  {
  CSP::ScopeGuard guard(argc, argv);

  const bool detail = false;
  TChem::     exec_space().print_configuration(std::cout, detail);
  TChem::host_exec_space().print_configuration(std::cout, detail);

  using real_type = double;
  using ordinal_type = int;
  using exec_space = Kokkos::DefaultExecutionSpace;
  using device_type = typename Tines::UseThisDevice<exec_space>::type;
  using real_type_3d_view_type = Tines::value_type_3d_view<real_type,device_type>;
  using real_type_2d_view_type = Tines::value_type_2d_view<real_type,device_type>;
  using ordinal_type_1d_view_type = Tines::value_type_1d_view<ordinal_type,device_type>;

  using host_exec_space = Kokkos::DefaultHostExecutionSpace;
  using host_device_type = typename Tines::UseThisDevice<host_exec_space>::type;

  using ordinal_type_1d_view_type_host = Tines::value_type_1d_view<ordinal_type,host_device_type>;
  using real_type_2d_view_type_host = Tines::value_type_2d_view<real_type,host_device_type>;
  using real_type_3d_view_type_host = Tines::value_type_3d_view<real_type,host_device_type>;

  const ordinal_type nBatch = 10;
  const ordinal_type n_variables = 15;
  const ordinal_type n_processes = 20;

  const ordinal_type nElem(3);
  const real_type csp_rtolvar(1e-3), csp_atolvar(1e-12);

  real_type_2d_view_type state("state", nBatch, n_variables);
  real_type_2d_view_type rhs("rhs",     nBatch, n_variables);
  real_type_3d_view_type jac("jac",     nBatch, n_variables, n_variables);
  real_type_3d_view_type S("S",         nBatch, n_variables, n_processes);
  real_type_2d_view_type RoP("RoP",     nBatch, n_processes);

  Kokkos::Random_XorShift64_Pool<device_type> random(13718);
  Kokkos::fill_random(state, random, real_type(1.0));
  Kokkos::fill_random(rhs, random, real_type(1.0));
  Kokkos::fill_random(jac, random, real_type(1.0));
  Kokkos::fill_random(S, random, real_type(1.0));
  Kokkos::fill_random(RoP, random, real_type(1.0));


  printf("Working in kernel ...\n");

  CSPKernelBatch kernelBatch(jac, rhs, state, nElem, csp_rtolvar, csp_atolvar);

  // compute eigensolution and sort w.r.t magnitude of eigenvalues
  kernelBatch.evalEigenSolution();
  // sort eigensolution w.r.t magnitude of eigenvalues
  // Setting CSP vectors:
  kernelBatch.evalCSPbasisVectors(); // A = eig_vec_R and B = A^{-1}

  real_type_3d_view_type B = kernelBatch.getLeftCSPVecDevice();
  real_type_3d_view_type A = kernelBatch.getRightCSPVecDevice();

  kernelBatch.evalCSP_Pointers();

  kernelBatch.evalTimeScales();
  real_type_2d_view_type_host time_scales_host = kernelBatch.getTimeScales();

  kernelBatch.evalModalAmp();
  real_type_2d_view_type_host modal_ampl_host = kernelBatch.getModalAmp();

  // Exhausted mode

  kernelBatch.evalM();

  ordinal_type_1d_view_type_host M_host = kernelBatch.getM();
  ordinal_type_1d_view_type M = kernelBatch.getMDevice();

  CSPIndexBatch indexBatch( A, B, S, RoP, M );

  indexBatch.evalParticipationIndex();
  real_type_3d_view_type_host PartIndex;
  PartIndex = indexBatch.getParticipationIndex();

  indexBatch.evalImportanceIndexSlow();
  real_type_3d_view_type_host SlowImpoIndex;

  SlowImpoIndex =  indexBatch.getImportanceIndexSlow();

  indexBatch.evalImportanceIndexFast();
  real_type_3d_view_type_host FastImpoIndex;
  FastImpoIndex = indexBatch.getImportanceIndexFast();

  // index class
  std::string P_ik_name = firstname + "_ParticipationIndex.dat";
  std::string Islow_jk_name = firstname + "_SlowImportanceIndex.dat";
  std::string Ifast_jk_name = firstname + "_FastImportanceIndex.dat";

  // index
  FILE *fout_Pim = fopen ( (P_ik_name).c_str(), "w" );
  FILE *fout_Isi = fopen ( (Islow_jk_name).c_str(), "w" );
  FILE *fout_Ifn = fopen ( (Ifast_jk_name).c_str(), "w" );

  // kernel
  std::string m_file_name = firstname + "_m.dat";
  std::string tau_file_name = firstname + "_tau.dat";
  std::string f_file_name = firstname + "_f.dat";

  FILE * fout = fopen ( (m_file_name).c_str(), "w" );
  FILE *fout_tau = fopen ( (tau_file_name).c_str(), "w" );
  FILE *fout_f = fopen ( (f_file_name).c_str(), "w" );

  std::string eig_val_real_file_name = firstname + "_eig_val_real.dat";
  std::string eig_val_imag_file_name = firstname + "_eig_val_imag.dat";
  FILE *fout_eig_val_real = fopen ( (eig_val_real_file_name).c_str(), "w" );
  FILE *fout_eig_val_imag = fopen ( (eig_val_imag_file_name).c_str(), "w" );

  std::string eig_vec_R_file_name = firstname + "_eig_vec_R.dat";
  FILE *fout_eig_vec_R = fopen ( (eig_vec_R_file_name).c_str(), "w" );
  std::string cspp_ij_name = firstname + "_cspPointers.dat";
  FILE *fout_cspP = fopen ( (cspp_ij_name).c_str(), "w" );


  real_type_3d_view_type_host A_host("A host", nBatch, n_variables, n_variables );
  Kokkos::deep_copy( A_host, A);
  real_type_2d_view_type_host eigenvalues_real_part_host = kernelBatch.getEigenValuesRealPart();
  real_type_2d_view_type_host eigenvalues_imag_part_host = kernelBatch.getEigenValuesImagPart();

  real_type_3d_view_type_host csp_pointers_host  = kernelBatch.getCSPPointers();


  std::vector<real_type> A_1d(n_variables * n_variables);

  for (size_t i = 0; i < nBatch; i++) {

    // state vector
    const auto state_at_i = Kokkos::subview(state, i, Kokkos::ALL());
    std::vector< real_type > state_std;
    TChem::convertToStdVector(state_std, state_at_i);
    // source vector
    const auto source_at_i = Kokkos::subview(rhs, i, Kokkos::ALL());
    std::vector< real_type > source_std;
    TChem::convertToStdVector(source_std, source_at_i);
    // jacobian
    const auto jac_at_i = Kokkos::subview(jac, i, Kokkos::ALL(), Kokkos::ALL());
    std::vector< std::vector< real_type > > jac_std;
    TChem::convertToStdVector(jac_std, jac_at_i);

    Kernel ker(n_variables, state_std, source_std, jac_std);

    // eigenvalue real
    const auto eigenvalues_real_part_host_at_i = Kokkos::subview(eigenvalues_real_part_host, i, Kokkos::ALL());
    std::vector< real_type  > eig_val_real_std;
    TChem::convertToStdVector(eig_val_real_std, eigenvalues_real_part_host_at_i);

    // eigenvalue imag
    const auto eigenvalues_imag_part_host_at_i = Kokkos::subview(eigenvalues_imag_part_host, i, Kokkos::ALL());
    std::vector< real_type  > eig_val_imag_std;
    TChem::convertToStdVector(eig_val_imag_std, eigenvalues_imag_part_host_at_i);


    // rigth eigenvectors

    const auto A_host_at_i = Kokkos::subview(A_host, i, Kokkos::ALL(), Kokkos::ALL());
    std::vector< std::vector< real_type > > A_std;
    TChem::convertToStdVector(A_std, A_host_at_i);

    // convert 2D TO 1D
    //row-major order
    int count=0;
    for (size_t k=0; k<n_variables; k++) {
      for (size_t j=0; j<n_variables; j++) {
        A_1d[count] = A_std[k][j];
        count++;
      }
    }

    ker.setEigenValVec(eig_val_real_std, eig_val_imag_std, A_1d);
    printf(" \n \n Cheking EigenSolution ... %zu\n", i );
    ker.DiagEigValVec();

    //Participation index
    for (int k = 0; k<n_variables; k++ ) {
      for (int j = 0; j<(n_processes); j++ ) {
        fprintf(fout_Pim,"%15.10e \t", PartIndex(i,k,j));
       }
      fprintf(fout_Pim,"\n");
    }

    // Slow importance index
    for (int k = 0; k<n_variables; k++ ) {
      for (int j = 0; j<(n_processes); j++ ) {
        fprintf(fout_Isi,"%15.10e \t", SlowImpoIndex(i,k,j));
      }
    fprintf(fout_Isi,"\n");
    }

    //
    //Fast importance index
    for (int k = 0; k<n_variables; k++ ) {
      for (int j = 0; j<(n_processes); j++ ) {
        fprintf(fout_Ifn,"%15.10e \t", FastImpoIndex(i,k,j));
      }
      fprintf(fout_Ifn,"\n");
    }

    // eig_val_real
    for (int k = 0; k<n_variables; k++ )
      fprintf(fout_eig_val_real,"%20.14e \t", eigenvalues_real_part_host(i,k) );
    fprintf(fout_eig_val_real,"\n");


    // eig_val_imag
    for (int k = 0; k<n_variables; k++ )
      fprintf(fout_eig_val_imag,"%20.14e \t", eigenvalues_imag_part_host(i,k));
    fprintf(fout_eig_val_imag,"\n");
    //
    //kernel

    //M
   fprintf(fout," %d \n", M_host(i));

    // eigenvector rigth
    for (int k = 0; k<n_variables; k++ ) {
      for (int j = 0; j<(n_variables); j++ ) {
        fprintf(fout_eig_vec_R,"%15.10e \t", A_host(i,k,j));
      }
      fprintf(fout_eig_vec_R,"\n");
    }

    //tau
    for (int k = 0; k<n_variables; k++ ) {
            fprintf(fout_tau,"%20.14e \t", time_scales_host(i,k));
      }
    fprintf(fout_tau,"\n");

    // f
    for (int k = 0; k<n_variables; k++ ) {
            fprintf(fout_f,"%20.14e \t", modal_ampl_host(i,k));
    }
    fprintf(fout_f,"\n");

    // csp pointer
    for (int k = 0; k<n_variables; k++ ) {
      for (int j = 0; j<(n_variables); j++ ) {
        fprintf(fout_cspP,"%15.10e \t", csp_pointers_host(i,k,j));
      }
      fprintf(fout_cspP,"\n");
    }

  }

  // index class
  fclose(fout_Pim);
  fclose(fout_Isi);
  fclose(fout_Ifn);

  fclose(fout_eig_vec_R);
  fclose(fout_eig_val_imag);
  fclose(fout_eig_val_real);

  // kernel class
  fclose(fout);
  fclose(fout_f);
  fclose(fout_tau);
  fclose(fout_cspP);

}
return 0;
}
