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
// #include "TChem.hpp"
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

  // const bool detail = false;
  // TChem::     exec_space::print_configuration(std::cout, detail);
  // TChem::host_exec_space::print_configuration(std::cout, detail);

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


  using policy_type = Kokkos::TeamPolicy<exec_space>;

  const ordinal_type nBatch = 10;
  const ordinal_type n_variables = 15;
  const ordinal_type n_processes = 20;

  ordinal_type_1d_view_type M("state", nBatch);
  real_type_3d_view_type A("A",     nBatch, n_variables, n_variables);
  real_type_3d_view_type S("S",         nBatch, n_variables, n_processes);
  real_type_2d_view_type RoP("RoP",     nBatch, n_processes);

  Kokkos::Random_XorShift64_Pool<device_type> random(13718);
  Kokkos::fill_random(M, random, ordinal_type(3));
  Kokkos::fill_random(A, random, real_type(1.0));
  Kokkos::fill_random(S, random, real_type(1.0));
  Kokkos::fill_random(RoP, random, real_type(1.0));


  real_type_2d_view_type work(" work ", nBatch, 2 * n_variables + n_variables * n_variables );
  real_type_3d_view_type B("B",     nBatch, n_variables, n_variables);

  using kernel_csplib_device = CSP::KernelComputation<device_type>;
  policy_type policy(nBatch, Kokkos::AUTO());

  kernel_csplib_device::evalLeftCSP_BasisVectorsBatch(
                        "Test::computeCSPbasisVectors::runDeviceBatch",
                        policy, A, B, work);


  printf("Working in kernel ...\n");

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

  std::string csp_left_vectors_name = firstname + "_CSPLeftVectors.dat";

  // index
  FILE *fout_Pim = fopen ( (P_ik_name).c_str(), "w" );
  FILE *fout_Isi = fopen ( (Islow_jk_name).c_str(), "w" );
  FILE *fout_Ifn = fopen ( (Ifast_jk_name).c_str(), "w" );

  FILE *fout_csp_left_vectors = fopen ( (csp_left_vectors_name).c_str(), "w" );


  real_type_3d_view_type_host B_host("B host", nBatch, n_variables, n_variables );
  Kokkos::deep_copy( B_host, A);



  for (size_t i = 0; i < nBatch; i++) {

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

    // eigenvector rigth
    for (int k = 0; k<n_variables; k++ ) {
      for (int j = 0; j<(n_variables); j++ ) {
        fprintf(fout_csp_left_vectors,"%15.10e \t", B_host(i,k,j));
      }
      fprintf(fout_csp_left_vectors,"\n");
    }


  }

  // index class
  fclose(fout_Pim);
  fclose(fout_Isi);
  fclose(fout_Ifn);
  fclose(fout_csp_left_vectors);



}
return 0;
}
