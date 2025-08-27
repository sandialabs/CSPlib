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
#include "chem_elem_ODE_TChem.hpp"
#include "TChem.hpp"
#include "CSPlib_CommandLineParser.hpp"
#include "Tines.hpp"
#include "CSPlib_ConfigDefs.h"


#if defined(CSP_ENABLE_TPL_YAML_CPP)
#include "yaml-cpp/yaml.h"
#endif

int
main(int argc, char* argv[])
{
  std::string prefixPath("../data/GRI30/");
  std::string firstname("");
  int use_analytical_Jacobian(0);
  std::string chemFile(prefixPath + "chem.inp");
  std::string thermFile(prefixPath + "therm.dat");
  std::string inputFile(prefixPath + "input.dat");
  int nBatch(1);
  bool verbose(false);
  int team_size(-1);
  int vector_size(-1);
  bool write_files(false);
  bool compute_and_sort_eigenpairs(true);
  bool use_tpl_if_avail(true);
  bool use_shared_workspace(true);

  int team_size_hessenberg(-1), vector_size_hessenberg(-1);
  int team_size_right_eigenvector_schur(-1),  vector_size_right_eigenvector_schur(-1);
  int team_size_gemm(-1),  vector_size_gemm(-1);
  int team_size_sort_right_eigen_pairs(-1), vector_size_sort_right_eigen_pairs(-1);
  int team_size_jac(-1), vector_size_jac(-1);

  CSP::CommandLineParser opts("This example carries out a csp analysis with TChem model class");
  opts.set_option<std::string>
  ("chemfile", "Chem file name e.g., chem.inp",
  &chemFile);
  opts.set_option<std::string>
  ("thermfile", "Therm file name e.g., therm.dat", &thermFile);
  opts.set_option<int>
  ("useAnalyticalJacobian",
   "Use a analytical jacobian; 0: hand-derived analytical jacobian, 1: numerical jacobian, other number: sacado Analytical jacobian  ", &use_analytical_Jacobian);
  opts.set_option<std::string>
  ("inputfile", "database file name e.g., input.dat", &inputFile);
  opts.set_option<bool>(
      "verbose", "If true, printout state vector, jac ...", &verbose);
  opts.set_option<int>("batchsize",
                        "Batchsize the same state vector described in statefile is cloned",
                        &nBatch);
  opts.set_option<int>("team_thread_size", "time thread size ", &team_size);
  opts.set_option<int>( "vector_thread_size", "vector thread size", &vector_size);
  opts.set_option<bool>("write-files", "If true, write output files", &write_files);
  opts.set_option<bool>("use-shared-workspace-jacobian", "If true, jacobian uses shared workspace", &use_shared_workspace);
  opts.set_option<int>( "vector_thread_size_hessenberg", "vector thread size hessenberg ", &vector_size_hessenberg);
  opts.set_option<int>("team_thread_size_hessenberg", "time thread size hessenberg", &team_size_hessenberg);
  opts.set_option<int>( "vector_thread_size_right_eigenvector_schur", "vector thread size right-eigenvectors schur ", &vector_size_right_eigenvector_schur);
  opts.set_option<int>("team_thread_size_right_eigenvector_schur", "time thread size right-eigenvectors schur", &team_size_right_eigenvector_schur);
  opts.set_option<int>("team_thread_size_gemm", "time thread size gemm", &team_size_gemm);
  opts.set_option<int>( "vector_thread_size_gemm", "vector thread size gemm", &vector_size_gemm);
  opts.set_option<int>("team_thread_size_sort_right_eigen_pairs", "time thread size sort right eigen pairs", &team_size_sort_right_eigen_pairs);
  opts.set_option<int>( "vector_thread_size_sort_right_eigen_pairs", "vector thread size sort right eigen pairs", &vector_size_sort_right_eigen_pairs);
  opts.set_option<std::string>("outputs", "outputs prefix or directory to save output files e.g., pos = pos_Jac.dat ", &firstname);
  opts.set_option<bool>("use-third-party-library", "use third party libraries to solve eigenproblem if avilable ", &use_tpl_if_avail);
  opts.set_option<bool>("sort-eigensolution", "sort eigenvalues and eigenvectors ", &compute_and_sort_eigenpairs);
  opts.set_option<int>("team_thread_size_jacobian", "time thread size ", &team_size_jac);
  opts.set_option<int>( "vector_thread_size_jacobian", "vector thread size", &vector_size_jac);

  const bool r_parse = opts.parse(argc, argv);
  if (r_parse) return 0; // print help return
  {
    {

   // if ones what to set all task with same team_thread_size and vector_thread_size
    if ( team_size > 0 && vector_size > 0) {
      printf("Using team_thread_size %d and vector_thread_size %d in eigensolver  \n",team_size, vector_size);
      team_size_hessenberg = team_size ;
      vector_size_hessenberg=vector_size;
      team_size_right_eigenvector_schur=team_size;
      vector_size_right_eigenvector_schur=vector_size;
      team_size_gemm=team_size;
      vector_size_gemm=vector_size;
      team_size_sort_right_eigen_pairs=team_size;
      vector_size_sort_right_eigen_pairs=vector_size;
    }

    using host_exec_space = Kokkos::DefaultHostExecutionSpace;
    using exec_space = Kokkos::DefaultExecutionSpace;

    FILE *fout_times = fopen ((firstname + "_eigensolver_walltimes.json").c_str(), "w" );


    CSP::ScopeGuard guard(argc, argv);

#if 0
    ordinal_type nVars;
    real_type_3d_view jac;
    {
      // model class: interface to tchem
      ChemElemODETChem  model(chemFile, thermFile);
      const ordinal_type nSpec = model.NumOfSpecies();
      const ordinal_type stateVecDim = TChem::Impl::getStateVectorSize(nSpec);
      real_type_2d_view state("StateVector", nBatch, stateVecDim);
      auto state_host = Kokkos::create_mirror_view(state);
      auto state_host_at_0 = Kokkos::subview(state_host, 0, Kokkos::ALL());
      TChem::Test::readStateVector(inputFile, nSpec, state_host_at_0);
      printf("reading state vector from : %s \n", inputFile.c_str());

      TChem::Test::cloneView(state_host);
      Kokkos::deep_copy(state, state_host);
      // set state vector in model class
      model.setStateVectorDB(state);
      nVars = model.getNumOfVariables();


      // eval jacobian
      model.evalJacMatrixDevice(use_analytical_Jacobian,
                                team_size_jac, // use default team size
                                vector_size_jac, // use default vector size
                                use_shared_workspace ); // do not use share memory // save it for eigensolver computation

      //
      // get Jacobian
      model.getJacMatrixDevice(jac);

      if (write_files) {
        //jacobian

        FILE *fout_jacobian = fopen ( (firstname+"_Jac.dat").c_str(), "w" );
        FILE *fout_rhs = fopen ( (firstname+"_state.dat").c_str(), "w" );

        auto jac_host = Kokkos::create_mirror_view(jac);
        Kokkos::deep_copy(jac_host, jac);

        for (size_t i = 0; i < nBatch; i++) {
          // Jacobian
          for (int k = 0; k<nVars; k++ ) {
            for (int j = 0; j<nVars; j++ ) {
              fprintf(fout_jacobian,"%20.14e \t", jac_host(i,k,j));
            }
            fprintf(fout_jacobian,"\n");
          }

          for (int k = 0; k<state_host.extent(1); k++ ) {
              fprintf(fout_rhs,"%20.14e \t", state_host(i,k));
          }
          fprintf(fout_rhs,"\n");
          }

          fclose(fout_jacobian);
          fclose(fout_rhs);

        }
    }
#endif

#if 1
    // run Jacobian in this host
    real_type_3d_view jac;
    ordinal_type nVars;
    {
      using host_device_type = typename Tines::UseThisDevice<host_exec_space>::type;
      TChem::KineticModelData kmd = TChem::KineticModelData(chemFile, thermFile);
      const auto kmcd_host = TChem::createGasKineticModelConstData<host_device_type>(kmd);
      const ordinal_type nSpec = kmcd_host.nSpec ;
      nVars = nSpec + 1;

      const ordinal_type stateVecDim = TChem::Impl::getStateVectorSize(nSpec);
      real_type_2d_view_host state_host("StateVector", nBatch, stateVecDim);
      auto state_host_at_0 = Kokkos::subview(state_host, 0, Kokkos::ALL());
      TChem::Test::readStateVector(inputFile, nSpec, state_host_at_0);
      printf("reading state vector from : %s \n", inputFile.c_str());
      TChem::Test::cloneView(state_host);

      real_type_3d_view_host jac_host("AnalyticalJacIgnition_Host",nBatch, nSpec+1, nSpec+1);
      const ordinal_type per_team_extent = JacobianReduced::getWorkSpaceSize(kmcd_host);
      ordinal_type per_team_scratch = TChem::Scratch<real_type_1d_view_host >::shmem_size(per_team_extent);
      real_type_2d_view_host workspace("workspace", nBatch, per_team_scratch);
      TChem::JacobianReduced
           ::runHostBatch( state_host, //gas
                           jac_host,
                           workspace,
                           kmcd_host);
      // copy jacobian to device
      jac = real_type_3d_view("AnalyticalJacIgnitionDevice",nBatch, nSpec+1, nSpec+1);
      Kokkos::deep_copy(jac, jac_host);

      if (write_files) {
        //jacobian

        FILE *fout_jacobian = fopen ( (firstname+"_Jac.dat").c_str(), "w" );
        FILE *fout_rhs = fopen ( (firstname+"_state.dat").c_str(), "w" );

        for (size_t i = 0; i < nBatch; i++) {
          // Jacobian
          for (int k = 0; k<nVars; k++ ) {
            for (int j = 0; j<nVars; j++ ) {
              fprintf(fout_jacobian,"%20.14e \t", jac_host(i,k,j));
            }
            fprintf(fout_jacobian,"\n");
          }

          for (int k = 0; k<state_host.extent(1); k++ ) {
              fprintf(fout_rhs,"%20.14e \t", state_host(i,k));
          }
          fprintf(fout_rhs,"\n");
          }

          fclose(fout_jacobian);
          fclose(fout_rhs);

        }
    }
#endif

    Tines::control_type control;

    // const bool use_tpl_if_avail(false);
    // const bool compute_and_sort_eigenpairs(false);

    real_type_3d_view A("right CSP basis vectors ", nBatch, nVars, nVars );
    real_type_2d_view eigenvalues_real_part("eigenvalues real part ", nBatch, nVars );
    real_type_2d_view eigenvalues_imag_part("eigenvalues imag part ", nBatch, nVars );
    const int wlen = 3 * nVars * nVars + 2 * nVars;
    real_type_2d_view work_eigensolver("work eigen solver", nBatch, wlen);

    /// tpl use
    control["Bool:UseTPL"].bool_value = use_tpl_if_avail;
    control["Bool:SolveEigenvaluesNonSymmetricProblem:Sort"].bool_value = compute_and_sort_eigenpairs;
    /// eigen solve
    if ( team_size_hessenberg > 0 && vector_size_hessenberg > 0) {
      printf("Hessenberg: team_thread_size %d and vector_thread_size %d in eigensolver  \n",team_size_hessenberg,vector_size_hessenberg);

      control["IntPair:Hessenberg:TeamSize"].int_pair_value = std::pair<int,int>(team_size_hessenberg,vector_size_hessenberg);
    }

    if  ( team_size_right_eigenvector_schur > 0 && vector_size_right_eigenvector_schur > 0) {
      printf("RightEigenvectorSchur: team_thread_size %d and vector_thread_size %d in eigensolver  \n",team_size_right_eigenvector_schur,vector_size_right_eigenvector_schur);
      control["IntPair:RightEigenvectorSchur:TeamSize"].int_pair_value = std::pair<int,int>(team_size_right_eigenvector_schur,vector_size_right_eigenvector_schur);
    }

    if ( team_size_gemm > 0 && vector_size_gemm > 0) {
      printf("Gemm: team_thread_size %d and vector_thread_size %d in eigensolver  \n",team_size_gemm,vector_size_gemm);

      control["IntPair:Gemm:TeamSize"].int_pair_value = std::pair<int,int>(team_size_gemm,vector_size_gemm);
    }

    if ( team_size_sort_right_eigen_pairs > 0 && vector_size_sort_right_eigen_pairs > 0) {
      control["IntPair:SortRightEigenPairs:TeamSize"].int_pair_value = std::pair<int,int>(team_size_sort_right_eigen_pairs,vector_size_sort_right_eigen_pairs);
    } // else use tines default values

    Kokkos::Timer timer;

    timer.reset();
    Tines::SolveEigenvaluesNonSymmetricProblemDevice<exec_space>
       ::invoke( exec_space(), jac, eigenvalues_real_part,
                eigenvalues_imag_part, A, work_eigensolver,
                 control);
    exec_space().fence();
    const real_type t_comp_eigen_solution = timer.seconds();
    fprintf(fout_times, "{%s: %d, \n","\"Number of samples\"", nBatch);
    fprintf(fout_times, "%s: %20.14e, \n","\"Total time\"", t_comp_eigen_solution);
    fprintf(fout_times, "%s: %20.14e }\n","\"Time per sample\"", t_comp_eigen_solution/nBatch);

    fclose(fout_times);

    if (write_files) {
      // eigen solution
      FILE *fout_eig_val_real = fopen ( (firstname + "_eig_val_real.dat").c_str(), "w" );
      FILE *fout_eig_val_imag = fopen ( (firstname + "_eig_val_imag.dat").c_str(), "w" );
      FILE *fout_eig_vec_R = fopen ( (firstname + "_eig_vec_R.dat").c_str(), "w" );


      auto A_host = Kokkos::create_mirror_view(A);
      Kokkos::deep_copy( A_host, A);

      auto eigenvalues_real_part_host = Kokkos::create_mirror_view(eigenvalues_real_part);
      Kokkos::deep_copy( eigenvalues_real_part_host, eigenvalues_real_part);

      auto eigenvalues_imag_part_host = Kokkos::create_mirror_view(eigenvalues_imag_part);
      Kokkos::deep_copy( eigenvalues_imag_part_host, eigenvalues_imag_part);

      for (size_t i = 0; i < nBatch; i++) {


        // eigenvector rigth
        for (int k = 0; k<nVars; k++ ) {
          for (int j = 0; j<nVars; j++ ) {
            fprintf(fout_eig_vec_R,"%15.10e \t", A_host(i,k,j));
          }
          fprintf(fout_eig_vec_R,"\n");
        }

        // eig_val_real
        for (int k = 0; k<nVars; k++ )
          fprintf(fout_eig_val_real,"%20.14e \t", eigenvalues_real_part_host(i,k) );
        fprintf(fout_eig_val_real,"\n");

        // eig_val_imag
        for (int k = 0; k<nVars; k++ )
          fprintf(fout_eig_val_imag,"%20.14e \t", eigenvalues_imag_part_host(i,k));
        fprintf(fout_eig_val_imag,"\n");

      } // end nBatch

      fclose(fout_eig_vec_R);
      fclose(fout_eig_val_imag);
      fclose(fout_eig_val_real);

    }



  }

  }

  return 0;

}
