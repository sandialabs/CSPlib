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


#include <iostream>
#include <vector>
#include <string>
#include "chem_elem_ODE_TChem.hpp"
#include "CSPlib_CommandLineParser.hpp"

#include "kernel.hpp"
#include "index.hpp"
#include "tools.hpp"
#include "util.hpp"
#include "indexBatch.hpp"
#include "kernelBatch.hpp"

#if defined(CSP_ENABLE_TPL_YAML_CPP)
#include "yaml-cpp/yaml.h"
#endif

#include "vio.h"
typedef Kokkos::DefaultExecutionSpace SpT;
typedef Kokkos::DefaultHostExecutionSpace HpT;

#define SET_TEAM_VECTOR_SIZE(name)                   \
    if (func_inx.size() > 0 ){                       \
      team_size = team_size_v[func_inx[name]];       \
      vector_size = vector_size_v[func_inx[name]];   \
    }                                                \
    printf(" %s using team_size %d  vector_size %d \n", name, team_size, vector_size );

void readDataBase(const std::string& filename,
                  std::vector< std::vector<double> >& database,
                  const int &nStateVariables)
{
  double atposition;
  printf("Reading from database \n ");
  std::string line;
  std::ifstream ixfs(filename);
  if (ixfs.is_open()) {

    while(ixfs >> atposition){
      // number of state variables plus time/iteration
      std::vector<double>vec(nStateVariables + 1,0.0); //
      vec[0] = atposition;
      for (int i=1; i<nStateVariables + 1; i++)
        ixfs >> vec[i];
      database.push_back(vec);
    }
  } else {
      std::cerr << " readDataBase: cannot open file "+ filename +"\n";
      exit(-1);
    }

  ixfs.close();
}

int main(int argc, char *argv[]) {

  // reactions mechanism and therm files
  std::string prefixPath("../runs/GRI3/data/");
  std::string firstname("");
  double csp_rtolvar(1.e-2);
  double csp_atolvar(1.e-8);

  std::string chemFile(prefixPath + "chem.inp");
  std::string thermFile(prefixPath + "therm.dat");
  std::string inputFile(prefixPath + "input.dat");
  std::string csp_device_settings("csp_device_settings.yaml");
  bool useTChemSolution(true);
  bool verbose(false);
  int increment(1);
  bool write_files(false);
  int use_analytical_Jacobian(0);
  bool use_yaml_settings_file(false);
  bool compute_csp_indices(true);
  // it uses for numerical experiments
  bool useCloneSamples(false);
  bool use_yaml(false);

  std::string variable1("Temperature");
  std::string variable2("CH4");


  CSP::CommandLineParser opts("This example carries out a csp analysis with TChem model class");
  opts.set_option<std::string>("prefix", "prefix to save output files e.g., pos_ ", &firstname);
  opts.set_option<double>("rtol", "relative tolerance for csp analysis e.g., 1e-2 ", &csp_rtolvar);
  opts.set_option<double>("atol", "absolute tolerance for csp analysis e.g., 1e-8 ", &csp_atolvar);
  opts.set_option<bool>("use-yaml", "If true, use yaml to parse input file", &use_yaml);
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
  opts.set_option<bool>
  ("useTChemSolution", "Use a solution produced by TChem e.g., true", &useTChemSolution);
  opts.set_option<bool>(
      "verbose", "If true, printout state vector, jac ...", &verbose);
  //
  opts.set_option<bool>(
      "use-clone-samples", "If true, one state vector will be clone.", &useCloneSamples);

  opts.set_option<bool>(
      "write-files", "If true, write output files", &write_files);
  opts.set_option<std::string>
  ("device-seeting-file", "Device setting file name e.g., csp_device_settings.yaml", &csp_device_settings);
  opts.set_option<bool>(
      "use-device-setting-file", "If true, read settings for Device from a the yaml file csp_device_settings.yaml", &use_yaml_settings_file);
  opts.set_option<bool>(
      "compute-csp-indices", "If true, computes fast/slow importances and participation indices", &compute_csp_indices);
  opts.set_option<int>("increment", " increment in database, e.g. 2  ", &increment);
  opts.set_option<std::string>
  ("variable1", "Compute Importance index for variable  e.g., Temperature", &variable1);
  opts.set_option<std::string>
  ("variable2", "Compute Importance index for variable  e.g., CH4", &variable2);

  const bool r_parse = opts.parse(argc, argv);
  if (r_parse) return 0; // print help return

  {

    ordinal_type team_size_jac(-1), vector_size_jac(-1);
    ordinal_type team_size(-1), vector_size(-1);
    bool use_shared_workspace(true);

    std::vector<ordinal_type> vector_size_v;
    std::vector<ordinal_type> team_size_v;
    std::map<std::string, ordinal_type> func_inx;

    // CSP gets Yaml from Tines

    if (use_yaml_settings_file){

      #if defined(CSP_ENABLE_TPL_YAML_CPP)
      std::string file_name_yaml(csp_device_settings);
      YAML::Node root = YAML::LoadFile(file_name_yaml);

      if (root["Jacobian"]){

        vector_size_jac = root["Jacobian"]["vector_size"].as<ordinal_type>();
        team_size_jac = root["Jacobian"]["team_size"].as<ordinal_type>();
        use_shared_workspace = root["Jacobian"]["use_shared_workspace"].as<bool>();
        const auto Jacobian_type = root["Jacobian"]["Jacobian_type"].as<std::string>();

        if (Jacobian_type == "Analytical")
          use_analytical_Jacobian = 0;
        else if (Jacobian_type == "Numerical")
          use_analytical_Jacobian = 1;
        else if (Jacobian_type == "Sacado")
          use_analytical_Jacobian = 2;
        else {
          printf("Jacobian type does not exit %s\n", Jacobian_type.c_str() );
          printf("Use Analytical, or Numerical, or Sacado \n" );
          exit(1);
        }



      }
      ordinal_type count(0);
      for (auto const& ifunction : root["Index_class"])
      {
        func_inx.insert(std::pair<std::string, ordinal_type>(ifunction.first.as<std::string>() ,count ));
        vector_size_v.push_back(ifunction.second["vector_size"].as<ordinal_type>());
        team_size_v.push_back(ifunction.second["team_size"].as<ordinal_type>());
        count++;
      }

      for (auto const& ifunction : root["Kernel_class"])
      {
        func_inx.insert(std::pair<std::string, ordinal_type>(ifunction.first.as<std::string>() ,count ));
        vector_size_v.push_back(ifunction.second["vector_size"].as<ordinal_type>());
        team_size_v.push_back(ifunction.second["team_size"].as<ordinal_type>());
        count++;
      }

      for (auto const& ifunction : root["Model_class"])
      {
        func_inx.insert(std::pair<std::string, ordinal_type>(ifunction.first.as<std::string>() ,count ));
        vector_size_v.push_back(ifunction.second["vector_size"].as<ordinal_type>());
        team_size_v.push_back(ifunction.second["team_size"].as<ordinal_type>());
        count++;
      }

      #else
      printf("This example requires Yaml ...\n" );
      #endif

    }

    {
    CSP::ScopeGuard guard(argc, argv);

    std::string time_file_name = firstname + "_csp_times.dat";
    FILE *fout_times = fopen (time_file_name.c_str(), "w" );

    const auto exec_space_instance = TChem::exec_space();


    Kokkos::Timer timer;

    fprintf(fout_times, "{\n");
    fprintf(fout_times, " \"Model Class\": \n {\n");

    timer.reset();
    ChemElemODETChem  model(chemFile, thermFile, use_yaml);
    exec_space_instance.fence();
    const real_type t_int_model_class = timer.seconds();
    fprintf(fout_times, "%s: %20.14e, \n","\"Int model class\"", t_int_model_class);

    real_type t_read_state(0);


    if (useTChemSolution) {
      // read a data base from a TChem++ Ingition solution
      std::vector<std::string> var_names;
      timer.reset();
      model.readIgnitionZeroDDataBaseFromFile(inputFile, var_names, increment);
      exec_space_instance.fence();
      t_read_state = timer.seconds();
      fprintf(fout_times, "%s: %20.14e, \n","\"Read state vector\"", t_read_state);

    } else {
      // read one state vector a make many clone samples; this is use for numerical experimentes
      // this option is only enable if useTChemSolution is false
      if (useCloneSamples) {

        printf("-------------------------------------------------------\n");
        printf("--------------------Warning----------------------------\n");
        printf("Using cloned samples ... only for numerical experiments\n");
        // in this scenario increment is equal to increment number of samples
        auto nBatch = increment;
        printf("Number of samples %d : \n", nBatch);
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

      } else {
        // read a data base that was not produced by TChem
        std::vector<std::vector <double> > state_db_read;
        // Density, pressure, temperature and species mass fraction
        const ordinal_type nSpec = model.NumOfSpecies();
        const ordinal_type numofStateVariables = TChem::Impl::getStateVectorSize(nSpec);
        readDataBase(inputFile, state_db_read, numofStateVariables );
        model.setStateVectorDB(state_db_read);
      }



    }

    printf("Working in model ...\n");

    // get number of variables in the ODE system
    auto ndiff_var = model.getNumOfVariables();
    std::cout<< "ndiff_var = "<< ndiff_var <<"\n";

    //save species names in a file to be use in post-processing
    {
      // get name of species
      std::vector<std::string> spec_name;
      model.getSpeciesNames(spec_name);

      std::string species_name = "speciesNames.dat";
      FILE *fout = fopen (  (species_name).c_str() , "w" );
      for (int i = 0; i<spec_name.size(); i++ )
          fprintf(fout,"%s \n", (spec_name[i]).c_str());
      fclose(fout);
    }

    //computes RHS
    SET_TEAM_VECTOR_SIZE("evalSourceVector")
    timer.reset();
    model.evalSourceVectorDevice(team_size, vector_size);
    exec_space_instance.fence();
    const real_type t_eval_source = timer.seconds();
    fprintf(fout_times, "%s: %20.14e, \n","\"Eval source term\"", t_eval_source);

    //computes jacobian
    timer.reset();
    // model.evalJacMatrix(use_analytical_Jacobian);
    model.evalJacMatrixDevice(use_analytical_Jacobian,
                              team_size_jac,
                              vector_size_jac,
                              use_shared_workspace );

    exec_space_instance.fence();
    const real_type t_eval_jacobian = timer.seconds();
    fprintf(fout_times, "%s: %20.14e, \n","\"Eval Jacobian\"", t_eval_jacobian);

    //compute Smatrix
    SET_TEAM_VECTOR_SIZE("evalSmatrix")
    timer.reset();
    model.evalSmatrix(team_size, vector_size);
    exec_space_instance.fence();
    const real_type t_eval_smatrix = timer.seconds();
    fprintf(fout_times, "%s: %20.14e, \n","\"Eval S matrix\"", t_eval_smatrix );

    const auto nReactions = model.NumOfReactions();
    // we split the net rate of progress in fwd and rev rate
    // if a reaction is irreversible one rate is set to zero
    const auto nTotalReactions = 2*nReactions;

    // compute rate of progress
    SET_TEAM_VECTOR_SIZE("evalRoP")
    timer.reset();
    model.evalRoP(team_size, vector_size);
    exec_space_instance.fence();
    const real_type t_eval_rop = timer.seconds();
    fprintf(fout_times, "%s: %20.14e, \n","\"Eval rate of progress\"", t_eval_rop);

    std::vector<double> RoP(nTotalReactions); // total

    const int nElem = model.getNumOfElements();



    /*get data from model class to perform csp analysis*/
    real_type_2d_view state_dev;
    timer.reset();
    model.getStateVectorDevice(state_dev);
    exec_space_instance.fence();
    const real_type t_get_state = timer.seconds();
    fprintf(fout_times, "%s: %20.14e, \n","\"Get state vector\"", t_get_state);

    real_type_2d_view rhs;
    timer.reset();
    model.getSourceVectorDevice(rhs);
    exec_space_instance.fence();
    const real_type t_get_source = timer.seconds();
    fprintf(fout_times, "%s: %20.14e, \n","\"Get source term\"", t_get_source);

    const int nSample = state_dev.extent(0);


    real_type_2d_view RoPdb_fwd;
    real_type_2d_view RoPdb_rev;
    timer.reset();
    model.getRoPDevice(RoPdb_fwd, RoPdb_rev);
    exec_space_instance.fence();
    const real_type t_get_rop = timer.seconds();
    fprintf(fout_times, "%s: %20.14e, \n","\"Get rate of progress\"", t_get_rop);

    real_type_3d_view Smatrixdb;
    timer.reset();
    model.getSmatrixDevice(Smatrixdb);
    exec_space_instance.fence();
    const real_type t_get_smatrix = timer.seconds();
    fprintf(fout_times, "%s: %20.14e, \n","\"Get S matrix\"", t_get_smatrix );

    real_type_3d_view jac;
    timer.reset();
    model.getJacMatrixDevice(jac);
    exec_space_instance.fence();
    const real_type t_get_jacobian = timer.seconds();
    fprintf(fout_times, "%s: %20.14e \n","\"Get Jacobian\"", t_get_jacobian);

    fprintf(fout_times, "}, \n ");// end model time

    printf("Number of states %d \n",nSample);

    printf("Working in kernel ...\n");

    if (write_files)
    {
      std::string jac_name = firstname + "_jac.dat";
      FILE *fout_jac = fopen ( (jac_name).c_str(), "w" );

      printf("Writing files ...\n");
      auto jac_host = Kokkos::create_mirror_view(jac);
      Kokkos::deep_copy(jac_host, jac);

      for (size_t i = 0; i < nSample; i++) {

        for (int k = 0; k<ndiff_var; k++ ) {
          for (int j = 0; j<(ndiff_var); j++ ) {
            fprintf(fout_jac,"%15.10e \t", jac_host(i,k,j));
          } //
          fprintf(fout_jac,"\n");
        } // end row

      } // end samples

      fclose(fout_jac);
    }
    fprintf(fout_times, "%s: %d, \n","\"Number of state vectors\"", nSample);

    fprintf(fout_times, " \"Kernel Class\": \n {\n ");

    timer.reset();
    CSPKernelBatch kernelBatch(jac, rhs, state_dev, nElem, csp_rtolvar, csp_atolvar);
    exec_space_instance.fence();
    const real_type t_init_kernel_class = timer.seconds();
    fprintf(fout_times, "%s: %20.14e, \n","\"Init kernel class\"", t_init_kernel_class);
    //
    // SET_TEAM_VECTOR_SIZE("evalEigenSolution")
    // // compute eigensolution and sort eigensolution w.r.t magnitude of eigenvalues
    // timer.reset();
    // kernelBatch.evalEigenSolution(team_size, vector_size);
    // exec_space_instance.fence();
    // const real_type t_comp_eigen_solution = timer.seconds();
    // fprintf(fout_times, "%s: %20.14e, \n","\"Compute eigensolution\"", t_comp_eigen_solution);

    Tines::control_type control;
      /// tpl use
    control["Bool:UseTPL"].bool_value = false;
    control["Bool:SolveEigenvaluesNonSymmetricProblem:Sort"].bool_value = true;
    SET_TEAM_VECTOR_SIZE("Hessenberg")
    /// eigen solve
    if ( team_size > 0 && vector_size > 0) {
      control["IntPair:Hessenberg:TeamSize"].int_pair_value = std::pair<int,int>(team_size,vector_size);
    } // else use tines default values
    SET_TEAM_VECTOR_SIZE("RightEigenvectorSchur")
    if ( team_size > 0 && vector_size > 0) {
          control["IntPair:RightEigenvectorSchur:TeamSize"].int_pair_value = std::pair<int,int>(team_size,vector_size);
    }
    SET_TEAM_VECTOR_SIZE("GemmEigen")
    if ( team_size > 0 && vector_size > 0) {
      control["IntPair:Gemm:TeamSize"].int_pair_value = std::pair<int,int>(team_size,vector_size);
    }
    SET_TEAM_VECTOR_SIZE("SortRightEigenPairs")
    if ( team_size > 0 && vector_size > 0) {
      control["IntPair:SortRightEigenPairs:TeamSize"].int_pair_value = std::pair<int,int>(team_size,vector_size);
    }

    timer.reset();
    kernelBatch.evalEigenSolution(control);
    exec_space_instance.fence();
    const real_type t_comp_eigen_solution = timer.seconds();
    fprintf(fout_times, "%s: %20.14e, \n","\"Compute eigensolution\"", t_comp_eigen_solution);






    // sort eigensolution w.r.t magnitude of eigenvalues
    // SET_TEAM_VECTOR_SIZE("sortEigenSolution")
    // timer.reset();
    // kernelBatch.sortEigenSolution(team_size, vector_size);
    // exec_space_instance.fence();
    // const real_type t_sort_eigen_values_vectors = timer.seconds();
    // fprintf(fout_times, "%s: %20.14e, \n","\"Sort eigensolution\"", t_sort_eigen_values_vectors);

    SET_TEAM_VECTOR_SIZE("evalCSPbasisVectors")
    // Setting CSP vectors:
    timer.reset();
    kernelBatch.evalCSPbasisVectors(team_size, vector_size); // A = eig_vec_R and B = A^{-1}
    exec_space_instance.fence();
    const real_type t_set_csp_vectors = timer.seconds();
    fprintf(fout_times, "%s: %20.14e, \n","\"Set csp vectors\"", t_set_csp_vectors);

    SET_TEAM_VECTOR_SIZE("evalCSP_Pointers")
    timer.reset();
    kernelBatch.evalCSP_Pointers(team_size, vector_size);
    exec_space_instance.fence();
    const real_type t_eval_csp_pointers = timer.seconds();
    fprintf(fout_times, "%s: %20.14e, \n","\"Eval csp pointer\"", t_eval_csp_pointers);

    SET_TEAM_VECTOR_SIZE("evalTimeScales")
    timer.reset();
    kernelBatch.evalTimeScales(team_size, vector_size);
    exec_space_instance.fence();
    const real_type t_eval_tau = timer.seconds();
    fprintf(fout_times, "%s: %20.14e, \n","\"Eval time scales\"", t_eval_tau);

    SET_TEAM_VECTOR_SIZE("evalModalAmp")
    timer.reset();
    kernelBatch.evalModalAmp(team_size, vector_size);
    exec_space_instance.fence();
    const real_type t_eval_mode = timer.seconds();
    fprintf(fout_times, "%s: %20.14e, \n","\"Eval amplitude of modes\"", t_eval_mode);

    SET_TEAM_VECTOR_SIZE("evalM")
    // Exhausted mode
    timer.reset();
    kernelBatch.evalM(team_size, vector_size);
    exec_space_instance.fence();
    const real_type t_eval_m = timer.seconds();

    if (write_files)
    {
      // note that this line does not have a comma.
      fprintf(fout_times, "%s: %20.14e, \n","\"Eval M\"", t_eval_m);
    } else
    {
      fprintf(fout_times, "%s: %20.14e\n","\"Eval M\"", t_eval_m);
    }


    real_type_3d_view_host csp_pointers_host;
    real_type_2d_view_host time_scales_host;
    ordinal_type_1d_view_host M_host;
    real_type_2d_view_host modal_ampl_host;

    if (write_files)
    {
      timer.reset();
      csp_pointers_host  = kernelBatch.getCSPPointers();
      exec_space_instance.fence();
      const real_type t_get_csp_pointers = timer.seconds();
      fprintf(fout_times, "%s: %20.14e, \n","\"Get csp pointer\"", t_get_csp_pointers);

      timer.reset();
      time_scales_host = kernelBatch.getTimeScales();
      exec_space_instance.fence();
      const real_type t_get_tau = timer.seconds();
      fprintf(fout_times, "%s: %20.14e, \n","\"Get time scales\"", t_get_tau);

      timer.reset();
      modal_ampl_host = kernelBatch.getModalAmp();
      exec_space_instance.fence();
      const real_type t_get_mode = timer.seconds();
      fprintf(fout_times, "%s: %20.14e, \n","\"Get amplitude of modes\"", t_get_mode);


      timer.reset();
      M_host = kernelBatch.getM();
      exec_space_instance.fence();
      const real_type t_get_m = timer.seconds();
      fprintf(fout_times, "%s: %20.14e \n","\"Get M\"", t_get_m);
    }

    fprintf(fout_times, "}, \n ");// end kernel time
    ordinal_type_1d_view M = kernelBatch.getMDevice();
    real_type_3d_view B = kernelBatch.getLeftCSPVecDevice();
    real_type_3d_view A = kernelBatch.getRightCSPVecDevice();

    if (compute_csp_indices) {

      fprintf(fout_times, " \"Index Class\": \n {\n ");

      printf("Working in index ...\n");

      timer.reset();
      CSPIndexBatch indexBatch( A, B, Smatrixdb, RoPdb_fwd, RoPdb_rev, M );
      exec_space_instance.fence();
      const real_type t_int_index_class = timer.seconds();
      fprintf(fout_times, "%s: %20.14e, \n","\"Init index class\"", t_int_index_class);

      SET_TEAM_VECTOR_SIZE("evalParticipationIndex")
      timer.reset();
      indexBatch.evalParticipationIndexFwdAndRev(team_size, vector_size);
      exec_space_instance.fence();
      const real_type t_eval_part_indx = timer.seconds();
      fprintf(fout_times, "%s: %20.14e, \n","\"Eval participation index\"", t_eval_part_indx);

      if (!write_files)
        indexBatch.freeParticipationIndexView();

      SET_TEAM_VECTOR_SIZE("evalImportanceIndexSlow")
      timer.reset();
      indexBatch.evalImportanceIndexSlowFwdAndRev(team_size, vector_size);
      exec_space_instance.fence();
      const real_type t_eval_slow_indx = timer.seconds();
      fprintf(fout_times, "%s: %20.14e,\n","\"Eval slow importance index\"", t_eval_slow_indx);

      if (!write_files)
        indexBatch.freeSlowImportanceIndexView();

      SET_TEAM_VECTOR_SIZE("evalImportanceIndexFast")
      timer.reset();
      indexBatch.evalImportanceIndexFastFwdAndRev(team_size, vector_size);
      exec_space_instance.fence();
      const real_type t_eval_fast_indx = timer.seconds();
      if (write_files)
      {
        // note that this lines has a comma!
        fprintf(fout_times, "%s: %20.14e,\n","\"Eval fast importance index\"", t_eval_fast_indx);
      } else
      {
        fprintf(fout_times, "%s: %20.14e\n","\"Eval fast importance index\"", t_eval_fast_indx);
      }
      if (!write_files)
        indexBatch.freeFastImportantIndexView();

      if (write_files)
      {
        real_type_3d_view_host SlowImpoIndex;
        real_type_3d_view_host FastImpoIndex;
        real_type_3d_view_host PartIndex;

        timer.reset();
        SlowImpoIndex =  indexBatch.getImportanceIndexSlow();
        exec_space_instance.fence();
        const real_type t_get_slow_indx = timer.seconds();
        fprintf(fout_times, "%s: %20.14e, \n","\"Get slow importance index\"", t_get_slow_indx);

        timer.reset();
        PartIndex = indexBatch.getParticipationIndex();
        exec_space_instance.fence();
        const real_type t_get_part_indx = timer.seconds();
        fprintf(fout_times, "%s: %20.14e, \n","\"Get participation index\"", t_get_part_indx);

        timer.reset();
        FastImpoIndex = indexBatch.getImportanceIndexFast();
        exec_space_instance.fence();
        const real_type t_get_fast_indx = timer.seconds();
        fprintf(fout_times, "%s: %20.14e \n","\"Get fast importance index\"", t_get_fast_indx);

        // index class
        std::string P_ik_name = firstname + "_ParticipationIndex.dat";
        std::string Islow_jk_name = firstname + "_SlowImportanceIndex.dat";
        std::string Ifast_jk_name = firstname + "_FastImportanceIndex.dat";

        FILE *fout_Pim = fopen ( (P_ik_name).c_str(), "w" );
        FILE *fout_Isi = fopen ( (Islow_jk_name).c_str(), "w" );
        FILE *fout_Ifn = fopen ( (Ifast_jk_name).c_str(), "w" );

        for (size_t i = 0; i < nSample; i++) {
          //Participation index
          for (int k = 0; k<ndiff_var; k++ ) {
            for (int j = 0; j<(nTotalReactions); j++ ) {
              fprintf(fout_Pim,"%15.10e \t", PartIndex(i,k,j));
             }
            fprintf(fout_Pim,"\n");
          }
          //
          // Slow importance index
          for (int k = 0; k<ndiff_var; k++ ) {
            for (int j = 0; j<(nTotalReactions); j++ ) {
              fprintf(fout_Isi,"%15.10e \t", SlowImpoIndex(i,k,j));
            }
          fprintf(fout_Isi,"\n");
          }

          //
          //Fast importance index
          for (int k = 0; k<ndiff_var; k++ ) {
            for (int j = 0; j<(nTotalReactions); j++ ) {
              fprintf(fout_Ifn,"%15.10e \t", FastImpoIndex(i,k,j));
            }
            fprintf(fout_Ifn,"\n");
          }
        }

        fclose(fout_Pim);
        fclose(fout_Isi);
        fclose(fout_Ifn);

      }

      fprintf(fout_times, "}, \n ");// end index time

    }

    if (write_files)
    {
      // kernel class
      std::string m_file_name = firstname + "_m.dat";
      std::string tau_file_name = firstname + "_tau.dat";
      std::string f_file_name = firstname + "_f.dat";

      FILE *fout = fopen ( (m_file_name).c_str(), "w" );
      FILE *fout_tau = fopen ( (tau_file_name).c_str(), "w" );
      FILE *fout_f = fopen ( (f_file_name).c_str(), "w" );

      // std::string num_rank_file_name = firstname + "_jac_numerical_rank.dat";
      // FILE *fout_num_rank = fopen ( (num_rank_file_name).c_str(), "w" );
      // model class

      std::string RoP_name = firstname + "_RoP.dat";
      std::string source_name = firstname + "_source.dat";
      std::string Smatrix_name = firstname + "_Smatrix.dat";
      std::string state_name = firstname + "_state.dat";

      FILE *fout_smatrix = fopen ( (Smatrix_name).c_str(), "w" );
      FILE *fout_RoP = fopen ( (RoP_name).c_str(), "w" );
      FILE *fout_source = fopen ( (source_name).c_str(), "w" );
      // FILE *fout_state_name = fopen ( (state_name).c_str(), "w" );

      std::string eig_val_real_file_name = firstname + "_eig_val_real.dat";
      std::string eig_val_imag_file_name = firstname + "_eig_val_imag.dat";
      FILE *fout_eig_val_real = fopen ( (eig_val_real_file_name).c_str(), "w" );
      FILE *fout_eig_val_imag = fopen ( (eig_val_imag_file_name).c_str(), "w" );

      std::string eig_vec_R_file_name = firstname + "_eig_vec_R.dat";
      FILE *fout_eig_vec_R = fopen ( (eig_vec_R_file_name).c_str(), "w" );

      std::string cspp_ij_name = firstname + "_cspPointers.dat";
      FILE *fout_cspP = fopen ( (cspp_ij_name).c_str(), "w" );

      real_type_3d_view_host A_host("A host", nSample, ndiff_var, ndiff_var );
      Kokkos::deep_copy( A_host, A);
      real_type_2d_view_host eigenvalues_real_part_host = kernelBatch.getEigenValuesRealPart();
      real_type_2d_view_host eigenvalues_imag_part_host = kernelBatch.getEigenValuesImagPart();

      printf("Writing files ...\n");

      // rate of progress
      auto RoPdb_fwd_host = Kokkos::create_mirror_view(RoPdb_fwd);
      Kokkos::deep_copy(RoPdb_fwd_host, RoPdb_fwd);

      auto RoPdb_rev_host = Kokkos::create_mirror_view(RoPdb_rev);
      Kokkos::deep_copy(RoPdb_rev_host, RoPdb_rev);

      auto Smatrix_host = Kokkos::create_mirror_view(Smatrixdb);
      Kokkos::deep_copy(Smatrix_host, Smatrixdb);

      auto rhs_host = Kokkos::create_mirror_view(rhs);
      Kokkos::deep_copy(rhs_host, rhs);

      // auto state_dev_host = Kokkos::create_mirror_view(state_dev);
      // Kokkos::deep_copy(state_dev_host, state_dev);

      for (size_t i = 0; i < nSample; i++) {
        // s matrix
        for (int k = 0; k<Smatrix_host.extent(1); k++ ) {
          for (int j = 0; j<Smatrix_host.extent(2); j++ ) {
            fprintf(fout_smatrix,"%15.10e \t", Smatrix_host(i,k,j));
          }
          fprintf(fout_smatrix,"\n");
        }
        // source
        for (int k = 0; k<ndiff_var; k++ )
          fprintf(fout_source,"%20.14e \t", rhs_host(i,k) );
        fprintf(fout_source,"\n");
        //state
        // for (int k = 0; k<ndiff_var; k++ )
        //   fprintf(fout_state_name,"%20.14e \t", state_dev_host(i,k) );
        // fprintf(fout_state_name,"\n");

        // eig_val_real
        for (int k = 0; k<ndiff_var; k++ )
          fprintf(fout_eig_val_real,"%20.14e \t", eigenvalues_real_part_host(i,k) );
        fprintf(fout_eig_val_real,"\n");

        // eig_val_imag
        for (int k = 0; k<ndiff_var; k++ )
          fprintf(fout_eig_val_imag,"%20.14e \t", eigenvalues_imag_part_host(i,k));
        fprintf(fout_eig_val_imag,"\n");

        // eigenvector rigth
        for (int k = 0; k<ndiff_var; k++ ) {
          for (int j = 0; j<(ndiff_var); j++ ) {
            fprintf(fout_eig_vec_R,"%15.10e \t", A_host(i,k,j));
          }
          fprintf(fout_eig_vec_R,"\n");
        }
         //M
        fprintf(fout," %d \n", M_host(i));

        for (int j = 0; j<(nTotalReactions/2); j++ )
          fprintf(fout_RoP,"%15.10e \t", RoPdb_fwd_host(i,j));
        for (int j = 0; j<(nTotalReactions/2); j++ )
          fprintf(fout_RoP,"%15.10e \t", RoPdb_rev_host(i,j));
        fprintf(fout_RoP,"\n");

        //tau
        for (int k = 0; k<ndiff_var; k++ ) {
                fprintf(fout_tau,"%20.14e \t", time_scales_host(i,k));
        }
        fprintf(fout_tau,"\n");

        // f
        for (int k = 0; k<ndiff_var; k++ ) {
                fprintf(fout_f,"%20.14e \t", modal_ampl_host(i,k));
        }
        fprintf(fout_f,"\n");

        // csp pointer
        for (int k = 0; k<ndiff_var; k++ ) {
          for (int j = 0; j<(ndiff_var); j++ ) {
            fprintf(fout_cspP,"%15.10e \t", csp_pointers_host(i,k,j));
          }
          fprintf(fout_cspP,"\n");
        }


      }

      // model class
      fclose(fout_RoP);
      fclose(fout_source);
      fclose(fout_smatrix);
      // fclose(fout_state_name);

      fclose(fout_eig_vec_R);
      fclose(fout_eig_val_imag);
      fclose(fout_eig_val_real);

      // index class
      // file with whole data base
      fclose(fout_cspP);

      // kernel class
      fclose(fout);
      fclose(fout_f);
      fclose(fout_tau);
      // fclose(fout_num_rank);

    }

    fprintf(fout_times, "%s: %20.14e \n ","\"dummy\"", 0);
    fprintf(fout_times, "} \n ");// end file
    fclose(fout_times);

    }


  }
  printf("Done ... \n" );


  return 0;
}
