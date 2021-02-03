/* =====================================================================================
CSPlib version 1.0
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
#include "CSPlib_CommandLineParser.hpp"
#include "chem_elem_ODE_TChem.hpp"

#include <sys/stat.h>

int main(int argc, char *argv[]) {

  // reactions mechanism and therm files
  std::string prefixPath("../../example/runs/GRI3/data/");
  std::string firstname("");
  std::string jacobianFileName("None");
  double csp_rtolvar(1.e-2); //1.e-7; // 1.e+3; //
  double csp_atolvar(1.e-8); //1.e-8; // 1.e+3; //
  int useNumJac(0);

  std::string chemFile(prefixPath + "chem.inp");
  std::string thermFile(prefixPath + "therm.dat");
  //pfr inputs
  std::string inputFile(prefixPath + "input.dat");

  CSP::CommandLineParser opts("This example computes reaction rates with a given state vector");
  opts.set_option<std::string>("prefix", "prefix to save output files e.g., pos_ ", &firstname);
  opts.set_option<std::string>
  ("chemfile", "Chem file name e.g., chem.inp",
  &chemFile);
  opts.set_option<std::string>
  ("thermfile", "Therm file name e.g., therm.dat", &thermFile);
  opts.set_option<std::string>
  ("inputfile", "data base file name e.g., input.dat", &inputFile);
  opts.set_option<double>("rtol", "relative tolerance for csp analysis e.g., 1e-2 ", &csp_rtolvar);
  opts.set_option<double>("atol", "absolute tolerance for csp analysis e.g., 1e-8 ", &csp_atolvar);
  opts.set_option<std::string>("jacobianFileName", "save jacobian matrix in binary format ", &jacobianFileName);
  opts.set_option<int>("useNumJac", "Use numerical jacobian if 0 analitical jacobian, 1 analyical jacbobian is used only if it avaible ", &useNumJac);

  const bool r_parse = opts.parse(argc, argv);
  if (r_parse) return 0; // print help return

  const char *ch = &(firstname.back());
  const char t  ='/';
  if (strcmp(&t,ch)) {
    struct stat info;
    if(stat( firstname.c_str(), &info ) != 0)
    {
        std::cout <<"Error:: Ouput directory " << firstname << " does not exists\n";
        exit(1);
    }
  }

  {
    CSP::ScopeGuard guard(argc, argv);

    // run on device with not template class with both device and host interface
    {

      //read a solution from TChem, eval source term, jacobian, Smatrix and rate of progress
      std::vector<std::string> var_names;

      ChemElemODETChem model(chemFile, thermFile);
      model.readIgnitionZeroDDataBaseFromFile(inputFile, var_names);
      model.evalSourceVector();
      model.evalJacMatrix(useNumJac);
      model.evalSmatrix();
      model.evalRoP();

      auto ndiff_var = model.getNumOfVariables();
      const auto nReactions = model.NumOfReactions();
      const auto nTotalReactions = 2*nReactions;

      if (jacobianFileName != "None")
      {
        std::string filenamejac = jacobianFileName+"_"+std::to_string(ndiff_var)+".dat";
        printf("Saving Jacobian Matrix in binary format: file name %s\n", filenamejac.c_str());
        Tines::writeView(filenamejac, model._jac_host);
      }

      std::vector< std::vector< double> > state_db;
      std::vector< std::vector< double> > source_db;
      std::vector< std::vector< std::vector< double> > > jac_db;
      std::vector< std::vector< std::vector< double> > > Smatrixdb;
      std::vector< std::vector< double> > RoP_db;

      // get data on the host spaces
      model.getStateVector(state_db);
      model.getSourceVector(source_db);
      model.getJacMatrix(jac_db);
      model.getSmatrix(Smatrixdb);
      model.getRoP(RoP_db);

      const int nSample = state_db.size();
      printf("Number of samples %d \n",nSample);

      std::vector<double> source(ndiff_var);
      std::vector<double> state(ndiff_var);
      std::vector<std::vector<double>> jac
      (ndiff_var,std::vector<double>(ndiff_var, 0.0) );
      std::vector<std::vector<double>>
      Smat(ndiff_var, std::vector<double>(nTotalReactions,0.0) );
      std::vector<double> RoP(nTotalReactions); // total

      // name file to save data
      std::string source_file_name = firstname + "_source.dat";
      std::string state_file_name = firstname + "_state.dat";
      std::string jac_name = firstname + "_jac.dat";
      std::string smat_name = firstname + "_Smat.dat";
      std::string RoP_name = firstname + "_RoP.dat";

      FILE *fout_source = fopen ( (source_file_name).c_str(), "w" );
      FILE *fout_state = fopen ( (state_file_name).c_str(), "w" );
      FILE *fout_jac = fopen ( (jac_name).c_str(), "w" );
      FILE *fout_smat = fopen ( (smat_name).c_str(), "w" );
      FILE *fout_RoP = fopen ( (RoP_name).c_str(), "w" );

      // explore the data base
      for (int i = 0; i < nSample; i++) {
        source = source_db[i];
        state  = state_db[i];
        jac    = jac_db[i];
        Smat   = Smatrixdb[i];
        RoP    = RoP_db[i];

        // state and source
        for (int k = 0; k< ndiff_var; k++ ) {
          fprintf(fout_source,"%20.24e \t", source[k]);
          fprintf(fout_state,"%20.24e \t", state[k]);
        }
        fprintf(fout_source,"\n");
        fprintf(fout_state,"\n");

        // jacobian
        for (int k = 0; k<ndiff_var; k++ ) {
          for (int j = 0; j<(ndiff_var); j++ ) {
              fprintf(fout_jac,"%20.24e \t", jac[k][j]);
          }
          fprintf(fout_jac,"\n");
        }

        //smatrix
        for (int k = 0; k<ndiff_var; k++ ) {
          for (int j = 0; j<(nTotalReactions); j++ ) {
            fprintf(fout_smat,"%20.24e \t", Smat[k][j]);
          }
          fprintf(fout_smat,"\n");
        }

        //RoP
        for (int j = 0; j<(nTotalReactions); j++ ) {
            fprintf(fout_RoP,"%20.24e \t", RoP[j]);
        }
        fprintf(fout_RoP,"\n");


      }

      fclose(fout_source);
      fclose(fout_state);
      fclose(fout_jac);
      fclose(fout_smat);
      fclose(fout_RoP);
    }

    //
    {

      //read a solution from TChem, eval source term, jacobian, Smatrix and rate of progress
      std::vector<std::string> var_names;

      ChemElemODETChem model(chemFile, thermFile);
      model.run_on_host(true);
      model.readIgnitionZeroDDataBaseFromFile(inputFile, var_names);
      model.evalSourceVector();
      model.evalJacMatrix(useNumJac);
      model.evalSmatrix();
      model.evalRoP();

      auto ndiff_var = model.getNumOfVariables();
      const auto nReactions = model.NumOfReactions();
      const auto nTotalReactions = 2*nReactions;

      std::vector< std::vector< double> > state_db;
      std::vector< std::vector< double> > source_db;
      std::vector< std::vector< std::vector< double> > > jac_db;
      std::vector< std::vector< std::vector< double> > > Smatrixdb;
      std::vector< std::vector< double> > RoP_db;

      // get data on the host spaces
      model.getStateVector(state_db);
      model.getSourceVector(source_db);
      model.getJacMatrix(jac_db);
      model.getSmatrix(Smatrixdb);
      model.getRoP(RoP_db);

      const int nSample = state_db.size();
      printf("Number of samples %d \n",nSample);

      std::vector<double> source(ndiff_var);
      std::vector<double> state(ndiff_var);
      std::vector<std::vector<double>> jac
      (ndiff_var,std::vector<double>(ndiff_var, 0.0) );
      std::vector<std::vector<double>>
      Smat(ndiff_var, std::vector<double>(nTotalReactions,0.0) );
      std::vector<double> RoP(nTotalReactions); // total

      // name file to save data
      std::string source_file_name = firstname + "_sourceHost.dat";
      std::string state_file_name = firstname + "_stateHost.dat";
      std::string jac_name = firstname + "_jacHost.dat";
      std::string smat_name = firstname + "_SmatHost.dat";
      std::string RoP_name = firstname + "_RoPHost.dat";

      FILE *fout_source = fopen ( (source_file_name).c_str(), "w" );
      FILE *fout_state = fopen ( (state_file_name).c_str(), "w" );
      FILE *fout_jac = fopen ( (jac_name).c_str(), "w" );
      FILE *fout_smat = fopen ( (smat_name).c_str(), "w" );
      FILE *fout_RoP = fopen ( (RoP_name).c_str(), "w" );

      // explore the data base
      for (int i = 0; i < nSample; i++) {
        source = source_db[i];
        state  = state_db[i];
        jac    = jac_db[i];
        Smat   = Smatrixdb[i];
        RoP    = RoP_db[i];

        // state and source
        for (int k = 0; k< ndiff_var; k++ ) {
          fprintf(fout_source,"%20.24e \t", source[k]);
          fprintf(fout_state,"%20.24e \t", state[k]);
        }
        fprintf(fout_source,"\n");
        fprintf(fout_state,"\n");

        // jacobian
        for (int k = 0; k<ndiff_var; k++ ) {
          for (int j = 0; j<(ndiff_var); j++ ) {
              fprintf(fout_jac,"%20.24e \t", jac[k][j]);
          }
          fprintf(fout_jac,"\n");
        }

        //smatrix
        for (int k = 0; k<ndiff_var; k++ ) {
          for (int j = 0; j<(nTotalReactions); j++ ) {
            fprintf(fout_smat,"%20.24e \t", Smat[k][j]);
          }
          fprintf(fout_smat,"\n");
        }

        //RoP
        for (int j = 0; j<(nTotalReactions); j++ ) {
            fprintf(fout_RoP,"%20.24e \t", RoP[j]);
        }
        fprintf(fout_RoP,"\n");


      }

      fclose(fout_source);
      fclose(fout_state);
      fclose(fout_jac);
      fclose(fout_smat);
      fclose(fout_RoP);
    }



  }
  printf("Done ... \n" );


  return 0;
}
