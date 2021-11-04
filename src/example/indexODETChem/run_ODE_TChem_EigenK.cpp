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

#include "vio.h"
typedef Kokkos::DefaultExecutionSpace SpT;
typedef Kokkos::DefaultHostExecutionSpace HpT;

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
  bool useTChemSolution(true);
  bool verbose(false);
  int increment(1);

  std::string variable1("Temperature");
  std::string variable2("CH4");


  CSP::CommandLineParser opts("This example carries out a csp analysis with TChem model class");
  opts.set_option<std::string>("prefix", "prefix to save output files e.g., pos_ ", &firstname);
  opts.set_option<double>("rtol", "relative tolerance for csp analysis e.g., 1e-2 ", &csp_rtolvar);
  opts.set_option<double>("atol", "absolute tolerance for csp analysis e.g., 1e-8 ", &csp_atolvar);
  opts.set_option<std::string>
  ("chemfile", "Chem file name e.g., chem.inp",
  &chemFile);
  opts.set_option<std::string>
  ("thermfile", "Therm file name e.g., therm.dat", &thermFile);
  opts.set_option<std::string>
  ("inputfile", "database file name e.g., input.dat", &inputFile);
  opts.set_option<bool>
  ("useTChemSolution", "Use a solution produced by TChem e.g., true", &useTChemSolution);
  opts.set_option<bool>(
      "verbose", "If true, printout state vector, jac ...", &verbose);
  opts.set_option<int>("increment", " increment in database, e.g. 2  ", &increment);
  opts.set_option<std::string>
  ("variable1", "Compute Importance index for variable  e.g., Temperature", &variable1);
  opts.set_option<std::string>
  ("variable2", "Compute Importance index for variable  e.g., CH4", &variable2);

  const bool r_parse = opts.parse(argc, argv);
  if (r_parse) return 0; // print help return

  {

    CSP::ScopeGuard guard(argc, argv);


    Kokkos::Timer timer;

    timer.reset();

    ChemElemODETChem  model(chemFile, thermFile);

    real_type t_read_state(0);

    if (useTChemSolution) {
      // read a data base from a TChem++ Ingition solution
      std::vector<std::string> var_names;
      model.readIgnitionZeroDDataBaseFromFile(inputFile, var_names, increment);
      t_read_state = timer.seconds();

    } else{
      // read a data base that was not produced by TChem
      std::vector<std::vector <double> > state_db_read;
      // Density, pressure, temperature and species mass fraction
      const int numofStateVariables = 3 + model.NumOfSpecies();
      readDataBase(inputFile, state_db_read, numofStateVariables );
      model.setStateVectorDB(state_db_read);
    }

    // get number of variables in the ODE system
    auto ndiff_var = model.getNumOfVariables();
    std::cout<< "ndiff_var = "<< ndiff_var <<"\n";

    // get name of species
    std::vector<std::string> spec_name;
    model.getSpeciesNames(spec_name);

    std::vector<double> state(ndiff_var);

    //save species names in a file to be use in post-processing
    std::string species_name = "speciesNames.dat";
    FILE *fout = fopen (  (species_name).c_str() , "w" );
    {
      for (int i = 0; i<spec_name.size(); i++ )
          fprintf(fout,"%s \n", (spec_name[i]).c_str());
      fclose(fout);
    }

    //computes RHS
    timer.reset();
    model.evalSourceVector();
    Kokkos::fence();
    const real_type t_eval_source = timer.seconds();
    std::vector<double> source(ndiff_var);

    //computes jacobian
    timer.reset();
    model.evalJacMatrix(0);
    Kokkos::fence();
    const real_type t_eval_jacobian = timer.seconds();
    std::vector<std::vector<double>> jac (ndiff_var,std::vector<double>(ndiff_var, 0.0) );

    //compute Smatrix
    timer.reset();
    model.evalSmatrix();
    Kokkos::fence();
    const real_type t_eval_smatrix = timer.seconds();

    const auto nReactions = model.NumOfReactions();
    // we split the net rate of progress in fwd and rev rate
    // if a reaction is irreversible one rate is set to zero
    const auto nTotalReactions = 2*nReactions;

    std::vector<std::vector<double>>
    Smat(ndiff_var, std::vector<double>(nTotalReactions,0.0) );

    // compute rate of progress
    timer.reset();
    model.evalRoP();
    Kokkos::fence();
    const real_type t_eval_rop = timer.seconds();

    std::vector<double> RoP(nTotalReactions); // total

    const int nElem = model.getNumOfElements();

    /*get data from model class to perform csp analysis*/
    std::vector< std::vector< double> > state_db;
    timer.reset();
    model.getStateVector(state_db);
    Kokkos::fence();
    const real_type t_get_state = timer.seconds();

    std::vector< std::vector< double> > source_db;
    timer.reset();
    model.getSourceVector(source_db);
    Kokkos::fence();
    const real_type t_get_source = timer.seconds();


    std::vector< std::vector< std::vector< double> > > jac_db;
    timer.reset();
    model.getJacMatrix(jac_db);
    Kokkos::fence();
    const real_type t_get_jacobian = timer.seconds();

    std::vector< std::vector< double> > RoP_db;
    timer.reset();
    model.getRoP(RoP_db);
    Kokkos::fence();
    const real_type t_get_rop = timer.seconds();

    std::vector< std::vector< std::vector< double> > > Smatrixdb;
    timer.reset();
    model.getSmatrix(Smatrixdb);
    Kokkos::fence();
    const real_type t_get_smatrix = timer.seconds();


    std::vector< std::vector< double> >  eig_val_real_bath;
    std::vector< std::vector< double> >  eig_val_imag_bath;
    std::vector< std::vector< std::vector< double> > > eig_vec_R_bath;

    timer.reset();
    EigenSolver::evalDevice(model._jac,
                            eig_val_real_bath,
                            eig_val_imag_bath,
                            eig_vec_R_bath);

    //
    Kokkos::fence();
    const real_type t_eval_and_get_eigen_solution = timer.seconds();


    // EigenSolver::evalHost(model._jac_host,
    //                       eig_val_real_bath,
    //                       eig_val_imag_bath,
    //                       eig_vec_R_bath);


    std::vector<double> eig_val_real;
    std::vector<double> eig_val_imag;
    std::vector<double> eig_vec_R(ndiff_var*ndiff_var);

    std::vector< std::vector< double> > eig_vec_R_2D;

    std::vector<double> csp_vec_R(ndiff_var*ndiff_var);
    std::vector<double> csp_vec_L(ndiff_var*ndiff_var);
    std::vector<double> tau_vec;
    std::vector<double> f_vec;

    std::vector<std::vector<double> > csp_vec_R_2d, csp_vec_L_2d;

    std::vector<std::vector<double> > P_ik    ;
    std::vector<std::vector<double> > Islow_jk;
    std::vector<std::vector<double> > Ifast_jk;
    std::vector<std::vector<double> > cspp_ij;

    // model class
    std::string jac_name = firstname + "_jac.dat";
    std::string RoP_name = firstname + "_RoP.dat";
    std::string source_name = firstname + "_source.dat";

    FILE *fout_jac = fopen ( (jac_name).c_str(), "w" );
    FILE *fout_RoP = fopen ( (RoP_name).c_str(), "w" );
    FILE *fout_source = fopen ( (source_name).c_str(), "w" );

    // kernel class
    std::string m_file_name = firstname + "_m.dat";
    std::string tau_file_name = firstname + "_tau.dat";
    std::string f_file_name = firstname + "_f.dat";

    fout = fopen ( (m_file_name).c_str(), "w" );
    FILE *fout_tau = fopen ( (tau_file_name).c_str(), "w" );
    FILE *fout_f = fopen ( (f_file_name).c_str(), "w" );

    std::string num_rank_file_name = firstname + "_jac_numerical_rank.dat";
    FILE *fout_num_rank = fopen ( (num_rank_file_name).c_str(), "w" );

    std::string eig_val_real_file_name = firstname + "_eig_val_real.dat";
    std::string eig_val_imag_file_name = firstname + "_eig_val_imag.dat";
    FILE *fout_eig_val_real = fopen ( (eig_val_real_file_name).c_str(), "w" );
    FILE *fout_eig_val_imag = fopen ( (eig_val_imag_file_name).c_str(), "w" );

    std::string eig_vec_R_file_name = firstname + "_eig_vec_R.dat";
    FILE *fout_eig_vec_R = fopen ( (eig_vec_R_file_name).c_str(), "w" );


    // index class
    std::string P_ik_name = firstname + "_ParticipationIndex.dat";
    std::string Islow_jk_name = firstname + "_SlowImportanceIndex.dat";
    std::string Ifast_jk_name = firstname + "_FastImportanceIndex.dat";
    std::string cspp_ij_name = firstname + "_cspPointers.dat";


    FILE *fout_Pim = fopen ( (P_ik_name).c_str(), "w" );
    FILE *fout_Isi = fopen ( (Islow_jk_name).c_str(), "w" );
    FILE *fout_Ifn = fopen ( (Ifast_jk_name).c_str(), "w" );
    FILE *fout_cspP = fopen ( (cspp_ij_name).c_str(), "w" );


    /*  get top rop from fast, slow, imporatnce indexes */
    const int Top_rop(2);
    const double threshold_rop(1e-2);

    /* csp pointer for one or few modes*/
    std::string Top_cssp_var_file_name = firstname + "_Mode0_cspPointersTopElemPosition.dat";
    FILE *fout_Top_var_cspp = fopen ( (Top_cssp_var_file_name).c_str(), "w" );
    std::vector<int> IndxListCSPP;


    std::vector<double> cspp_k0;
    std::string mode0_cspp_file_name = firstname + "_Mode0_cspPointers.dat";
    FILE *fout_cspp_mode0 = fopen ( (mode0_cspp_file_name).c_str(), "w" );

    /*Participation index  */
    std::string Top_part_rop_file_name = firstname + "_Mode0_ParticipationIndexTopElemPosition.dat";
    FILE *fout_Top_rop_part = fopen ( (Top_part_rop_file_name).c_str(), "w" );
    std::vector<int> IndxListPart;

    std::vector<double> P_k;
    std::string mode0_file_name = firstname + "_Mode0_ParticipationIndex.dat";
    FILE *fout_mode0 = fopen ( (mode0_file_name).c_str(), "w" );

    /* Temperature */
    /* Slow importance index  */
    const int indxTemp = model.getVarIndex(variable1) ;
    std::string Top_rop_file_name = firstname + "_" +variable1+"_SlowImportanceIndexTopElemPosition.dat";
    FILE *fout_Top_rop = fopen ( (Top_rop_file_name).c_str(), "w" );
    std::vector<int> IndxList;

    std::vector<double> Islow_k;
    std::string SlowIndVar0_file_name = firstname + "_" + variable1+ "_SlowImportanceIndex.dat";
    FILE *fout_SlowIndVar0 = fopen ( (SlowIndVar0_file_name).c_str(), "w" );

    /*Fast importance index  */
    std::string Top_fast_rop_file_name = firstname + "_" +variable1+ "_FastImportanceIndexTopElemPosition.dat";
    FILE *fout_Top_rop_fast = fopen ( (Top_fast_rop_file_name).c_str(), "w" );
    std::vector<int> IndxListFast;

    std::vector<double> Ifast_k;
    std::string FastIndVar0_file_name = firstname +"_" + variable1+  "_FastImportanceIndex.dat";
    FILE *fout_FastIndVar0 = fopen ( (FastIndVar0_file_name).c_str(), "w" );

    /* Slow importance index  */
    const int indxCH4 = model.getVarIndex(variable2) ;
    std::string ch4_Slow_Top_rop_file_name = firstname + "_" +variable2+"_SlowImportanceIndexTopElemPosition.dat";
    FILE *fout_Top_rop_ch4 = fopen ( (ch4_Slow_Top_rop_file_name).c_str(), "w" );
    std::vector<int> IndxListch4;

    std::vector<double> Islow_k_ch4;
    std::string Slowch4_file_name = firstname + "_" + variable2+ "_SlowImportanceIndex.dat";
    FILE *fout_SlowIndch4 = fopen ( (Slowch4_file_name).c_str(), "w" );

    /*Fast importance index  */
    std::string ch4_Top_fast_rop_file_name = firstname + "_" +variable2+ "_FastImportanceIndexTopElemPosition.dat";
    FILE *fout_Top_rop_fast_ch4 = fopen ( (ch4_Top_fast_rop_file_name).c_str(), "w" );
    std::vector<int> IndxListFastch4;

    std::vector<double> Ifast_k_ch4;
    std::string Fastch4_file_name =  firstname +"_" + variable2+  "_FastImportanceIndex.dat";
    FILE *fout_FastIndch4 = fopen ( (Fastch4_file_name).c_str(), "w" );

    const int nSample = state_db.size();

    printf("Number of states %d \n",nSample);

    real_type t_init_kernel_class(0);
    real_type t_sort_eigen_values_vectors(0);
    real_type t_set_csp_vectors(0);
    real_type t_get_eval_m(0);
    real_type t_get_eval_tau(0);
    real_type t_get_eval_mode(0);
    real_type t_get_eval_csp_pointers(0);
    real_type t_int_index_class(0);
    real_type t_eval_and_get_part_indx(0);
    real_type t_eval_and_get_fast_indx(0);
    real_type t_eval_and_get_slow_indx(0);
    real_type t_write_files(0);


    CSPIndex idx(nTotalReactions, ndiff_var);
    std::vector<double> Smat1D(ndiff_var * nTotalReactions);


    for (int i = 0; i < nSample; i++) {
      source = source_db[i];
      state  = state_db[i];
      jac    = jac_db[i];
      Smat   = Smatrixdb[i];
      RoP    = RoP_db[i];

      eig_val_real = eig_val_real_bath[i];
      eig_val_imag = eig_val_imag_bath[i];
      eig_vec_R_2D    = eig_vec_R_bath[i];


      // convert 2D TO 1D
      //row-major order
      int count=0;
      for (size_t k=0; k<ndiff_var; k++) {
        for (size_t j=0; j<ndiff_var; j++) {
          eig_vec_R[count] = eig_vec_R_2D[k][j];
          count++;
        }
      }

      int count2=0;
      for (size_t k=0; k<ndiff_var; k++) {
        for (size_t j=0; j<nTotalReactions; j++) {
          Smat1D[count2] = Smat[k][j];
          count2++;
        }
      }

      timer.reset();
      Kernel ker(ndiff_var, state, source, jac);
      Kokkos::fence();
      t_init_kernel_class += timer.seconds();

      if (verbose) {
          // check if state vector was correctly read
          std::cout << "state at : " <<  i << std::endl;
          printf("Temperature %20.14e \n",state[0] );
          for (int k = 0; k < spec_name.size(); k++)
            printf(" %s %20.14e\n",(spec_name[k]).c_str(),state[k+1] );

          // source
          std::cout << "source at : " << i <<std::endl;
          printf("Temperature %20.14e \n",source[0] );
          for (int k = 0; k < spec_name.size(); k++)
            printf(" %s %20.14e\n",(spec_name[k]).c_str(),source[k+1] );

          // jacobian
          Util::Print::mat<double>("jac", RIF, Out2d, Dbl, 3, 3, jac);
          // Smat and RoP
          //compute rhs (source) from Matrix produc of Smat and RoP
          //RHS = Smat * RoP
          std::vector<double> rhsSmaRop(nTotalReactions);
          CSP::MatrixVectorMul(Smat, RoP, rhsSmaRop);
          for (int i = 0; i < ndiff_var; i++) {
              printf("i %d rhs (Smat*RoP) %e source %e diff %e \n", i, rhsSmaRop[i],
              source[i], (rhsSmaRop[i]-source[i])/(source[i]+1e-23) );
          }
     }

      if (verbose) {
        printf("----------- Sample No %d ------------\n", i );
      }

      int jac_rank = ker.computeJacobianNumericalRank();
      fprintf(fout_num_rank," %d \n", jac_rank);


      timer.reset();
      ker.setEigenValVec(eig_val_real, eig_val_imag, eig_vec_R);
      // Sorting eigen values and vectors in ascending order
      // of, sign(eig_val_real)*Mod(eig_val_real + i * eig_val_imag)
      ker.sortEigValVec();
      ker.getEigenValVec(eig_val_real, eig_val_imag, eig_vec_R);
      Kokkos::fence();
      t_sort_eigen_values_vectors += timer.seconds();

      if (verbose) {
        ker.DiagEigValVec();
      }
      // Setting CSP vectors:
      timer.reset();
      ker.setCSPVec(); // A = eig_vec_R and B = A^{-1}
      ker.getCSPVec(csp_vec_L, csp_vec_R);
      Kokkos::fence();
      t_set_csp_vectors += timer.seconds();

      if (verbose) {
       ker.DiagOrthogonalityCSPVec();
      }
      //
      timer.reset();
      ker.evalCSPPointers();
      ker.getCSPPointers( cspp_ij );
      Kokkos::fence();
      t_get_eval_csp_pointers += timer.seconds();

      /*csp pointer  */
      // for mode 0
      ker.evalAndGetCSPPointers(0, cspp_k0);

      //========================================================================================
      // Time scales:

      timer.reset();
      ker.evalTau();
      ker.getTau(tau_vec);
      Kokkos::fence();
      t_get_eval_tau += timer.seconds();

      timer.reset();
      ker.evalModalAmp( );
      ker.getModalAmp( f_vec );
      Kokkos::fence();
      t_get_eval_mode += timer.seconds();

      timer.reset();
      ker.setCSPerr(csp_rtolvar, csp_atolvar);

      // Exhausted mode
      int NofDM = 0;
      ker.evalM(nElem);
      ker.getM(NofDM);
      Kokkos::fence();
      t_get_eval_m += timer.seconds();
      fprintf(fout," %d \n", NofDM);

    // instantiate CSP Index class
      //===================================================================================================================

      // std::cout << "Testing Index class members\n";

      CSP::construct_2D_from_1D(ndiff_var, ndiff_var, csp_vec_R, csp_vec_R_2d);
      CSP::construct_2D_from_1D(ndiff_var, ndiff_var, csp_vec_L, csp_vec_L_2d);

      // instantiate CSP Index class
      timer.reset();
      idx.initChemKinModel( NofDM,
                            eig_val_real,
                            eig_val_imag,
                            csp_vec_R_2d,
                            csp_vec_L_2d,
                            Smat,
                            RoP);

      // CSPIndex idx(nTotalReactions, ndiff_var,
      //              NofDM, eig_val_real, eig_val_imag,
      //              csp_vec_R_2d, csp_vec_L_2d, Smat, RoP );
      //
      Kokkos::fence();
      t_int_index_class += timer.seconds();


      // idx.evalBetaV2(csp_vec_L, Smat1D);

      timer.reset();
      idx.evalParticipationIndex();
      idx.getParticipationIndex ( P_ik     );
      Kokkos::fence();
      t_eval_and_get_part_indx += timer.seconds();

      timer.reset();
      idx.evalImportanceIndexSlow();
      idx.getImportanceIndexSlow( Islow_jk );
      Kokkos::fence();
      t_eval_and_get_slow_indx += timer.seconds();

      timer.reset();
      idx.evalImportanceIndexFast();
      idx.getImportanceIndexFast( Ifast_jk );
      Kokkos::fence();
      t_eval_and_get_fast_indx += timer.seconds();

      /* eval and get participation index for one mode*/
      int modeIndx(0);
      idx.evalAndGetParticipationIndex(modeIndx, P_k);



      /* eval and get slow importance index for one variable */
      idx.evalAndGetImportanceIndexSlow(indxTemp, Islow_k);
      idx.evalAndGetImportanceIndexSlow(indxCH4, Islow_k_ch4);

      /* eval and get fast importance index for one variable */
      idx.evalAndGetImportanceIndexFast(indxTemp, Ifast_k);
      idx.evalAndGetImportanceIndexFast(indxCH4, Ifast_k_ch4);


      /* get top rate of progess */
      idx.getTopIndex(cspp_k0, Top_rop, threshold_rop,  IndxListCSPP );
      // Participation index for mode 0
      idx.getTopIndex(P_k, Top_rop, threshold_rop,  IndxListPart );
      // Temperature slow and fast importance index
      idx.getTopIndex(Islow_k, Top_rop, threshold_rop,  IndxList );
      idx.getTopIndex(Ifast_k, Top_rop, threshold_rop,  IndxListFast );

      idx.getTopIndex(Islow_k_ch4, Top_rop, threshold_rop,  IndxListch4 );
      idx.getTopIndex(Ifast_k_ch4, Top_rop, threshold_rop,  IndxListFastch4 );



      timer.reset();
      // jac
      for (int k = 0; k<ndiff_var; k++ ) {
        for (int j = 0; j<(ndiff_var); j++ ) {
          fprintf(fout_jac,"%15.10e \t", jac[k][j]);
        }
        fprintf(fout_jac,"\n");
      }
      // rate of progress
      for (int j = 0; j<(nTotalReactions); j++ ) {
        fprintf(fout_RoP,"%15.10e \t", RoP[j]);
      }
      fprintf(fout_RoP,"\n");
      //source term
      for (int j = 0; j<(ndiff_var); j++ ) {
          fprintf(fout_source,"%15.10e \t", source[j]);
      }
      fprintf(fout_source,"\n");

      // eigenvector rigth
      for (int k = 0; k<ndiff_var; k++ ) {
        for (int j = 0; j<(ndiff_var); j++ ) {
          fprintf(fout_eig_vec_R,"%15.10e \t", eig_vec_R[k*ndiff_var + j]);
        }
        fprintf(fout_eig_vec_R,"\n");
      }

      // eig_val_real
      for (int k = 0; k<eig_val_real.size(); k++ )
            fprintf(fout_eig_val_real,"%20.14e \t", eig_val_real[k]);
      fprintf(fout_eig_val_real,"\n");


      // eig_val_imag
      for (int k = 0; k<eig_val_real.size(); k++ )
            fprintf(fout_eig_val_imag,"%20.14e \t", eig_val_imag[k]);
      fprintf(fout_eig_val_imag,"\n");


      //tau
      for (int k = 0; k<tau_vec.size(); k++ ) {
            fprintf(fout_tau,"%20.14e \t", tau_vec[k]);
      }
      fprintf(fout_tau,"\n");

      // f
      for (int k = 0; k<tau_vec.size(); k++ ) {
            fprintf(fout_f,"%20.14e \t", f_vec[k]);
      }
      fprintf(fout_f,"\n");

      //Participation index
      for (int k = 0; k<ndiff_var; k++ ) {
        for (int j = 0; j<(nTotalReactions); j++ ) {
          fprintf(fout_Pim,"%15.10e \t", P_ik[k][j]);
        }
        fprintf(fout_Pim,"\n");
      }
      // Slow importance index
      for (int k = 0; k<ndiff_var; k++ ) {
        for (int j = 0; j<(nTotalReactions); j++ ) {
          fprintf(fout_Isi,"%15.10e \t", Islow_jk[k][j]);
        }
        fprintf(fout_Isi,"\n");
      }
      //Fast importance index
      for (int k = 0; k<ndiff_var; k++ ) {
        for (int j = 0; j<(nTotalReactions); j++ ) {
          fprintf(fout_Ifn,"%15.10e \t", Ifast_jk[k][j]);
        }
        fprintf(fout_Ifn,"\n");
      }

    // csp pointer
      for (int k = 0; k<ndiff_var; k++ ) {
        for (int j = 0; j<(ndiff_var); j++ ) {
          fprintf(fout_cspP,"%15.10e \t", cspp_ij[k][j]);
        }
        fprintf(fout_cspP,"\n");
      }
      Kokkos::fence();
      t_write_files += timer.seconds();



      /*csp pointer for one mode  */
      for (int j = 0; j<(ndiff_var); j++ ) {
        fprintf(fout_cspp_mode0,"%15.10e \t", cspp_k0[j]);
      }
      fprintf(fout_cspp_mode0,"\n");


      /* participation index for one mode or variable  */
      for (int j = 0; j<(nTotalReactions); j++ ) {
        fprintf(fout_FastIndVar0,"%15.10e \t", Ifast_k[j]);
      }
      fprintf(fout_FastIndVar0,"\n");

      for (int j = 0; j<(nTotalReactions); j++ ) {
        fprintf(fout_FastIndch4,"%15.10e \t", Ifast_k_ch4[j]);
      }
      fprintf(fout_FastIndch4,"\n");


      for (int j = 0; j<(nTotalReactions); j++ ) {
        fprintf(fout_mode0,"%15.10e \t", P_k[j]);
      }
      fprintf(fout_mode0,"\n");

      for (int j = 0; j<(nTotalReactions); j++ ) {
        fprintf(fout_SlowIndVar0,"%15.10e \t", Islow_k[j]);
      }
      fprintf(fout_SlowIndVar0,"\n");

      for (int j = 0; j<(nTotalReactions); j++ ) {
        fprintf(fout_SlowIndch4,"%15.10e \t", Islow_k_ch4[j]);
      }
      fprintf(fout_SlowIndch4,"\n");


    } //end of samples

    /* save top indeces */

    for (size_t j = 0; j< IndxList.size(); j++ ) {
      fprintf(fout_Top_rop,"%d \t", IndxList[j]);
    }
    fprintf(fout_Top_rop,"\n");

    for (size_t j = 0; j< IndxListPart.size(); j++ ) {
      fprintf(fout_Top_rop_part,"%d \t", IndxListPart[j]);
    }
    fprintf(fout_Top_rop_part,"\n");


    for (size_t j = 0; j< IndxListCSPP.size(); j++ ) {
      fprintf(fout_Top_var_cspp,"%d \t", IndxListCSPP[j]);
    }
    fprintf(fout_Top_var_cspp,"\n");


    for (size_t j = 0; j< IndxListFast.size(); j++ ) {
      fprintf(fout_Top_rop_fast,"%d \t", IndxListFast[j]);
    }
    fprintf(fout_Top_rop_fast,"\n");

    for (size_t j = 0; j< IndxList.size(); j++ ) {
      fprintf(fout_Top_rop_ch4,"%d \t", IndxListch4[j]);
    }
    fprintf(fout_Top_rop_ch4,"\n");

    for (size_t j = 0; j< IndxListFast.size(); j++ ) {
      fprintf(fout_Top_rop_fast_ch4,"%d \t", IndxListFastch4[j]);
    }
    fprintf(fout_Top_rop_fast_ch4,"\n");

    /* close files */
    // file with element position for RoP
    fclose(fout_Top_rop);
    fclose(fout_Top_rop_fast);
    fclose(fout_Top_rop_part);
    fclose(fout_Top_var_cspp);

    fclose(fout_Top_rop_ch4);
    fclose(fout_Top_rop_fast_ch4);

    // model class
    fclose(fout_jac);
    fclose(fout_RoP);
    fclose(fout_source);

    // kernel class
    fclose(fout);
    fclose(fout_f);
    fclose(fout_tau);

    fclose(fout_num_rank);
    fclose(fout_eig_vec_R);
    fclose(fout_eig_val_imag);
    fclose(fout_eig_val_real);

    // index class
    // file with whole data base
    fclose(fout_Pim);
    fclose(fout_Isi);
    fclose(fout_Ifn);
    fclose(fout_cspP);

    // file with mode 0, temperature and ch4
    fclose(fout_mode0);
    fclose(fout_SlowIndVar0);
    fclose(fout_FastIndVar0);
    fclose(fout_FastIndch4);
    fclose(fout_SlowIndch4);
    fclose(fout_cspp_mode0);

    // Kokkos::fence(); /// timing purpose


    // printf(" read state vector       : %e\n", t_read_state);
    // printf(" eval source term        : %e\n", t_eval_source);
    // printf(" eval S matrix           : %e\n", t_eval_smatrix );
    // printf(" eval rate of prog       : %e\n", t_eval_rop);
    // printf(" eval Jacobian           : %e\n", t_eval_jacobian );
    // printf(" get state vector        : %e\n", t_get_state );
    // printf(" get source term         : %e\n", t_get_source);
    // printf(" get Jacobian            : %e\n", t_get_jacobian);
    // printf(" get S matrix            : %e\n", t_get_smatrix);
    // printf(" get rate of prog        : %e\n", t_get_rop);
    // printf(" eigen solver            : %e\n", t_eval_and_get_eigen_solution);
    // printf(" init kernel class       : %e\n", t_init_kernel_class );
    // printf(" sort eigen solution     : %e\n", t_sort_eigen_values_vectors);
    // printf(" set csp vectors         : %e\n", t_set_csp_vectors);
    // printf(" get and eval m          : %e\n", t_get_eval_m);
    // printf(" get and eval tau        : %e\n", t_get_eval_tau);
    // printf(" get and eval modes      : %e\n", t_get_eval_mode);
    // printf(" get and eval csp p      : %e\n", t_get_eval_csp_pointers);
    // printf(" init index class        : %e\n", t_int_index_class);
    // printf(" get and eval part indx  : %e\n", t_eval_and_get_part_indx);
    // printf(" get and eval fast indx  : %e\n", t_eval_and_get_fast_indx);
    // printf(" get and eval slow indx  : %e\n", t_eval_and_get_slow_indx);


    std::string time_file_name = firstname + "_csp_times.dat";

    FILE *fout_times = fopen (time_file_name.c_str(), "w" );

    fprintf(fout_times, "%s, %20.14e \n","read state vector", t_read_state);
    fprintf(fout_times, "%s, %20.14e \n","eval source term", t_eval_source);
    fprintf(fout_times, "%s, %20.14e \n","eval S matrix",t_eval_smatrix );
    fprintf(fout_times, "%s, %20.14e \n","eval rate of progress", t_eval_rop);
    fprintf(fout_times, "%s, %20.14e \n","eval Jacobian", t_eval_jacobian);
    fprintf(fout_times, "%s, %20.14e \n","get state vector", t_get_state);
    fprintf(fout_times, "%s, %20.14e \n","get source term", t_get_source);
    fprintf(fout_times, "%s, %20.14e \n","get Jacobian", t_get_jacobian);
    fprintf(fout_times, "%s, %20.14e \n","get S matrix",t_get_smatrix );
    fprintf(fout_times, "%s, %20.14e \n","get rate of progress", t_get_rop);
    fprintf(fout_times, "%s, %20.14e \n","eigen solver", t_eval_and_get_eigen_solution);
    fprintf(fout_times, "%s, %20.14e \n","init kernel class", t_init_kernel_class);
    fprintf(fout_times, "%s, %20.14e \n","sort eigen solution", t_sort_eigen_values_vectors);
    fprintf(fout_times, "%s, %20.14e \n","set csp vectors", t_set_csp_vectors);
    fprintf(fout_times, "%s, %20.14e \n","get and eval m", t_get_eval_m);
    fprintf(fout_times, "%s, %20.14e \n","get and eval tau", t_get_eval_tau);
    fprintf(fout_times, "%s, %20.14e \n","get and eval amplitude of modes", t_get_eval_mode);
    fprintf(fout_times, "%s, %20.14e \n","get and eval csp pointer", t_get_eval_csp_pointers);
    fprintf(fout_times, "%s, %20.14e \n","init index class", t_int_index_class);
    fprintf(fout_times, "%s, %20.14e \n","get and eval participation index", t_eval_and_get_part_indx);
    fprintf(fout_times, "%s, %20.14e \n","get and eval fast importance index", t_eval_and_get_fast_indx);
    fprintf(fout_times, "%s, %20.14e \n","get and eval slow importance index", t_eval_and_get_slow_indx);
    fprintf(fout_times, "%s, %20.14e \n","write files", t_write_files);

    fclose(fout_times);

  }

  printf("Done ... \n" );


  return 0;
}
