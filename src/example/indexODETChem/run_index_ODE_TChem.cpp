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

void readDataBase(const std::string& filename,
                  std::vector< std::vector<double> >& database,
                  const int &nStateVariables)
{
  double atposition;
  printf("Reading from data base \n ");
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
  int use_analytical_Jacobian(0);

  std::string chemFile(prefixPath + "chem.inp");
  std::string thermFile(prefixPath + "therm.dat");
  std::string inputFile(prefixPath + "input.dat");
  bool useTChemSolution(true);
  bool verbose(false);

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
  //
  opts.set_option<int>
  ("useAnalyticalJacobian",
   "Use a analytical jacobian; 0: hand-derived analytical jacobian, 1: numerical jacobian, other number: sacado Analytical jacobian  ", &use_analytical_Jacobian);
  opts.set_option<bool>(
      "verbose", "If true, printout state vector, jac ...", &verbose);

  const bool r_parse = opts.parse(argc, argv);
  if (r_parse) return 0; // print help return

  {

    CSP::ScopeGuard guard(argc, argv);

    ChemElemODETChem  model(chemFile, thermFile);

    if (useTChemSolution) {
      // read a data base from a TChem++ Ingition solution
      std::vector<std::string> var_names;
      model.readIgnitionZeroDDataBaseFromFile(inputFile,var_names);
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
    model.evalSourceVector();
    std::vector<double> source(ndiff_var);

    //computes jacobian

    model.evalJacMatrix(use_analytical_Jacobian);
    std::vector<std::vector<double>> jac (ndiff_var,std::vector<double>(ndiff_var, 0.0) );

    //compute Smatrix
    model.evalSmatrix();

    const auto nReactions = model.NumOfReactions();
    // we split the net rate of progress in fwd and rev rate
    // if a reaction is irreversible one rate is set to zero
    const auto nTotalReactions = 2*nReactions;

    std::vector<std::vector<double>>
    Smat(ndiff_var, std::vector<double>(nTotalReactions,0.0) );

    // compute rate of progress
    model.evalRoP();

    std::vector<double> RoP(nTotalReactions); // total

    const int nElem = model.getNumOfElements();

    /*get data from model class to perform csp analysis*/
    std::vector< std::vector< double> > state_db;
    model.getStateVector(state_db);

    std::vector< std::vector< double> > source_db;
    model.getSourceVector(source_db);

    std::vector< std::vector< std::vector< double> > > jac_db;
    model.getJacMatrix(jac_db);

    std::vector< std::vector< double> > RoP_db;
    model.getRoP(RoP_db);

    std::vector< std::vector< std::vector< double> > > Smatrixdb;
    model.getSmatrix(Smatrixdb);

    std::vector<double> eig_val_real;
    std::vector<double> eig_val_imag;
    std::vector<double> eig_vec_L;
    std::vector<double> eig_vec_R;

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

    FILE *fout_jac = fopen ( (jac_name).c_str(), "w" );
    FILE *fout_RoP = fopen ( (RoP_name).c_str(), "w" );

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

    std::string eig_vec_L_file_name = firstname + "_eig_vec_L.dat";
    std::string eig_vec_R_file_name = firstname + "_eig_vec_R.dat";
    FILE *fout_eig_vec_L = fopen ( (eig_vec_L_file_name).c_str(), "w" );
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

    std::string var_name = "Temperature";

    /* Slow importance index  */
    const int indxTemp = model.getVarIndex(var_name) ;
    std::string Top_rop_file_name = firstname + "_" +var_name+"_SlowImportanceIndexTopElemPosition.dat";
    FILE *fout_Top_rop = fopen ( (Top_rop_file_name).c_str(), "w" );
    std::vector<int> IndxList;

    std::vector<double> Islow_k;
    std::string SlowIndVar0_file_name = firstname + "_" + var_name+ "_SlowImportanceIndex.dat";
    FILE *fout_SlowIndVar0 = fopen ( (SlowIndVar0_file_name).c_str(), "w" );

    /*Fast importance index  */
    std::string Top_fast_rop_file_name = firstname + "_" +var_name+ "_FastImportanceIndexTopElemPosition.dat";
    FILE *fout_Top_rop_fast = fopen ( (Top_fast_rop_file_name).c_str(), "w" );
    std::vector<int> IndxListFast;

    std::vector<double> Ifast_k;
    std::string FastIndVar0_file_name = firstname +"_" + var_name+  "_FastImportanceIndex.dat";
    FILE *fout_FastIndVar0 = fopen ( (FastIndVar0_file_name).c_str(), "w" );

    var_name = "CH4";

    /* Slow importance index  */
    const int indxCH4 = model.getVarIndex(var_name) ;
    std::string ch4_Slow_Top_rop_file_name = firstname + "_" +var_name+"_SlowImportanceIndexTopElemPosition.dat";
    FILE *fout_Top_rop_ch4 = fopen ( (ch4_Slow_Top_rop_file_name).c_str(), "w" );
    std::vector<int> IndxListch4;

    std::vector<double> Islow_k_ch4;
    std::string Slowch4_file_name = firstname + "_" + var_name+ "_SlowImportanceIndex.dat";
    FILE *fout_SlowIndch4 = fopen ( (Slowch4_file_name).c_str(), "w" );

    /*Fast importance index  */
    std::string ch4_Top_fast_rop_file_name = firstname + "_" +var_name+ "_FastImportanceIndexTopElemPosition.dat";
    FILE *fout_Top_rop_fast_ch4 = fopen ( (ch4_Top_fast_rop_file_name).c_str(), "w" );
    std::vector<int> IndxListFastch4;

    std::vector<double> Ifast_k_ch4;
    std::string Fastch4_file_name =  firstname +"_" + var_name+  "_FastImportanceIndex.dat";
    FILE *fout_FastIndch4 = fopen ( (Fastch4_file_name).c_str(), "w" );

    const int nSample = state_db.size();

    std::string csp_pointer_fast_space_name = firstname + "_cspPointersFastSubSpace.dat";
    std::vector<double> csp_pointer_fast_space(ndiff_var,0.0);
    FILE *fout_cspP_fast_subspace = fopen ( (csp_pointer_fast_space_name).c_str(), "w" );

    printf("Number of states %d \n",nSample);


    for (int i = 0; i < nSample; i++) {
      source = source_db[i];
      state  = state_db[i];
      jac    = jac_db[i];
      Smat   = Smatrixdb[i];
      RoP    = RoP_db[i];

      Kernel ker(ndiff_var, state, source, jac);

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

      printf("----------- Sample No %d ------------\n", i );

      int jac_rank = ker.computeJacobianNumericalRank();
      fprintf(fout_num_rank," %d \n", jac_rank);

      // Eigen solution:
      ker.evalEigenValVec();
      // Sorting eigen values and vectors in ascending order
      // of, sign(eig_val_real)*Mod(eig_val_real + i * eig_val_imag)
      ker.sortEigValVec();
      ker.getEigenValVec(eig_val_real, eig_val_imag, eig_vec_L, eig_vec_R);

      ker.DiagEigValVec();

      // Setting CSP vectors:
      ker.setCSPVec(); // A = eig_vec_R and B = A^{-1}
      ker.getCSPVec(csp_vec_L, csp_vec_R);

      ker.DiagOrthogonalityCSPVec();

      //
      ker.evalCSPPointers();
      ker.getCSPPointers( cspp_ij );

      /*csp pointer  */
      // for mode 0
      ker.evalAndGetCSPPointers(0, cspp_k0);

      //========================================================================================
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
      fprintf(fout," %d \n", NofDM);

      ker.evalAndGetCSPPointersFastSubSpace(csp_pointer_fast_space);

    // instantiate CSP Index class
      //===================================================================================================================

      // std::cout << "Testing Index class members\n";

      CSP::construct_2D_from_1D(ndiff_var, ndiff_var, csp_vec_R, csp_vec_R_2d);
      CSP::construct_2D_from_1D(ndiff_var, ndiff_var, csp_vec_L, csp_vec_L_2d);

      // instantiate CSP Index class
      CSPIndex idx(nTotalReactions, ndiff_var,
                   NofDM, eig_val_real, eig_val_imag,
                   csp_vec_R_2d, csp_vec_L_2d, Smat, RoP );

      idx.evalParticipationIndex();
      idx.evalImportanceIndexSlow();
      idx.evalImportanceIndexFast();


      idx.getParticipationIndex ( P_ik     );
      idx.getImportanceIndexSlow( Islow_jk );
      idx.getImportanceIndexFast( Ifast_jk );



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



      //
      for (int j = 0; j<(ndiff_var); j++ ) {
        fprintf(fout_cspP_fast_subspace,"%15.10e \t", csp_pointer_fast_space[j]);
      }
      fprintf(fout_cspP_fast_subspace,"\n");
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

      // eigenvector left
      for (int k = 0; k<ndiff_var; k++ ) {
        for (int j = 0; j<(ndiff_var); j++ ) {
          fprintf(fout_eig_vec_L,"%15.10e \t", eig_vec_L[k*ndiff_var + j]);
        }
        fprintf(fout_eig_vec_L,"\n");
      }
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

    // kernel class
    fclose(fout);
    fclose(fout_f);
    fclose(fout_tau);

    fclose(fout_num_rank);
    fclose(fout_eig_vec_L);
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
    fclose(fout_cspP_fast_subspace);

  }

  printf("Done ... \n" );


  return 0;
}
