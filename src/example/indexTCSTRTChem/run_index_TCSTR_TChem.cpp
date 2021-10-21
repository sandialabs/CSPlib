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
#include "util.hpp"
#include "chem_elem_TCSTR_TChem.hpp"
#include "kernel.hpp"
#include "index.hpp"
#include "tools.hpp"
#include "CSPlib_CommandLineParser.hpp"


#include "vio.h"
// #define CSP_ENABLE_KOKKOS_EIGEN_SOLVER
int main(int argc, char *argv[]) {

  // reactions mechanism and therm files
  std::string prefixPath("");
  std::string firstname("");
  double csp_rtolvar(1.e-2); //1.e-7; // 1.e+3; //
  double csp_atolvar(1.e-8); //1.e-8; // 1.e+3; //
  int number_of_algebraic_constraints(0);

  double mdotIn(3.596978981250784e-06);
  double Vol(0.00013470);
  double Acat (0.0013074);
  bool verbose(false);
  bool isoThermic(false);
  bool compute_M_wo_exp(false);

  std::string chemFile(prefixPath + "chem.inp");
  std::string thermFile(prefixPath + "therm.dat");
  std::string chemSurfFile(prefixPath + "chemSurf.inp");
  std::string thermSurfFile(prefixPath + "thermSurf.dat");

  //cstr inputs
  std::string inputFile(prefixPath + "CSTRSolution.dat");
  std::string initialConditionFile("sample.dat");
  int use_analytical_Jacobian(1);

  CSP::CommandLineParser opts("This example carries out a CSP analysis with the ChemElemTCSTR_TChem class using a transient continuous stirred tank reactor ");
  // opts.set_option<std::string>("inputsPath", "path to inputs data(chem.inp, therm.dat ...) e.g.,data/", &prefixPath);
  opts.set_option<std::string>("prefix", "prefix to save output files e.g., pos_ ", &firstname);
  opts.set_option<double>("rtol", "relative tolerance for csp analysis e.g., 1e-2 ", &csp_rtolvar);
  opts.set_option<double>("atol", "absolute tolerance for csp analysis e.g., 1e-8 ", &csp_atolvar);
  opts.set_option<double>("Acat", "Catalytic area [m2]", &Acat);
  opts.set_option<double>("Vol", "Reactor Volumen [m3]", &Vol);
  opts.set_option<double>("mdotIn", "Inlet mass flow rate [kg/s]", &mdotIn);
  opts.set_option<bool>("isoThermic", "if True, reaction is isotermic", &isoThermic);
  opts.set_option<bool>("computeM_WoExpFactor", "if True, reaction is isotermic", &compute_M_wo_exp);
  opts.set_option<int>
  ("useAnalyticalJacobian",
   "Use a analytical jacobian; 0: sacado analytical jacobian, 1: numerical jacobian", &use_analytical_Jacobian);
  opts.set_option<int>
  ("numberOfAlgebraicConstraints",
   "number of alegraic constraints, if it bigger than 1 system is tried as a DAE", &number_of_algebraic_constraints);

  opts.set_option<std::string>
  ("chemfile", "Chem file name e.g., chem.inp",
  &chemFile);
  opts.set_option<std::string>
  ("thermfile", "Therm file name e.g., therm.dat", &thermFile);
  opts.set_option<std::string>
  ("inputfile", "data base file name e.g., input.dat", &inputFile);
  opts.set_option<std::string>
  ("chemSurffile","Chem file name e.g., chemSurf.inp",
   &chemSurfFile);
  opts.set_option<std::string>
  ("thermSurffile", "Therm file name e.g.,thermSurf.dat",
  &thermSurfFile);
  opts.set_option<std::string>
  ("samplefile", "Input state file name e.g., input.dat", &initialConditionFile);
  opts.set_option<bool>(
      "verbose", "If true, printout state vector, jac ...", &verbose);

  const bool r_parse = opts.parse(argc, argv);
  if (r_parse) return 0; // print help return


  {
    CSP::ScopeGuard guard(argc, argv);

    if (verbose) {
     printf("Reactor: Mass flow inlet %e,  Volume %e Catalytic area %e\n", mdotIn, Vol, Acat);
    }
    ChemElemTCSTR_TChem  model(chemFile, chemSurfFile,
                                 thermFile,  thermSurfFile,
                                 number_of_algebraic_constraints );
    //
    //set dimension of _state_vec, _source_vec,  and  _jmat
    model.init();

    // read a data base from a TChem++ TCSTR solution
    std::vector<std::string> var_names;
    model.readDataBaseFromFile(inputFile, var_names);

    // get number of variables in the ODE system
    auto ndiff_var = model.getNumOfVariables();
    std::cout<< "ndiff_var = "<< ndiff_var <<"\n";

    // get name of species
    std::vector<std::string> spec_name;
    model.getSpeciesNames(spec_name);

    std::vector<double> state(ndiff_var);

    if (verbose) {
      //get state vector at fist index
      model.getStateVector(state);
      std::cout << "state: " << std::endl;
      printf("Temperature %20.14e \n",state[0] );
      for (int k = 0; k < spec_name.size(); k++)
        printf(" %s %20.14e\n",(spec_name[k]).c_str(),state[k+1] );
    }

    //saving species names in a file for posprocessing
    std::string species_name = firstname + "_speciesNames.dat";
    FILE *fout = fopen (  (species_name).c_str() , "w" );
    {
      for (int i = 0; i<spec_name.size(); i++ )
          fprintf(fout,"%s \n", (spec_name[i]).c_str());
      fclose(fout);
    }
    // set parameter for CSTR
    model.setCSTR(initialConditionFile, mdotIn,  Vol, Acat, isoThermic  );
    //computes RHS on devices
    model.evalSourceVector();
    std::vector<double> source(ndiff_var);
    //
    if (verbose) {
      //get source vector at first index
      model.getSourceVector(source);
      std::cout << "source: " << std::endl;
      printf("Temperature %20.14e \n",source[0] );
      for (int k = 0; k < spec_name.size(); k++)
        printf(" %s %20.14e\n",(spec_name[k]).c_str(),source[k+1] );
    }

    model.evalJacMatrix(use_analytical_Jacobian);
    std::vector<std::vector<double>> jac (ndiff_var,std::vector<double>(ndiff_var, 0.0) );

    if (verbose) {
      //get jac at first index
      model.getJacMatrix(jac);
      Util::Print::mat<double>("jac", RIF, Out2d, Dbl, 3, 3, jac);
    }
#if defined(CSP_ENABLE_KOKKOS_EIGEN_SOLVER)

    printf("CSPlib is using Tines' EigenSolver\n");

    std::vector< std::vector< double> > eig_vec_L_2D;
    std::vector< std::vector< double> > eig_vec_R_2D;

    std::vector< std::vector< double> >  eig_val_real_bath;
    std::vector< std::vector< double> >  eig_val_imag_bath;
    std::vector< std::vector< std::vector< double> > > eig_vec_L_bath;
    std::vector< std::vector< std::vector< double> > > eig_vec_R_bath;


    EigenSolver::evalDevice(model._jac,
                            eig_val_real_bath,
                            eig_val_imag_bath,
                            eig_vec_R_bath);

#endif

    model.evalSmatrix();

    // compute and get S matrixes for gas and surface
    const auto nGasReactions     = model.getNumofGasReactions();
    const auto nSurfaceReactions = model.getNumofSurfaceReactions();

    // rate of progress: gas/surface forward, reverse, and conv from T-CSTR
    const auto nRoP = model.getNumOfRateOfProcesses();

    std::vector<std::vector<double>>
    Smat(ndiff_var, std::vector<double>(nRoP,0.0) );

    model.evalRoP();
    std::vector<double> RoP(nRoP); // total

    //compute rhs (source) from Matrix produc of Smat and RoP
    //RHS = Smat * RoP
    if (verbose) {
      model.getSmatrix(Smat);

      model.getRoP(RoP);

      std::vector<double> rhsSmaRop(nRoP);
      model.verifySmatRoP(Smat, RoP, rhsSmaRop);

      printf("Temperature (Smat*RoP) %e rhs %e  diff %e\n", rhsSmaRop[0], source[0],
       (rhsSmaRop[0]-source[0])/(source[0]+1e-23)  );

      for (int k = 0; k < spec_name.size(); k++)
        printf(" %s (Smat*RoP) %e rhs %e  diff %e\n",(spec_name[k]).c_str(), rhsSmaRop[k+1], source[k+1],
         (rhsSmaRop[k+1]-source[k+1])/(source[k+1]+1e-23) );

    }

    const int nElem = model.getNumOfElements();
    //==========
    std::vector< std::vector< double> > state_db;
    model.getStateDBonHost(state_db);

    std::vector< std::vector< double> > source_db;
    model.getSourceDBonHost(source_db);

    std::vector< std::vector< std::vector< double> > > jac_db;
    model.getJacobianDBonHost(jac_db);

    std::vector< std::vector< double> > RoP_db;
    model.getRoPDBonHost(RoP_db);

    std::vector< std::vector< std::vector< double> > > Smatrixdb;
    model.getSmatrixDBonHost(Smatrixdb);

    std::vector<double> eig_val_real;
    std::vector<double> eig_val_imag;

    std::vector<double> eig_vec_L(ndiff_var*ndiff_var);
    std::vector<double> eig_vec_R(ndiff_var*ndiff_var);


    std::vector<double> csp_vec_R(ndiff_var*ndiff_var);
    std::vector<double> csp_vec_L(ndiff_var*ndiff_var);
    std::vector<double> tau_vec;
    std::vector<double> f_vec;

    int nmode = ndiff_var;
    std::vector<std::vector<double> > csp_vec_R_2d, csp_vec_L_2d;
    //
    std::vector<std::vector<double> > dRoP(ndiff_var, std::vector<double>(nRoP, 0.0) ); // Couldn't find TChem2 function to populate it yet.
    std::vector<double> Wbar(nmode,0);

    std::vector<std::vector<double> > P_ik    ;
    std::vector<std::vector<double> > Islow_jk;
    std::vector<std::vector<double> > Ifast_jk;
    std::vector<std::vector<double> > cspp_ij;


    std::vector< double > gfast(ndiff_var,0);

    std::string m_file_name = firstname + "_m.dat";
    std::string tau_file_name = firstname + "_tau.dat";
    std::string jac_name = firstname + "_jac.dat";
    std::string RoP_name = firstname + "_RoP.dat";

    std::string state_name = firstname + "_state.dat";
    std::string source_name = firstname + "_source.dat";
    std::string csp_vec_R_name = firstname + "_csp_vec_R.dat";
    std::string Smat_name = firstname + "_Smat.dat";

    std::string eigen_value_real_name = firstname +"_eigen_value_real.dat";
    std::string eigen_value_imag_name = firstname +"_eigen_value_imag.dat";

    fout = fopen ( (m_file_name).c_str(), "w" );
    FILE *fout_state = fopen ( (state_name).c_str(), "w" );
    FILE *fout_source = fopen ( (source_name).c_str(), "w" );

    FILE *fout_tau = fopen ( (tau_file_name).c_str(), "w" );
    FILE *fout_jac = fopen ( (jac_name).c_str(), "w" );
    FILE *fout_RoP = fopen ( (RoP_name).c_str(), "w" );
    FILE *f_out_csp_vec_R = fopen ( (csp_vec_R_name).c_str(), "w" );
    FILE *f_out_Smat = fopen ( (Smat_name).c_str(), "w" );

    FILE *f_out_eigen_value_real = fopen ( (eigen_value_real_name).c_str(), "w" );
    FILE *f_out_eigen_value_imag = fopen ( (eigen_value_imag_name).c_str(), "w" );

    std::string eigenResidual_file_name = firstname + "_eigenHighResidual.dat";
    FILE *fout_eigenResidual = fopen ( (eigenResidual_file_name).c_str(), "w" );

    std::string gfast_file_name = firstname + "_gfast.dat";
    FILE *fout_gfast = fopen ( (gfast_file_name).c_str(), "w" );

    std::string P_ik_name = firstname + "_ParticipationIndex.dat";
    std::string Islow_jk_name = firstname + "_SlowImportanceIndex.dat";
    std::string Ifast_jk_name = firstname + "_FastImportanceIndex.dat";
    std::string cspp_ij_name = firstname + "_cspPointers.dat";

    FILE *fout_Pim = fopen ( (P_ik_name).c_str(), "w" );
    FILE *fout_Isi = fopen ( (Islow_jk_name).c_str(), "w" );
    FILE *fout_Ifn = fopen ( (Ifast_jk_name).c_str(), "w" );
    FILE *fout_cspP = fopen ( (cspp_ij_name).c_str(), "w" );

    std::string num_rank_file_name = firstname + "_jac_numerical_rank.dat";
    FILE *fout_num_rank = fopen ( (num_rank_file_name).c_str(), "w" );


    std::string magMode_file_name = firstname + "_f.dat";
    FILE *fout_magMode = fopen ( (magMode_file_name).c_str(), "w" );

    // char norm='1'; // '1' or 'O'= 1-norm; 'I'= Infinity-norm
    // double cond_num;
    std::string csp_pointer_fast_space_name = firstname + "_cspPointersFastSubSpace.dat";
    std::vector<double> csp_pointer_fast_space(ndiff_var,0.0);
    FILE *fout_cspP_fast_subspace = fopen ( (csp_pointer_fast_space_name).c_str(), "w" );

    const int nSample = state_db.size();

    printf("Number of states %d \n", nSample);


    for (int i = 0; i < nSample; i++) {
      source = source_db[i];
      state  = state_db[i];
      jac    = jac_db[i];
      Smat   = Smatrixdb[i];
      RoP    = RoP_db[i];

      // compute rhs (source) from Matrix produc of Smat and RoP
      // RHS = Smat * RoP
      if (verbose) {
        printf("----------- Sample No %d ------------\n", i );
        std::vector<double> rhsSmaRop(nRoP);
        model.verifySmatRoP(Smat, RoP, rhsSmaRop);
        for (int k = 0; k < ndiff_var; k++) {
          std::string var_k_name;
          if (k != 0) {
            var_k_name = spec_name[k-1];
          }else {
            var_k_name = "Temperature";
          }
          printf("k %d variable %s Smat*RoP %e source %e diff %e \n", k, var_k_name.c_str(), rhsSmaRop[k],
          source[k], (rhsSmaRop[k]-source[k])/(source[k]+1e-23) );
        }

      }
      // std::cout<<"state "<< state <<"\n";

      Kernel ker(ndiff_var, state, source, jac);

      int jac_rank = ker.computeJacobianNumericalRank();
      fprintf(fout_num_rank," %d \n", jac_rank);

#if !defined(CSP_ENABLE_KOKKOS_EIGEN_SOLVER)
      // Eigen solution:
      ker.evalEigenValVec();
#else
     //
     eig_val_real = eig_val_real_bath[i];
     eig_val_imag = eig_val_imag_bath[i];
     eig_vec_R_2D    = eig_vec_R_bath[i];

     // convert 2D TO 1D
     int count=0;
     for (size_t k=0; k<ndiff_var; k++) {
       for (size_t j=0; j<ndiff_var; j++) {
         eig_vec_R[count] = eig_vec_R_2D[k][j];
         count++;
       }
     }
     ker.setEigenValVec(eig_val_real, eig_val_imag, eig_vec_R);

#endif

      // ker.DiagEigValVec();
      // fprintf(fout_eigenResidual," %e \n", ker.getEigenResidual());

      // Sorting eigen values and vectors in ascending order
      // of, sign(eig_val_real)*Mod(eig_val_real + i * eig_val_imag)
      ker.sortEigValVec();
      ker.getEigenValVec(eig_val_real, eig_val_imag, eig_vec_L, eig_vec_R);

      // Setting CSP vectors:
      ker.setCSPVec(); // A = eig_vec_R and B = A^{-1}
      ker.getCSPVec(csp_vec_L, csp_vec_R);


      //========================================================================================
      // Time scales:
      // std::cout << "Evaluating time scales\n";
      // std::cout << "Calling evalTau and getTau:"<<std::endl;
      ker.evalTau();
      ker.getTau(tau_vec);
      // Util::Print::vec<double>("tau_vec", Col, O1d, Dbl, tau_vec.size(), tau_vec);

      ker.evalModalAmp( );
      ker.getModalAmp( f_vec );

      ker.setCSPerr(csp_rtolvar, csp_atolvar);
      //========================================================================================
      // Exhausted mode
      // std::cout<<"+++++++++++++++++++++++++=\n";
      int NofDM = 0;
      if (compute_M_wo_exp) {
        // compute number of exhausted model(M) without exporential factor
        // which means amplitude of the mode (f) is constant over dt
        ker.evalM_WoExp(nElem);
      if (verbose ) {
        printf("Computing M, assuming f is constant \n");
      }
      } else {
        // compute M assuming that f is not constant, and it has exp. decay
        ker.evalM(nElem);
      }

      ker.getM(NofDM);
      fprintf(fout," %d \n", NofDM);

      ker.evalCSPPointers();
      ker.getCSPPointers( cspp_ij );

      ker.evalAndGetCSPPointersFastSubSpace(csp_pointer_fast_space);


#if 0
      ker.evalAndGetgfast(gfast);
      for (int k = 0; k < ndiff_var; k++) {
        printf("i %d gfast %e source %e ratio(gfas/g) %e \n", k, gfast[k],
        source[k], (gfast[k])/source[k] );
      }

#endif



      // ker.evalAndGetMvec(Mvec,nElem);

      // std::cout << " M (no. of Exhausted modes)  = " << NofDM <<"\n";

      // instantiate CSP Index class
      //===================================================================================================================

      // std::cout << "Testing Index class members\n";
      CSP::construct_2D_from_1D<double>(ndiff_var, ndiff_var, csp_vec_R, csp_vec_R_2d);
      CSP::construct_2D_from_1D<double>(ndiff_var, ndiff_var, csp_vec_L, csp_vec_L_2d);

      // instantiate CSP Index class
      CSPIndex idx(nRoP, ndiff_var,
                   NofDM, eig_val_real, eig_val_imag,
                   csp_vec_R_2d, csp_vec_L_2d, Smat, RoP  );

      // idx.initChemKinModel( Smat, RoP, dRoP, Wbar); // twice the nreac

      idx.evalParticipationIndex();
      idx.evalImportanceIndexSlow();
      idx.evalImportanceIndexFast();


      idx.getParticipationIndex ( P_ik     );
      idx.getImportanceIndexSlow( Islow_jk );
      idx.getImportanceIndexFast( Ifast_jk );


      //save data
      for (int j = 0; j<(ndiff_var); j++ ) {
      fprintf(fout_cspP_fast_subspace,"%15.10e \t", csp_pointer_fast_space[j]);
      }
      fprintf(fout_cspP_fast_subspace,"\n");

      for (int j = 0; j<(ndiff_var); j++ ) {
          fprintf(f_out_eigen_value_real,"%15.10e \t", eig_val_real[j]);
      }
      fprintf(f_out_eigen_value_real,"\n");

      for (int j = 0; j<(ndiff_var); j++ ) {
          fprintf(f_out_eigen_value_imag,"%15.10e \t", eig_val_imag[j]);
      }
      fprintf(f_out_eigen_value_imag,"\n");


      for (int j = 0; j<(ndiff_var); j++ ) {
          fprintf(fout_state,"%15.10e \t", state[j]);
      }
      fprintf(fout_state,"\n");


      for (int j = 0; j<(ndiff_var); j++ ) {
          fprintf(fout_source,"%15.10e \t", source[j]);
      }
      fprintf(fout_source,"\n");

        // jac
      for (int i = 0; i<ndiff_var; i++ ) {
        for (int j = 0; j<(ndiff_var); j++ ) {
           fprintf(fout_jac,"%15.10e \t", jac[i][j]);
        }
        fprintf(fout_jac,"\n");
      }

      for (int j = 0; j<(nRoP); j++ ) {
       fprintf(fout_RoP,"%15.10e \t", RoP[j]);
      }
      fprintf(fout_RoP,"\n");

      for (int k = 0; k<ndiff_var; k++ ) {
        for (int j = 0; j<(ndiff_var); j++ ) {
          fprintf(f_out_csp_vec_R,"%15.10e \t", csp_vec_R_2d[k][j]);
        }
        fprintf(f_out_csp_vec_R,"\n");
      }



      for (int k = 0; k<ndiff_var; k++ ) {
        for (int j = 0; j<(nRoP); j++ ) {
          fprintf(f_out_Smat,"%15.10e \t", Smat[k][j]);
        }
        fprintf(f_out_Smat,"\n");
      }
      for (int k = 0; k<gfast.size(); k++ ) {
            fprintf(fout_gfast,"%20.14e \t", gfast[k]);
      }
      fprintf(fout_gfast,"\n");

      // mode amplitud
      for (int k = 0; k<f_vec.size(); k++ ) {
            fprintf(fout_magMode,"%20.14e \t", f_vec[k]);
      }
      fprintf(fout_magMode,"\n");


      //tau
      for (int k = 0; k<tau_vec.size(); k++ ) {
            fprintf(fout_tau,"%20.14e \t", tau_vec[k]);
      }
      fprintf(fout_tau,"\n");

      //Participation index
      for (int k = 0; k<ndiff_var; k++ ) {
        for (int j = 0; j<(nRoP); j++ ) {
          fprintf(fout_Pim,"%15.10e \t", P_ik[k][j]);
        }
        fprintf(fout_Pim,"\n");
      }


      // Slow importance index
      for (int k = 0; k<ndiff_var; k++ ) {
        for (int j = 0; j<(nRoP); j++ ) {
          fprintf(fout_Isi,"%15.10e \t", Islow_jk[k][j]);
        }
        fprintf(fout_Isi,"\n");
      }

      //Fast importance index
      for (int k = 0; k<ndiff_var; k++ ) {
        for (int j = 0; j<(nRoP); j++ ) {
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


    } //end of samples
    fclose(fout);
    fclose(fout_tau);
    fclose(fout_eigenResidual);
    fclose(fout_gfast);
    fclose(f_out_csp_vec_R);
    fclose(f_out_Smat);

    fclose(fout_Pim);
    fclose(fout_Isi);
    fclose(fout_Ifn);
    fclose(fout_cspP);
    fclose(fout_RoP);
    fclose(fout_num_rank);
    fclose(fout_magMode);
    fclose(fout_state);
    fclose(fout_source);
    fclose(fout_jac);
    fclose(f_out_eigen_value_real);
    fclose(f_out_eigen_value_imag);
    fclose(fout_cspP_fast_subspace);



  }
  printf("Done ... \n" );


  return 0;
}
