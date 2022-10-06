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

#include "CSPlib_CommandLineParser.hpp"
#include "chem_elem_TCSTRI_TChem.hpp"
#include "indexBatch.hpp"
#include "kernelBatch.hpp"
#include "tools.hpp"
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
  bool write_files(false);
  int poisoning_species_idx(-1);
  int nElem(-1);

  std::string chem_file(prefixPath + "chem.inp");
  std::string therm_file(prefixPath + "therm.dat");
  std::string chem_surf_file(prefixPath + "chemSurf.inp");
  std::string therm_surf_file(prefixPath + "thermSurf.dat");

  //cstr inputs
  std::string input_file(prefixPath + "CSTRSolution.dat");
  std::string initial_condition_file("sample.dat");
  int use_analytical_Jacobian(1);
  std::string my_list_slow_index("NONE");
  std::string my_list_fast_index("NONE");
  std::string my_list_part_index("NONE");
  bool use_yaml(false);

  CSP::CommandLineParser opts("This example carries out a CSP analysis with the ChemElemTCSTR_TChem class using a transient continuous stirred tank reactor ");
  opts.set_option<std::string>("prefix", "prefix to save output files e.g., pos_ ", &firstname);
  opts.set_option<std::string>(
    "inputs-path", "prefixPath e.g.,inputs/", &prefixPath);
  opts.set_option<std::string>
  ("chemfile", "Chem file name of gas phase e.g., chem.inp",&chem_file);
  opts.set_option<std::string>
  ("thermfile", "Therm file name of gas phase  e.g., therm.dat", &therm_file);
  opts.set_option<std::string>
  ("surf-chemfile","Chem file name of surface phase e.g., chemSurf.inp",&chem_surf_file);
  opts.set_option<std::string>
  ("surf-thermfile", "Therm file name of surface phase e.g.,thermSurf.dat",&therm_surf_file);
  opts.set_option<std::string>
  ("samplefile", "Input state file name of gas phase e.g., input.dat", &initial_condition_file);
  opts.set_option<std::string>
  ("inputfile", "data base file name e.g., input.dat", &input_file);
  opts.set_option<std::string>
  ("list-slow-imp-index", "save slow importance index for these variables (id)   e.g.,1,2,3", &my_list_slow_index);
  opts.set_option<std::string>
  ("list-fast-imp-index", "save fast importance index for these variables (id)   e.g.,1,2,3", &my_list_fast_index);
  opts.set_option<std::string>
  ("list-part-index", "save participation index for these modes (id)   e.g.,1,2,3", &my_list_part_index);
  opts.set_option<real_type>("catalytic-area", "Catalytic area [m2]", &Acat);
  opts.set_option<real_type>("reactor-volume", "Reactor Volumen [m3]", &Vol);
  opts.set_option<real_type>("inlet-mass-flow", "Inlet mass flow rate [kg/s]", &mdotIn);
  opts.set_option<int>("number-of-elements",
                       "Number of elements, defualt get number of elements from TChem (-1)",
                       &nElem);
  opts.set_option<int>("number-of-algebraic-constraints",
                       "Number of algebraic constraints",
                       &number_of_algebraic_constraints);
  opts.set_option<double>("rtol", "relative tolerance for csp analysis e.g., 1e-2 ", &csp_rtolvar);
  opts.set_option<double>("atol", "absolute tolerance for csp analysis e.g., 1e-8 ", &csp_atolvar);
  opts.set_option<bool>("use-yaml", "If true, use yaml to parse input file", &use_yaml);
  opts.set_option<int>
  ("use-analytical-Jacobian",
   "Use a analytical jacobian; 1: sacado analytical jacobian, 0: numerical jacobian", &use_analytical_Jacobian);
  opts.set_option<bool>(
       "write-files", "If true, write output files", &write_files);
  opts.set_option<bool>(
      "verbose", "If true, printout state vector, jac ...", &verbose);
  //
  opts.set_option<int>("index-poisoning-species",
                       "catalysis deactivation, index for species",
                       &poisoning_species_idx);

  const bool r_parse = opts.parse(argc, argv);
  if (r_parse) return 0; // print help return


  {
    CSP::ScopeGuard guard(argc, argv);


    if (verbose) {
     printf("Reactor: Inlet mass flow rate %e, Reactor volume %e Catalytic area %e\n", mdotIn, Vol, Acat);
    }

    CSP::ChemElemTCSTRI_TChem  model(chem_file, chem_surf_file,
                                     therm_file,  therm_surf_file,
                                     number_of_algebraic_constraints,
                                     use_yaml );

    model.readDataBaseFromFile(input_file);
    model.setCSTR(initial_condition_file, mdotIn,  Vol, Acat, poisoning_species_idx);

    if (use_analytical_Jacobian > 0) {
      printf("CSPlib uses Sacado Jacobian\n");
      // sacado Jacobian
      model.evalSacadoJacobianAndSource();
    } else {
      printf("CSPlib uses Numerical Jacobian\n");
      // numerical Jacobian
      model.evalSourceVector();
      model.evalNumericalJacobian();
    }


    model.evalRoPDevice();
    model.evalSmatrixDevice();

    real_type_3d_view jacobian;
    real_type_2d_view rhs;
    real_type_2d_view state_csp;
    model.getJacMatrixDevice(jacobian);
    model.getSourceVectorDevice(rhs);
    model.getStateVectorDevice(state_csp);

    real_type_2d_view rop_fwd;
    real_type_2d_view rop_rev;
    real_type_3d_view Smat;
    model.getRoPDevice(rop_fwd, rop_rev);
    model.getSmatrixDevice(Smat);

    const real_type nBatch = model.getNumOfSamples();
    const real_type nVars = model.getNumOfVariables();
    const real_type nTotalProcesses = model.getNumOfProcesseswRevFwd();

    if (verbose)
    {
      real_type_2d_view diff("difference",nBatch, nVars );
      real_type_2d_view rhs_s_rop("rhs_s_rop",nBatch, nVars );
      /* compute diff = (rhs - S*RoP)/rhs, diff should be zero (or close to zero )*/
      verfiySmatRop<exec_space>::runBatch(Smat, rop_fwd, rop_rev, rhs, rhs_s_rop,  diff);

      const auto diff_host = Kokkos::create_mirror_view(diff);
      Kokkos::deep_copy(diff_host, diff);

      const auto rhs_host = Kokkos::create_mirror_view(rhs);
      Kokkos::deep_copy(rhs_host, rhs);

      const auto rhs_s_rop_host = Kokkos::create_mirror_view(rhs_s_rop);
      Kokkos::deep_copy(rhs_s_rop_host, rhs_s_rop);
      // print in order: serial
      for (ordinal_type i = 0; i < nBatch; i++) {
        printf("----Sample No %d----\n",i );
        for (size_t j = 0; j < nVars; j++)
          printf("No:%d rhs:%20.14e rhs_s_rop:%20.14e, diff:%20.14e\n", j, rhs_host(i,j),rhs_s_rop_host(i,j), diff_host(i,j) );
        printf("\n");

      }
    }

    if (write_files)
    {

      std::string RoP_rev_name = firstname + "_RoP_rev.dat";
      std::string RoP_fwd_name = firstname + "_RoP_fwd.dat";
      std::string source_name = firstname + "_source.dat";
      std::string Smatrix_name = firstname + "_Smatrix.dat";

      FILE *fout_smatrix = fopen ( (Smatrix_name).c_str(), "w" );
      FILE *fout_rop_rev = fopen ( (RoP_rev_name).c_str(), "w" );
      FILE *fout_rop_fwd = fopen ( (RoP_fwd_name).c_str(), "w" );
      FILE *fout_source = fopen ( (source_name).c_str(), "w" );


      const int n_processes = nTotalProcesses/2;

      FILE* fout_jacobian = fopen ( (firstname+"_Jac.dat").c_str(), "w" );

      const auto jacobian_host = Kokkos::create_mirror_view(jacobian);
      Kokkos::deep_copy(jacobian_host, jacobian);

      const auto Smat_host = Kokkos::create_mirror_view(Smat);
      Kokkos::deep_copy(Smat_host,Smat);

      // rate of progress
      const auto rop_fwd_host = Kokkos::create_mirror_view(rop_fwd);
      Kokkos::deep_copy(rop_fwd_host, rop_fwd);

      const auto rop_rev_host = Kokkos::create_mirror_view(rop_rev);
      Kokkos::deep_copy(rop_rev_host, rop_rev);

      const auto rhs_host = Kokkos::create_mirror_view(rhs);
      Kokkos::deep_copy(rhs_host, rhs);

      for (size_t i = 0; i < nBatch; i++) {
        // jacobian
        for (int k = 0; k<nVars; k++ ) {
          for (int j = 0; j<nVars; j++ ) {
            fprintf(fout_jacobian,"%20.14e \t", jacobian_host(i,k,j));
          }
          fprintf(fout_jacobian,"\n");
        }

        // s matrix
        for (int k = 0; k<nVars; k++ ) {
          for (int j = 0; j<n_processes; j++ ) {
            fprintf(fout_smatrix,"%20.14e \t", Smat_host(i,k,j));
          }
          fprintf(fout_smatrix,"\n");
        }

        // RoP rev
        for (int j = 0; j<(n_processes); j++ )
        {
          fprintf(fout_rop_rev,"%20.14e \t", rop_rev_host(i,j));
          // printf("rop_rev_host(i,j) %d %d %e\n",i, j, rop_rev_host(i,j) );
        }
        fprintf(fout_rop_rev,"\n");

        // RoP fwd
        for (int j = 0; j<(n_processes); j++ )
        {
          fprintf(fout_rop_fwd,"%20.14e \t", rop_fwd_host(i,j));
          // printf("rop_fwd_host(i,j) %d %d %e\n",i, j, rop_fwd_host(i,j) );
        }
        fprintf(fout_rop_fwd,"\n");

        // rhs
        for (int j = 0; j<(nVars); j++ )
        {
          fprintf(fout_source,"%20.14e \t", rhs_host(i,j));
        }
        fprintf(fout_source,"\n");

      } //end nBatch


      fclose(fout_rop_rev);
      fclose(fout_rop_fwd);
      fclose(fout_source);
      fclose(fout_smatrix);

      fclose(fout_jacobian);

    }

    //if (nElem < 0){
      // if nElem is negative, it is computed by TChem.
      // TChem will obtain this value from kinetic model
    //  nElem = model.getNumOfElements();
    //}
    printf("Number of conserved modes %d \n",nElem );

    CSPKernelBatch kernelBatch(jacobian, rhs, state_csp, nElem, csp_rtolvar, csp_atolvar);
    kernelBatch.evalEigenSolution(); // it includes sort eigenvalues and eigenvectors
    // kernelBatch.sortEigenSolution();
    kernelBatch.evalCSPbasisVectors();
    kernelBatch.evalCSP_Pointers();
    kernelBatch.evalTimeScales();
    kernelBatch.evalModalAmp();
    kernelBatch.evalM();

    ordinal_type_1d_view M = kernelBatch.getMDevice();
    real_type_3d_view B = kernelBatch.getLeftCSPVecDevice();
    real_type_3d_view A = kernelBatch.getRightCSPVecDevice();

    if (write_files)
    {

      FILE *fout_eig_val_real = fopen ( (firstname + "_eig_val_real.dat").c_str(), "w" );
      FILE *fout_eig_val_imag = fopen ( (firstname + "_eig_val_imag.dat").c_str(), "w" );
      FILE *fout_eig_vec_R = fopen ( (firstname + "_eig_vec_R.dat").c_str(), "w" );

      real_type_3d_view_host A_host("A host", nBatch, nVars, nVars );

      Kokkos::deep_copy( A_host, A);
      real_type_2d_view_host eigenvalues_real_part_host = kernelBatch.getEigenValuesRealPart();
      real_type_2d_view_host eigenvalues_imag_part_host = kernelBatch.getEigenValuesImagPart();

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

    CSPIndexBatch indexBatch( A, B, Smat, rop_fwd, rop_rev, M );

    indexBatch.evalParticipationIndexFwdAndRev();
    indexBatch.evalImportanceIndexSlowFwdAndRev();
    indexBatch.evalImportanceIndexFastFwdAndRev();

    ordinal_type_1d_view_host M_host = kernelBatch.getM();
    real_type_2d_view_host modal_ampl_host = kernelBatch.getModalAmp();
    real_type_2d_view_host time_scales_host = kernelBatch.getTimeScales();
    real_type_3d_view_host csp_pointers_host  = kernelBatch.getCSPPointers();

    real_type_3d_view_host SlowImpoIndex;
    real_type_3d_view_host FastImpoIndex;
    real_type_3d_view_host PartIndex;
    SlowImpoIndex =  indexBatch.getImportanceIndexSlow();
    FastImpoIndex = indexBatch.getImportanceIndexFast();
    PartIndex = indexBatch.getParticipationIndex();


    /// header
    std::string indent("    "), indent2(indent + indent), indent3(indent2 + indent);
    std::string delimiter(",");

    auto write_index = [indent, indent2, indent3]( std::string file_name,
                                                   std::vector<ordinal_type> ind_list,
                                                   const real_type_3d_view_host &data)
    {

    const ordinal_type m = data.extent(0), n = data.extent(1), l = data.extent(2);

    std::ofstream ofs_idx;
    ofs_idx.open(file_name);
    {
      std::string msg("Error: fail to open file " + file_name);
      CSPLIB_CHECK_ERROR(!ofs_idx.is_open(), msg.c_str());
    }

    ofs_idx << "{\n" << indent << "\"variable idx\" : ["<<ind_list[0];
    for (ordinal_type k = 1; k < ind_list.size(); k++) {
      ofs_idx <<","<< ind_list[k] ;
    }
    ofs_idx << "],\n";

    ofs_idx << indent << "\"number of samples\" :" << m << ",\n";

    ofs_idx << std::scientific << std::setprecision(15);

    for (ordinal_type i = 0; i < m; i++) {
      // time
      ofs_idx << indent2 << "{\n";
      ofs_idx << indent3 << "\"sample\" : " << i << ",\n";
      // write first iteam to avoid trailing comma
      ofs_idx << indent3 << "\"data\" : [ ";
      ofs_idx << "["<< data(i, 0, 0);
      for (ordinal_type k = 1; k < l; k++) {
        ofs_idx << "," << data(i, 0, k);
      } // end k
      ofs_idx << "]\n";

      for (auto it = ind_list.begin() + 1; it != ind_list.end(); ++it) {
        int j = *it;
        ofs_idx << ",["<< data(i, j, 0);
        for (ordinal_type k = 1; k < l; k++) {
          ofs_idx << "," << data(i, j, k);
        } // end k
        ofs_idx << "]\n";
      }

      // for (ordinal_type j : ind_list ){
      //   ofs_idx << ",["<< data(i, j, 0);
      //   for (ordinal_type k = 1; k < l; k++) {
      //     ofs_idx << "," << data(i, j, k);
      //   } // end k
      //   ofs_idx << "]\n";
      // } // end j

      ofs_idx << "]\n";
      ofs_idx << indent2 << "},\n";
    } // end samples
    // add a empty bracket to avoid trailing comma
    ofs_idx << "{}\n}\n";
    ofs_idx.close();

  };

    if (write_files)
    {
      // get name of species
      std::vector<std::string> spec_name;
      model.getSpeciesNames(spec_name);

      //saving species names in a file for posprocessing
      FILE *fout_sp = fopen (  (firstname + "_speciesNames.dat").c_str() , "w" );
      for (int i = 0; i<spec_name.size(); i++ )
        fprintf(fout_sp,"%s \n", (spec_name[i]).c_str());
      fclose(fout_sp);

      // kernel class
      std::string m_file_name = firstname + "_m.dat";
      std::string tau_file_name = firstname + "_tau.dat";
      std::string f_file_name = firstname + "_f.dat";
      std::string cspp_ij_name = firstname + "_cspPointers.dat";

      FILE *fout = fopen ( (m_file_name).c_str(), "w" );
      FILE *fout_tau = fopen ( (tau_file_name).c_str(), "w" );
      FILE *fout_f = fopen ( (f_file_name).c_str(), "w" );
      FILE *fout_cspP = fopen ( (cspp_ij_name).c_str(), "w" );

      printf("Writing files ...\n");
      {



      for (size_t i = 0; i < nBatch; i++) {         //M
        fprintf(fout," %d \n", M_host(i));
        //tau
        for (int k = 0; k<nVars; k++ ) {
                fprintf(fout_tau,"%20.14e \t", time_scales_host(i,k));
          }
        fprintf(fout_tau,"\n");

        // f
        for (int k = 0; k<nVars; k++ ) {
                fprintf(fout_f,"%20.14e \t", modal_ampl_host(i,k));
        }
        fprintf(fout_f,"\n");

        // csp pointer
        for (int k = 0; k<nVars; k++ ) {
          for (int j = 0; j<(nVars); j++ ) {
            fprintf(fout_cspP,"%20.14e \t", csp_pointers_host(i,k,j));
          }
          fprintf(fout_cspP,"\n");
        }

      }

      }

      // kernel class
      fclose(fout);
      fclose(fout_f);
      fclose(fout_tau);
      fclose(fout_cspP);

      // unnormalized index
      real_type_3d_view UnnormSlowImpoIndex;
      indexBatch.evalUnnormalizedImportanceIndexSlowFwdAndRev(UnnormSlowImpoIndex);
      real_type_3d_view UnnormFastImpoIndex;
      indexBatch.evalUnnormalizedImportanceIndexFastFwdAndRev(UnnormFastImpoIndex);

      auto UnnormSlowImpoIndex_host = Kokkos::create_mirror_view(UnnormSlowImpoIndex);
      Kokkos::deep_copy(UnnormSlowImpoIndex_host,UnnormSlowImpoIndex);

      auto UnnormFastImpoIndex_host = Kokkos::create_mirror_view(UnnormFastImpoIndex);
      Kokkos::deep_copy(UnnormFastImpoIndex_host,UnnormFastImpoIndex);


      if (my_list_part_index =="NONE") {
        // writing all variables
        std::string P_ik_name = firstname + "_ParticipationIndex.dat";
        FILE *fout_Pim = fopen ( (P_ik_name).c_str(), "w" );

        for (size_t i = 0; i < nBatch; i++) {
          for (int k = 0; k<nVars; k++ ) {
            for (int j = 0; j<(nTotalProcesses); j++ ) {
              fprintf(fout_Pim,"%20.14e \t", PartIndex(i,k,j));
            } // end j
            fprintf(fout_Pim,"\n");
          } // end k
        } // end i

        fclose(fout_Pim);

      } else {

        // save variables from list
        std::vector<int> part_ind_list;
        CSP::parseString(my_list_part_index, delimiter, part_ind_list );

        std::string file_name = firstname+"_PartIndx.json";
        write_index(file_name, part_ind_list, PartIndex );
      } // end slow importance index

      //Fast importance index
      if (my_list_fast_index =="NONE") {

        std::string Ifast_jk_name = firstname + "_FastImportanceIndex.dat";
        FILE *fout_Ifn = fopen ( (Ifast_jk_name).c_str(), "w" );

        FILE *fout_un_Ifi = fopen ( (firstname + "_UnnormalizedFastImportanceIndex.dat").c_str(), "w" );

        // save all variables
        for (size_t i = 0; i < nBatch; i++) {
        for (int k = 0; k<nVars; k++ ) {
          for (int j = 0; j<(nTotalProcesses); j++ ) {
            fprintf(fout_Ifn,"%20.14e \t", FastImpoIndex(i,k,j));
          }
          fprintf(fout_Ifn,"\n");
        } // end fast

        // fast importance index

        for (int k = 0; k<nVars; k++ ) {
          for (int j = 0; j<(nTotalProcesses); j++ ) {
            fprintf(fout_un_Ifi,"%20.14e \t", UnnormFastImpoIndex_host(i,k,j));
          }
        fprintf(fout_un_Ifi,"\n");
      } // end un fast

        }

        fclose(fout_Ifn);
        fclose(fout_un_Ifi);

      } else {

        std::vector<int> fast_ind_list;
        CSP::parseString(my_list_fast_index, delimiter, fast_ind_list );

        std::string file_name = firstname+"_FastImpIndx.json";
        write_index(file_name, fast_ind_list, FastImpoIndex );
        write_index(firstname+"_UnnormalizedFastImpIndx.json", fast_ind_list, UnnormFastImpoIndex_host );

      } // end fast importance index

      if (my_list_slow_index =="NONE") {

        std::string Islow_jk_name = firstname + "_SlowImportanceIndex.dat";
        FILE *fout_Isi = fopen ( (Islow_jk_name).c_str(), "w" );

        FILE *fout_un_Isi = fopen ( (firstname + "_UnnormalizedSlowImportanceIndex.dat").c_str(), "w" );


        for (size_t i = 0; i < nBatch; i++) {
          // Slow importance index
          for (int k = 0; k<nVars; k++ ) {
            for (int j = 0; j<(nTotalProcesses); j++ ) {
              fprintf(fout_Isi,"%20.14e \t", SlowImpoIndex(i,k,j));
            }
          fprintf(fout_Isi,"\n");
        } // end slow index

        for (int k = 0; k<nVars; k++ ) {
          for (int j = 0; j<(nTotalProcesses); j++ ) {
            fprintf(fout_un_Isi,"%20.14e \t", UnnormSlowImpoIndex_host(i,k,j));
          }
        fprintf(fout_un_Isi,"\n");
        }

        }

        fclose(fout_Isi);
        fclose(fout_un_Isi);

      } else {

        // ony write variables from my list
        std::vector<int> slow_ind_list;
        CSP::parseString(my_list_slow_index, delimiter, slow_ind_list );

        std::string file_name = firstname+"_SlowImpIndx.json";
        write_index(file_name, slow_ind_list, SlowImpoIndex );
        write_index(firstname+"_UnnormalizedSlowImpIndx.json", slow_ind_list, UnnormSlowImpoIndex_host );

      }

      // index clas

    }


  }
  printf("Done ... \n" );


  return 0;
}
