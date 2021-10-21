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
#include "gODE.hpp"
#include "kernel.hpp"
#include "CSPlib_CommandLineParser.hpp"
#include "eigendecomposition_kokkos.hpp"
// #include "chem_elem_ODE_TChem.hpp"

int rhs_Davis_Skodje(const std::vector<double>& state, std::vector<double>& source){
    const double epsilon = 0.01;
    const double y = state[0];
    const double z = state[1];
    source[0] = (-y+z/(1.+z))/epsilon - z/(1.+z)/(1.+z);
    source[1] = -z;
  return(0);
}

int jac_Davis_Skodje(const std::vector<double>& state, std::vector<std::vector<double>>& jac, int flag){
  const double epsilon = 0.01;
  const double y = state[0];
  const double z = state[1];

  jac[0][0] = -1./epsilon;
  jac[1][0] = 0;
  jac[0][1] = 2. * z / std::pow( z + 1. , 3.) - 1. / std::pow( z + 1., 2) +
              ( - z / std::pow( z + 1. , 2.) + 1. / ( z + 1. ) ) / epsilon;
  jac[1][1] = -1;
  return(0);
}

int state_Davis_Skodje(std::vector<double>& state, const double t, std::vector<double>& state0){
    const double epsilon = 0.01;
    const double y0 = state0[0];
    const double z0 = state0[1];

    state[0] = (y0-z0/(1.+z0)) * std::exp(-t/epsilon) + z0*std::exp(-t)/(1.+z0*std::exp(-t));
    state[1] = z0*std::exp(-t);

    return 1;

}

int main(int argc, char *argv[]) {


  CSP::ScopeGuard guard(argc, argv);

  double csp_rtolvar(1.e-3); //1.e-7; // 1.e+3; //
  double csp_atolvar(1.e-14); //1.e-8; // 1.e+3; //
  double y0(2);
  double z0(1e-2);
  double tend(4);
  int nPoints(2000);

  CSP::CommandLineParser opts("This example Number of exhausted and time scale for mDavis Skodje problem");
  opts.set_option<double>("rtol", "relative tolerance for csp analysis e.g., 1e-2 ", &csp_rtolvar);
  opts.set_option<double>("atol", "absolute tolerance for csp analysis e.g., 1e-8 ", &csp_atolvar);
  opts.set_option<double>("y0", "initial value for y e.g., 2 ", &y0);
  opts.set_option<double>("z0", "initial value for y e.g., 1e-2 ", &z0);
  opts.set_option<double>("tend", "time end e.g., 4 ", &tend);
  opts.set_option<int>("nPoints", "number of points  e.g., 2000 ", &nPoints);

  const bool r_parse = opts.parse(argc, argv);
  if (r_parse) return 0; // print help return

  int flag=1;

GeneralODE mDavis_Skodje(
  std::function<int(const std::vector<double>&, std::vector<double>&)> (std::move(rhs_Davis_Skodje)),
  std::function<int(const std::vector<double>&, std::vector<std::vector<double>>&, int)> (std::move(jac_Davis_Skodje))
);

const int ndiff_var(2);
mDavis_Skodje.init(ndiff_var);
std::vector<double> stateDS (2);
stateDS[0] = y0;
stateDS[1] = z0;

std::vector<double> state (ndiff_var);
std::vector<double> source (ndiff_var);
std::vector<std::vector<double>> jac (ndiff_var,std::vector<double>(ndiff_var,0));

const double dt = tend/nPoints;
double t(0);

std::vector<double> csp_vec_R(ndiff_var*ndiff_var);
std::vector<double> csp_vec_L(ndiff_var*ndiff_var);
std::vector<double> tau_vec;
std::vector<double> f_vec;

std::vector<std::vector<double> > cspp_ij;

const int nElem(0);

std::string firstname("");

std::string mNew_file_name = firstname + "_m.dat";
FILE *fout = fopen ( (mNew_file_name).c_str(), "w" );

std::string tau_file_name = firstname + "_tau.dat";
FILE *fout_tau = fopen ( (tau_file_name).c_str(), "w" );

std::string num_rank_file_name = firstname + "_jac_numerical_rank.dat";
FILE *fout_num_rank = fopen ( (num_rank_file_name).c_str(), "w" );

std::string magMode_file_name = firstname + "_magMode.dat";
FILE *fout_magMode = fopen ( (magMode_file_name).c_str(), "w" );

std::string state_file_name = firstname + "_state.dat";
FILE *fout_state = fopen ( (state_file_name).c_str(), "w" );

std::string time_file_name = firstname + "_time.dat";
FILE *fout_time = fopen ( (time_file_name).c_str(), "w" );

std::string cspp_ij_name = firstname + "_cspPointers.dat";
FILE *fout_cspP = fopen ( (cspp_ij_name).c_str(), "w" );


// databases
std::vector< std::vector< double> > state_db ;
std::vector< std::vector< double> > source_db ;
std::vector< std::vector< std::vector< double > > > jac_db ;

for (int sp = 0; sp < nPoints; sp++) {

  state_Davis_Skodje(state, t, stateDS);
  mDavis_Skodje.setStateVector(state);
  mDavis_Skodje.evalSourceVector();
  mDavis_Skodje.getSourceVector(source);

  mDavis_Skodje.evalJacMatrix(flag);
  mDavis_Skodje.getJacMatrix(jac);

  fprintf(fout_time," %e \n", t);
  t +=dt;

  //state
  for (int k = 0; k<state.size(); k++ ) {
        fprintf(fout_state,"%20.14e \t", state[k]);
  }
  fprintf(fout_state,"\n");

  state_db.push_back(state);
  source_db.push_back(source);
  jac_db.push_back(jac);

}

std::vector<double> eig_val_real;
std::vector<double> eig_val_imag;
std::vector<double> eig_vec_R(ndiff_var*ndiff_var);
std::vector< std::vector< double> > eig_vec_R_2D;

std::vector< std::vector< double> >  eig_val_real_bath;
std::vector< std::vector< double> >  eig_val_imag_bath;
std::vector< std::vector< std::vector< double> > > eig_vec_R_bath;

EigenSolver::evalDevice(jac_db,
                        eig_val_real_bath,
                        eig_val_imag_bath,
                        eig_vec_R_bath);

for (int sp = 0; sp < nPoints; sp++) {
  state  = state_db[sp];
  source = source_db[sp];
  jac    = jac_db[sp];

  eig_val_real = eig_val_real_bath[sp];
  eig_val_imag = eig_val_imag_bath[sp];
  eig_vec_R_2D    = eig_vec_R_bath[sp];

  // convert 2D TO 1D
  int count=0;
  for (size_t k=0; k<ndiff_var; k++) {
    for (size_t j=0; j<ndiff_var; j++) {
      eig_vec_R[count] = eig_vec_R_2D[k][j];
      count++;
    }
  }

  Kernel ker(ndiff_var, state, source, jac);
  int jac_rank = ker.computeJacobianNumericalRank();
  fprintf(fout_num_rank," %d \n", jac_rank);


  ker.setEigenValVec(eig_val_real, eig_val_imag, eig_vec_R);
  // Sorting eigen values and vectors in ascending order
  // of, sign(eig_val_real)*Mod(eig_val_real + i * eig_val_imag)
  ker.sortEigValVec();
  // Setting CSP vectors:
  ker.setCSPVec(); // A = eig_vec_R and B = A^{-1}
  ker.getCSPVec(csp_vec_L, csp_vec_R);

  ker.evalTau();
  ker.getTau(tau_vec);
  ker.evalModalAmp( );
  ker.getModalAmp( f_vec );

  ker.setCSPerr(csp_rtolvar, csp_atolvar);

  int NofDM = 0;
  ker.evalM(nElem);
  ker.getM(NofDM);
  fprintf(fout," %d \n", NofDM);

  //csp_pointer
  ker.evalCSPPointers();
  ker.getCSPPointers( cspp_ij );

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

  // csp pointer
    for (int k = 0; k<ndiff_var; k++ ) {
      for (int j = 0; j<(ndiff_var); j++ ) {
        fprintf(fout_cspP,"%15.10e \t", cspp_ij[k][j]);
      }
      fprintf(fout_cspP,"\n");
    }

}

fclose(fout_magMode);
fclose(fout_num_rank);
fclose(fout_tau);
fclose(fout);
fclose(fout_state);
fclose(fout_time);
fclose(fout_cspP);

//====================================================================
  return 0;
}
