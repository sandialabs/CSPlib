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


#include <gtest/gtest.h>
#include "tools_tines.hpp"

int
main(int argc, char* argv[])
{
  int r_val(0);
  {
    ::testing::InitGoogleTest(&argc, argv);
    r_val = RUN_ALL_TESTS();
  }
  return r_val;
}

TEST(ChemElemTCSTR_TChem, ODE)
{
  std::string exec="../../example/indexTCSTRTChem/run_index_TCSTR_TChem.exe";
  std::string inputs="inputs/";

  // input files
  std::string chemfile      = inputs + "chemgri30.inp";
  std::string thermfile     = inputs + "thermgri30.dat";
  std::string chemSurffile  = inputs + "chemSurf.inp";
  std::string thermSurffile = inputs + "thermSurf.dat";
  std::string inputfile     = inputs + "CSTRSolutionODE.dat";
  std::string samplefile    = inputs + "sample_phi1.dat";
  // tolerances
  std::string rtol="1e-3";
  std::string atol="1e-13";
  // reactor scenario parameters
  std::string Acat="1.347e-2";
  std::string Vol="1.347e-1";
  std::string mdotIn="1e-2";
  // Jacobian, 0 is useAnalyticalJacobian
  std::string useAnalyticalJacobian="0";
  std::string verbose="false";

  std::string invoke=( exec + " --verbose=" + verbose
       + " --useAnalyticalJacobian=" + useAnalyticalJacobian
       + " --samplefile=" + samplefile
       + " --rtol=" + rtol + " --atol=" + atol + " --mdotIn=" + mdotIn
       + " --Acat=" + Acat + " --Vol=" + Vol +" --inputfile="+inputfile +
       + " --chemfile="+chemfile + " --thermfile=" +thermfile
       + " --chemSurffile="+chemSurffile + " --thermSurffile="+thermSurffile);

  // 0 corresponds to an ODE system
  std::string number_of_algebraic_constraints="0";
  std::string prefix="Constraint0";
  std::string invoke1= invoke + " --prefix=" + prefix
  + " --numberOfAlgebraicConstraints=" + number_of_algebraic_constraints;

  const auto invoke_1_c_str = invoke1.c_str();
  printf("testing : %s\n", invoke_1_c_str);
  std::system(invoke_1_c_str);

  /// compare with ref
  EXPECT_TRUE(CSP::Test::compareFilesValues("Constraint0_m.dat",
					"outputs_ref/Constraint0_m.dat") );

}

TEST(ChemElemTCSTR_TChem, DAE_No4)
{
  std::string exec="../../example/indexTCSTRTChem/run_index_TCSTR_TChem.exe";


  std::string inputs="inputs/";

  // input files
  std::string chemfile      = inputs + "chemgri30.inp";
  std::string thermfile     = inputs + "thermgri30.dat";
  std::string chemSurffile  = inputs + "chemSurf.inp";
  std::string thermSurffile = inputs + "thermSurf.dat";
  std::string inputfile     = inputs + "CSTRSolutionODE.dat";
  std::string samplefile    = inputs + "sample_phi1.dat";
  // tolerances
  std::string rtol="1e-3";
  std::string atol="1e-13";
  // reactor scenario parameters
  std::string Acat="1.347e-2";
  std::string Vol="1.347e-1";
  std::string mdotIn="1e-2";
  // Jacobian, 0 is useAnalyticalJacobian
  std::string useAnalyticalJacobian="0";
  std::string verbose="false";

  std::string invoke=( exec + " --verbose=" + verbose
       + " --useAnalyticalJacobian=" + useAnalyticalJacobian
       + " --samplefile=" + samplefile
       + " --rtol=" + rtol + " --atol=" + atol + " --mdotIn=" + mdotIn
       + " --Acat=" + Acat + " --Vol=" + Vol +" --inputfile="+inputfile +
       + " --chemfile="+chemfile + " --thermfile=" +thermfile
       + " --chemSurffile="+chemSurffile + " --thermSurffile="+thermSurffile);

  // 4 corresponds to a DAE system
  std::string number_of_algebraic_constraints="4";
  std::string prefix="Constraint4";
  std::string invoke1= invoke + " --prefix=" + prefix
  + " --numberOfAlgebraicConstraints=" + number_of_algebraic_constraints;

  const auto invoke_1_c_str = invoke1.c_str();
  printf("testing : %s\n", invoke_1_c_str);
  std::system(invoke_1_c_str);

  /// compare with ref
  EXPECT_TRUE(CSP::Test::compareFilesValues("Constraint4_m.dat",
					"outputs_ref/Constraint4_m.dat") );

}

TEST(ChemElemTCSTR_TChem, DAE_No11)
{
  std::string exec="../../example/indexTCSTRTChem/run_index_TCSTR_TChem.exe";

  std::string inputs="inputs/";

  // input files
  std::string chemfile      = inputs + "chemgri30.inp";
  std::string thermfile     = inputs + "thermgri30.dat";
  std::string chemSurffile  = inputs + "chemSurf.inp";
  std::string thermSurffile = inputs + "thermSurf.dat";
  std::string inputfile     = inputs + "CSTRSolutionODE.dat";
  std::string samplefile    = inputs + "sample_phi1.dat";
  // tolerances
  std::string rtol="1e-3";
  std::string atol="1e-13";
  // reactor scenario parameters
  std::string Acat="1.347e-2";
  std::string Vol="1.347e-1";
  std::string mdotIn="1e-2";
  // Jacobian, 0 is useAnalyticalJacobian
  std::string useAnalyticalJacobian="0";
  std::string verbose="false";

  std::string invoke=( exec + " --verbose=" + verbose
       + " --useAnalyticalJacobian=" + useAnalyticalJacobian
       + " --samplefile=" + samplefile
       + " --rtol=" + rtol + " --atol=" + atol + " --mdotIn=" + mdotIn
       + " --Acat=" + Acat + " --Vol=" + Vol +" --inputfile="+inputfile +
       + " --chemfile="+chemfile + " --thermfile=" +thermfile
       + " --chemSurffile="+chemSurffile + " --thermSurffile="+thermSurffile);

  // 11 corresponds to a DAE system
  std::string number_of_algebraic_constraints="11";
  std::string prefix="Constraint11";
  std::string invoke1= invoke + " --prefix=" + prefix
  + " --numberOfAlgebraicConstraints=" + number_of_algebraic_constraints;

  const auto invoke_1_c_str = invoke1.c_str();
  printf("testing : %s\n", invoke_1_c_str);
  std::system(invoke_1_c_str);

  /// compare with ref
  EXPECT_TRUE(CSP::Test::compareFilesValues("Constraint11_m.dat",
					"outputs_ref/Constraint11_m.dat") );

}
