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

TEST(ChemElemODETChem, ODE)
{
  std::string exec="../../example/indexODETChem/run_index_ODE_TChem.exe";
  std::string inputs="inputs/";

  // input files
  std::string chemfile      = inputs + "chem.inp";
  std::string thermfile     = inputs + "therm.dat";
  std::string inputfile     = inputs + "input.dat";
  // tolerances
  std::string rtol="1e-8";
  std::string atol="1e-14";

  // Jacobian, 0 is useAnalyticalJacobian
  std::string useAnalyticalJacobian="0";
  std::string verbose="false";

  std::string invoke=( exec + " --verbose=" + verbose
       + " --useAnalyticalJacobian=" + useAnalyticalJacobian
       + " --rtol=" + rtol + " --atol=" + atol
       + " --inputfile="+inputfile +
       + " --chemfile="+chemfile + " --thermfile=" +thermfile);

  const auto invoke_c_str = invoke.c_str();
  printf("testing : %s\n", invoke_c_str);
  std::system(invoke_c_str);

  /// compare with ref
  EXPECT_TRUE(CSP::Test::compareFilesValues("_CH4_FastImportanceIndexTopElemPosition.dat",
					"outputs_ref/_CH4_FastImportanceIndexTopElemPosition.dat") );

  EXPECT_TRUE(CSP::Test::compareFilesValues("_CH4_SlowImportanceIndexTopElemPosition.dat",
					"outputs_ref/_CH4_SlowImportanceIndexTopElemPosition.dat") );

  EXPECT_TRUE(CSP::Test::compareFilesValues("_Mode0_ParticipationIndexTopElemPosition.dat",
					"outputs_ref/_Mode0_ParticipationIndexTopElemPosition.dat") );

  EXPECT_TRUE(CSP::Test::compareFilesValues("_Temperature_FastImportanceIndexTopElemPosition.dat",
					"outputs_ref/_Temperature_FastImportanceIndexTopElemPosition.dat") );

  EXPECT_TRUE(CSP::Test::compareFilesValues("_Temperature_SlowImportanceIndexTopElemPosition.dat",
					"outputs_ref/_Temperature_SlowImportanceIndexTopElemPosition.dat") );

  EXPECT_TRUE(CSP::Test::compareFilesValues("_jac_numerical_rank.dat",
					"outputs_ref/_jac_numerical_rank.dat") );
  //
  EXPECT_TRUE(CSP::Test::compareFilesValues("_m.dat", "outputs_ref/_m.dat") );

}
