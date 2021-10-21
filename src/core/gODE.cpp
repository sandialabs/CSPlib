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

//===========================================================
GeneralODE::GeneralODE(
    std::function<int(const std::vector<double>& state, std::vector<double>& source)> gfunc,
    std::function<int(const std::vector<double>& state, std::vector<std:: vector<double>>& jmat, unsigned int flag)> jfunc
   ) {
  rhsfunc = gfunc;
  jacfunc = jfunc;
  return;
}
//===========================================================
int GeneralODE::evalSourceVector() { 
  rhsfunc(_state_vec, _source_vec);
  return(0);
} 

//===========================================================
int GeneralODE::evalJacMatrix(unsigned int useJacAnl) { 
  jacfunc(_state_vec, _jmat, useJacAnl);
  return(0);
} 

//===========================================================
