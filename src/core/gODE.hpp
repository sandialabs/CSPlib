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


#ifndef MODEL_CSP_gODE
#define MODEL_CSP_gODE

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>
#include <unistd.h>      
#include <functional>

#include "model.hpp"

class GeneralODE: public Model {

 public:
  GeneralODE(
    std::function<int(const std::vector<double>&, std::vector<double>&)>,
    std::function<int(const std::vector<double>&, std::vector<std::vector<double>>&, unsigned int )>
  );

  int evalSourceVector();                     // from Model
  int evalJacMatrix(unsigned int useJacAnl);  // from Model

 private:
  std::function<void(const std::vector<double>&, std::vector<double>&)> rhsfunc;
  std::function<void(const std::vector<double>&, std::vector<std::vector<double>>&, unsigned int )> jacfunc;

};


#endif  //end of header guard
