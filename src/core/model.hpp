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


#ifndef MODEL_CSP
#define MODEL_CSP

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>
#include <unistd.h>

class Model
{
 // friend class Kernel;

 public:
  Model(): Nvars(0) {};

  int init();
  int init(int Nvars);
  int setNumOfVariables(int Nvars);
  int getNumOfVariables();

  int setStateVector(const std::vector<double>& state_vec);
  int getStateVector(std::vector<double>& state_vec);

  int setStateDB(const std::vector<double>& state_db);
  int getStateDB(std::vector<double>& state_db);

  int getSourceVector(std::vector<double>& source_vec);

  int getJacMatrix(std::vector<std::vector<double> >& jmat);

  virtual int evalSourceVector()=0;
  virtual int evalJacMatrix(unsigned int useJacAnl)=0;

 protected:

  // single state vector structures
  int Nvars;
  std::vector<double>               _state_vec;
  std::vector<double>               _source_vec;
  std::vector<std::vector<double> > _jmat;

  // data base data structures -- for efficiency arguments -- are all stored
  // in 1-dimensional vectors -- and intended to be ultimately communicated
  // to tchem as such

  // number of entries (state vectors, source vectors, jac matrices) in db
  int Ndb;

  // database of state vectors (vec_0,vec_1,...,vec_(Ndb-1))
  // i-th state variable in j-th state vector in db is at: _state_db[i+j*Nvars]
  std::vector<double>               _state_db;

  // database of source vectors (vec_0,vec_1,...,vec_(Ndb-1))
  // i-th source term in j-th source vector in db is at: _source_db[i+j*Nvars]
  std::vector<double>               _source_db;

  // database of jacobian matrices (mat_0,mat_1,...,mat_(Ndb-1))
  // each matrix is stored column major in 1D (col_0,col_1,...,col_(Nvars-1))
  // entry in row i, col j, of jac matrix k is stored in _jac_db at:
  // _jac_db[i+j*Nvars+k*Nvars*Nvars]
  std::vector<double>               _jac_db;

 private:

};

#endif  //end of header guard
