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


#ifndef TOOLS_CSP
#define TOOLS_CSP

#include <algorithm>

#include <string.h>
#include <typeinfo>
#include <cmath>
#include <numeric>      // std::inner_product

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <complex>

#include <iostream>
#include <sys/time.h>

#include <fstream>

/// csp configure header
#include "CSPlib_ConfigDefs.h"

namespace CSP {

  void MatrixVectorMul(std::vector<std::vector<double> >&  A,
                           std::vector<double>& b, std::vector<double>& c);

  int LinearSystemSolve(
  	       int nrowA, int ncolA,
  	       int nrowB, int ncolB,
                 std::vector<double> &matA_1d,
                 std::vector<double> &matB_1d,
                 std::vector<double> &matX_1d);

  int LinearSystemSolve(
  	       int nrowA, int ncolA,
  	       int nrowB, int ncolB,
                 std::vector<std::vector<double> > &matA_2d,
                 std::vector<std::vector<double> > &matB_2d,
                 std::vector<std::vector<double> > &matX_2d);

  //-----------------------------------------------------------
  /*
     Construct 2d matrix of dimension nrow by ncol,
     from 1d vector of size (nrow*ncol), read in row-major form.
     check the use of std::insert
  */

  template<typename T>
  void construct_2D_from_1D(int nrow, int ncol, std::vector<T> &D1_in, std::vector<std::vector<T> > &D2_out)
  {

    D2_out = std::vector<std::vector<T> >(nrow);

    auto it = D1_in.begin();
    // row-major
    for (size_t i=0; i<nrow; i++) {
      // Both of the following two line should work.
      //D2_out[i].insert(D2_out[i].end(), &D1_in[i*ncol], &D1_in[(i+1)*ncol] );
      D2_out[i].insert(D2_out[i].end(), it+(i*ncol), it+((i+1)*ncol) );
    }

  }

  //===========================================================
  /*
     Construct 1d vector of size (nrow*ncol) from 2d matrix of
     dimension "nrow by ncol" - output vector in row major form :
     (row_1, row_2, ..., row_n)
  */

  template<typename T>
  void construct_1D_from_2D(std::vector<std::vector<T> > &D2_in, std::vector<T> &D1_out)
  {
    const int nrow = D2_in[0].size();
    const int ncol = D2_in.size();
    for (size_t i=0; i<nrow; i++) {
      for (size_t j=0; j<ncol; j++) {
        //D1_out.insert(D1_out.end(), &D2_in[i][0], &D2_in[i][ncol-1] );
        D1_out.push_back(D2_in[i][j]);
      }
    }


  }

  //===========================================================
  /*
     Construct 1d vector of size (nrow*ncol) from 2d matrix of
     dimension "nrow by ncol" - output vector in row major form :
     (row_1, row_2, ..., row_n)
  */
  template<typename T>
  void construct_1D_from_2D ( int nrow, int ncol,
    std::vector<std::vector<T> > &D2_in, std::vector<T> &D1_out) {

    std::vector<T> D1_out_buf;
    int count=0;
    for (size_t i=0; i<nrow; i++) {
      for (size_t j=0; j<ncol; j++) {
        //D1_out.insert(D1_out.end(), &D2_in[i][0], &D2_in[i][ncol-1] );
        D1_out_buf.push_back(D2_in[i][j]);
        count++;
      }
    }
    D1_out = D1_out_buf;

  }

  //===========================================================
  // Convert real matrix (from LAPACK) to comlex vector
  void convert_Real_to_Complex_Vector(std::vector<bool> &real_comlex_eigval_flag ,
     std::vector<double> &vec_real,
      std::vector<std::complex<double> > &vec_complex);




}



#endif  //end of header guard
