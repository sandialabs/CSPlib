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


#ifndef _UTIL_SEEN_
#define _UTIL_SEEN_

#include <iostream>
#include <cstdio>
#include <vector>
#include <string>
#include <complex>
#include <cmath>
#include <iomanip>

#if 1
enum  OutputFormat {
  Out1d,
  O1d,
  Out2d,
  O2d
};

enum Layout {
  RightIndexFirst,
  RIF,
  LeftIndexFirst,
  LIF,
  RowMajorMatrix,
  RowMajorMat,
  //RowMajor,
  RowMaj,
  RM,
  ColMajorMatrix,
  ColMajorMat,
  //ColMajor,
  ColMaj,
  CM,
  Contiguous,
  Cont,
  Row,
  Column,
  Col
};

enum DataType {
  Dbl,   // double
  Int,   // int
  Cmplx  // complex
};

#endif

namespace Util {
/*
 Utilities for printing matix and vector in different way:
*/
namespace Print {

// Printing out a given matrix in matrix format -
template <typename T>
void mat( std::string name_of_mat, Layout layout ,
          OutputFormat formt,
          DataType dtype,
          size_t nrow,
          size_t ncol,
          std::vector<std::vector<T> > &mat) {

  std::cout<<"\n";
  for (size_t i=0; i<name_of_mat.size(); i++ ) std::cout<< "=";
  std::cout<<"\n"<< name_of_mat;

  switch (layout)
  {
  case RightIndexFirst : case RIF :
    std::cout<< "( Right Index First )"<<"\n";
    for (size_t i=0; i<nrow; i++) {
      for (size_t j=0; j<ncol; j++)
        std::cout << std::scientific << std::setw(12) << std::setprecision(4) << mat[i][j] << "\t";
      printf("\n");
    }
    break;

  case LeftIndexFirst : case LIF :
    std::cout<< "( Left Index First )"<<"\n";
    for (size_t j=0; j<ncol; j++) {
      for (size_t i=0; i<nrow; i++)
        std::cout << std::scientific << std::setw(12) << std::setprecision(4) << mat[i][j] << "\t";
      printf("\n");
    }
    break;

  default : std::cout<< __func__ << ": Please select the layout of the matrix"<<"\n";

  }

} // end of Print::mat

// Printing out a given vector in matrix format
template <typename T>
void vec( std::string name_of_vec,
          Layout layout ,
          OutputFormat formt,
          DataType dtype,
          size_t nrow,
          size_t ncol,
          std::vector<T> &vec) {

  std::cout<<"\n";
  for (size_t i=0; i<name_of_vec.size(); i++ ) std::cout<< "=";
  std::cout<<"\n"<< name_of_vec;

  switch (layout)
  {
  case ColMajorMatrix : case ColMajorMat : case ColMaj : case CM :
    std::cout<< "( Vector read as Column Major format )"<<"\n";
    for (size_t i=0; i<nrow; i++) {
      for (size_t j=0; j<ncol; j++)
        std::cout << std::scientific << std::setw(10) << std::setprecision(2) << vec[nrow*j+i] << "\t";
      printf("\n");
    }
    break;

  case RowMajorMatrix : case RowMajorMat : case RowMaj : case RM :
    std::cout<< "( Vector read as Row Major format )"<<"\n";
    for (size_t i=0; i<nrow; i++) {
      for (size_t j=0; j<ncol; j++)
        std::cout << std::scientific << std::setw(10) << std::setprecision(2) << vec[ncol*i+j] << "\t";
      printf("\n");
    }
    break;

  case Contiguous : case Cont :
    std::cout<< "( Vector read contiguously )"<<"\n";
    for (size_t i=0; i<nrow*ncol; i++)
        std::cout << std::scientific << std::setw(10) << std::setprecision(2) << vec[i] << "\t";
    printf("\n");
    break;

  default : std::cout<< __func__ << ": Please select the layout of the vector representing a matrix"<<"\n";

  }
} // end of Print::vec for printing vector in matrix form


// Printing out a given vector in vector format
template <typename T>
void vec( std::string name_of_vec,
          Layout layout,
          OutputFormat formt,
          DataType dtype,
          size_t nrow,
          std::vector<T> &vec) {

  std::cout<<"\n";
  for (size_t i=0; i<name_of_vec.size(); i++ ) std::cout<< "=";
  std::cout<<"\n"<< name_of_vec;

  switch (layout)
  {
  case Column : case Col :
    std::cout<< "( 1-D Vector in column format )"<<"\n";
    for (size_t i=0; i<nrow; i++)
        std::cout << std::scientific << std::setw(10) << std::setprecision(2) << vec[i] << "\n";
    printf("\n");
    break;

  case Row :
    std::cout<< "( 1-D Vector in row format )"<<"\n";
    for (size_t i=0; i<nrow; i++)
        std::cout << std::scientific << std::setw(10) << std::setprecision(2) << vec[i] << "\t";
    printf("\n");
    break;

  default : std::cout<< __func__ << ": Please select required output format of the vector"<<"\n";

  }

}



} // namespace Print

namespace Math {

// Transpose a matrix given in vector form
template <typename T>
void transpose( std::string name_of_vec,
          Layout layout ,
          DataType dtype,
          size_t nrow,
          size_t ncol,
          std::vector<T> &vec_in,
          std::vector<T> &vec_out) {

  switch (layout)
  {
  case ColMajorMatrix : case ColMajorMat : case ColMaj : case CM :
    std::cout<< "( Vector was in Column Major format )"<<"\n";
    for (size_t i=0; i<nrow; i++) {
      for (size_t j=0; j<ncol; j++) {
          vec_out.push_back( vec_in[nrow*j+i] );
      }
    }
    break;

  case RowMajorMatrix : case RowMajorMat : case RowMaj : case RM :
    std::cout<< "( Vector was in Row Major format )"<<"\n";
    for (size_t i=0; i<nrow; i++) {
      for (size_t j=0; j<ncol; j++) {
          vec_out.push_back( vec_in[ncol*i+j] );
      }
    }
    break;

  default : std::cout<< __func__ << ": Please select the layout of the vector representing a matrix"<<"\n";

  } //switch
}



}

} // namespace Util

#endif
