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



//FR
#include "tools.hpp"
#include "Tines.hpp"


// parse a string
void CSP::parseString(std::string &my_string, // in
                 std::string &delimiter, // in
                 std::vector<int>& my_values// out
               ){
//my_string => "1,2,3"
// delimiter=> ,
// my_values => 1 2 3
size_t pos = 0;
std::string token;
while ((pos = my_string.find(delimiter)) != std::string::npos) {
  token = my_string.substr(0, pos);
  my_values.push_back(std::stoi(token));
  my_string.erase(0, pos + delimiter.length());
}
my_values.push_back(std::stoi(my_string));

}

// matrix vector multiplication A*b = c using std vectors
void CSP::MatrixVectorMul(std::vector<std::vector<double> >&  A,
                         std::vector<double>& b, std::vector<double>& c)
    {

  int nrow_A = (int) A.size();
  int nrow_B = (int) b.size();

  for (int i=0; i<nrow_A; i++) { // loop over rows
      c[i] = 0.0;
      for (int j=0; j<nrow_B; j++){
        c[i] += A[i][j] * b[j];
      }
  }


}


void CSP::convert_Real_to_Complex_Vector(std::vector<bool> &real_comlex_eigval_flag ,
   std::vector<double> &vec_real,
    std::vector<std::complex<double> > &vec_complex)
{

  int dim = real_comlex_eigval_flag.size();
  int nrow = dim;
  int ncol = dim;
  vec_complex = std::vector<std::complex<double> >(dim*dim);

  for (size_t j=0; j<ncol; j++ ) {

    if (real_comlex_eigval_flag[j] == false) {
      for (size_t i=0; i<nrow; i++ )
        vec_complex[i*ncol+j]= std::complex<double>(vec_real[i*ncol+j], 0.0);
    } else {

      for (size_t i=0; i<nrow; i++ ) {
        vec_complex[i*ncol+j]     = std::complex<double>(vec_real[i*ncol+j],  vec_real[i*ncol+(j+1)]);
        vec_complex[i*ncol+(j+1)] = std::complex<double>(vec_real[i*ncol+j], -vec_real[i*ncol+(j+1)] );
      }

      j++;
      continue;
    }
  }


}

//-----------------------------------------------------------
/*
  Solution to a real system of linear equations,
    matA * mat_X = mat_B,
  where mat_X is the solution matrix. 1d interface.

  This function overwrite matA_1d and matX_1d.
*/
int CSP::LinearSystemSolve(
	       int nrowA, int ncolA,
	       int nrowB, int ncolB,
               std::vector<double> &matA_1d,
               std::vector<double> &matB_1d,
               std::vector<double> &matX_1d)
{

//#ifdef DEBUG_MODE
  std::cout<< "From : " <<__func__ <<" : "<< __FILE__<<" : "  << __LINE__ <<std::endl;
  std::cout<< "nrowA, ncolA   =" << nrowA <<", " << ncolA <<std::endl;
  std::cout<< "nrowB, ncolB   =" << nrowB <<", " << ncolB <<std::endl;
  std::cout<< "Size of matA_1d   =" << matA_1d.size() <<std::endl;
  std::cout<< "Size of matB_1d   =" << matB_1d.size() <<std::endl;
  std::cout<< "\n" <<std::endl;
//#endif

  struct timeval begin1, end1;
  printf("\n --> Tines LinearSolve :\n");
  {
    int wlen(0);
    Tines::SolveLinearSystem_WorkSpaceHostTPL(nrowA, ncolA, ncolB, wlen);

    std::vector<double> work(wlen);
    gettimeofday( &begin1, NULL );
    {
      int matrix_rank(0);
      Tines::SolveLinearSystem_HostTPL(nrowA, ncolA, ncolB,
				       (double*)&matA_1d[0], ncolA, 1,
				       (double*)&matX_1d[0], ncolB, 1,
				       (double*)&matB_1d[0], ncolB, 1,
				       (double*)&work[0], wlen,
				       matrix_rank);
    }
      gettimeofday( &end1, NULL );
  }
  double wall_time = 1.0 * ( end1.tv_sec - begin1.tv_sec ) +
                 1.0e-6 * ( end1.tv_usec - begin1.tv_usec );

  printf("\tWall time for Tines LinearSolve : %f sec\n",wall_time);

  fflush( stdout );
  return 0;
} // end of LinearSystemSolve // 1d interface


//-----------------------------------------------------------
// Matrix input in 2d std::vector
int CSP::LinearSystemSolve(
	       int nrowA, int ncolA,
	       int nrowB, int ncolB,
               std::vector<std::vector<double> > &matA_2d,
               std::vector<std::vector<double> > &matB_2d,
               std::vector<std::vector<double> > &matX_2d)
{

  std::vector<double> matA_1d;
  std::vector<double> matB_1d;
  std::vector<double> matX_1d;

  // output linear matrix will be in row major form:
  CSP::construct_1D_from_2D<double> ( nrowA, ncolA, matA_2d, matA_1d);
  CSP::construct_1D_from_2D<double> ( nrowB, ncolB, matB_2d, matB_1d);

  LinearSystemSolve(
	       nrowA, ncolA,
	       nrowB, ncolB,
               matA_1d,
               matB_1d,
               matX_1d);

  CSP::construct_2D_from_1D<double>(nrowA, ncolB, matX_1d, matX_2d);

  return 0;
} // end of LinearSystemSolve // 2d interface
