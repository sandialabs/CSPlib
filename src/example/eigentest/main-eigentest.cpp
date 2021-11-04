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


//#define TEST_MKL
#define TEST_CHECK
/// Kokkos headers
#include "Kokkos_Core.hpp"

typedef Kokkos::DefaultExecutionSpace SpT;
typedef Kokkos::DefaultHostExecutionSpace HpT;
typedef double value_type;

#include "test-matrix.hpp"

#if defined (TEST_MKL)
#include "test-mkl.hpp"
#endif

#if defined (TEST_MAGMA)
#include "test-magma.hpp"
#endif

#include "test-kokkos.hpp"

#if defined (TEST_CHECK)
#include "test-check.hpp"
#endif

int main(int argc, char* argv[]) {

  Kokkos::initialize(argc, argv);
  {
    Kokkos::print_configuration(std::cout);

    /// input arguments parsing
    int N = -1; //1e3; /// # of problems (batch size)
    int Blk = -1; //; /// dimension of the problem
    char *filename = NULL;
    double tol = 1e-6;
    for (int i=1;i<argc;++i) {
      const std::string& token = argv[i];
      if (token == std::string("-N")) N = std::atoi(argv[++i]);
      if (token == std::string("-B")) Blk = std::atoi(argv[++i]);
      if (token == std::string("-F")) filename = argv[++i];
      if (token == std::string("-T")) tol = std::atof(argv[++i]);
    }
    const int niter_beg = -2, niter_end = 3;

    ///
    /// Create an array of matrices (random or from file)
    /// - It stores the matrices in two formats
    ///   - Host   : layour right
    ///   - Kokkos : polymorphic layout according to execution space
    ///
    TestCSP::TestMatrix problem;
    if (filename == NULL) {
      N = N < 0 ? 1e3 : N;
      Blk = Blk < 0 ? 10 : Blk;
      problem.setRandomMatrix(N, Blk);
    } else {
      problem.setMatrixFromFile(filename, N, Blk);
    }
    printf(" :::: Testing Eigenvalue solver (N = %d, Blk = %d)\n", problem.getBatchsize(), problem.getBlocksize());    

#if defined (TEST_MKL)
    {
      /// 
      /// MKL can use layout right (row major format)
      ///
      TestCSP::TestMKL eig_mkl(problem.getBatchsize(), problem.getBlocksize());
      double t_mkl(0);
      for (int iter=niter_beg;iter<niter_end;++iter) {    
        eig_mkl.setProblem(problem.getProblemHost());
        const double t = eig_mkl.runTest();
        t_mkl += (iter >= 0)*t;
      }
      printf("MKL Test\n");
      printf("========\n");
#if defined (TEST_CHECK)
      TestCSP::TestCheck check(problem.getBatchsize(), 
                               problem.getBlocksize(),
                               problem.getProblemHost(),
                               eig_mkl._E,
                               eig_mkl._V,
                               true);
      const auto pass = check.checkTest(tol);
      printf("MKL           Eigensolver left  test %s with a tol %e\n", (pass.first  ? "passed" : "fail"), tol);
      printf("MKL           Eigensolver right test %s with a tol %e\n", (pass.second ? "passed" : "fail"), tol);
#endif
      const double t_mkl_per_problem = (t_mkl/double(niter_end*problem.getBatchsize()));
      const double n_mkl_problems_per_second = 1.0/t_mkl_per_problem;
      printf("MKL           Eigensolver Time: %e seconds , %e seconds per problem , %e problems per second\n", t_mkl, t_mkl_per_problem, n_mkl_problems_per_second);


    }
#endif

#if defined (TEST_MAGMA)
    {
      ///
      /// Magma uses column major format only 
      /// - Magma uses an hybrid algorithm: Hessenberg reduction on a device and QR iterations are on host
      /// - Matrices on host are interfaced to Magma
      /// - transpose input matrix 
      /// - transpose eigen vectors
      TestCSP::TestMagma eig_magma(problem.getBatchsize(), problem.getBlocksize());
      double t_magma(0);
      for (int iter=niter_beg;iter<niter_end;++iter) {    
        eig_magma.setProblem(problem.getProblemHost());
        const double t = eig_magma.runTest();
        t_magma += (iter >= 0)*t;
      }
      printf("Magma Test\n");
      printf("==========\n");
#if defined (TEST_CHECK)
      eig_magma.postUpdate();
      TestCSP::TestCheck check(problem.getBatchsize(), 
                               problem.getBlocksize(),
                               problem.getProblemHost(),
                               eig_magma._E,
                               eig_magma._V,
                               true);
      const auto pass = check.checkTest(tol);
      printf("Magma         Eigensolver left  test %s with a tol %e\n", (pass.first  ? "passed" : "fail"), tol);
      printf("Magma         Eigensolver right test %s with a tol %e\n", (pass.second ? "passed" : "fail"), tol);
#endif
      const double t_magma_per_problem = (t_magma/double(niter_end*problem.getBatchsize()));
      const double n_magma_problems_per_second = 1.0/t_magma_per_problem;
      printf("Magma         Eigensolver Time: %e seconds , %e seconds per problem , %e problems per second\n", t_magma, t_magma_per_problem, n_magma_problems_per_second);
    }
#endif

    {
      TestCSP::TestKokkos eig_kk(problem.getBatchsize(), problem.getBlocksize());
      double t_kk(0); 
      for (int iter=niter_beg;iter<niter_end;++iter) {          
        eig_kk.setProblem(problem.getProblemKokkos());
        const double t = eig_kk.runTest();
        t_kk += (iter >= 0)*t;
      }
      printf("KokkosBatched Test\n");
      printf("==================\n");
#if defined (TEST_CHECK)
      TestCSP::TestCheck check(problem.getBatchsize(), 
                               problem.getBlocksize(),
                               problem.getProblemKokkos(),
                               eig_kk._E,
                               eig_kk._V,
                               true);
      const auto pass = check.checkTest(tol);
      printf("KokkosBatched Eigensolver left  test %s with a tol %e\n", (pass.first  ? "passed" : "fail"), tol);
      printf("KokkosBatched Eigensolver right test %s with a tol %e\n", (pass.second ? "passed" : "fail"), tol);
#endif
      const double t_kk_per_problem = (t_kk/double(niter_end*problem.getBatchsize()));
      const double n_kk_problems_per_second = 1.0/t_kk_per_problem;
      printf("KokkosBatched Eigensolver Time: %e seconds , %e seconds per problem , %e problems per second\n", t_kk, t_kk_per_problem, n_kk_problems_per_second);
    }

  }
  Kokkos::finalize();
  
  return 0;
}
