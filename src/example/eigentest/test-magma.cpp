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


#if defined (TEST_MKL)
#include "mkl.h"
#endif

/// Kokkos headers
#include "Kokkos_Core.hpp"
#include "Kokkos_Timer.hpp"
#include "Kokkos_Random.hpp"

/// KokkosKernels headers
#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_Eigendecomposition_Serial_Internal.hpp" 

typedef Kokkos::DefaultExecutionSpace SpT;
typedef Kokkos::DefaultHostExecutionSpace HpT;
typedef double value_type;

///  We only compare with MKL as MAGMA is not really converted as a pure GPU implementation.
///  MAGMA adopts an hybrid algorithm where Hessenberg reduction is computed on GPU and the
///  Francis algorithm is computed on CPU.
int main(int argc, char* argv[]) {

  Kokkos::initialize(argc, argv);
  {
    Kokkos::print_configuration(std::cout);                                                       

    typedef Kokkos::Details::ArithTraits<value_type> ats;
    Kokkos::Timer timer;
    
    /// input arguments parsing
    int N = 1e4; /// # of problems (batch size)
    int Blk = 100; /// dimension of the problem
    for (int i=1;i<argc;++i) {
      const std::string& token = argv[i];
      if (token == std::string("-N")) N = std::atoi(argv[++i]);
      if (token == std::string("-B")) Blk = std::atoi(argv[++i]);
    }
    
    printf(" :::: Testing Eigenvalue solver (N = %d, Blk = %d)\n", N, Blk);
    const value_type zero(0);

    /// N problems
    Kokkos::View<value_type***, HpT> A_mkl ("A_mkl",  N, Blk, Blk); 
    Kokkos::View<value_type***, SpT> A_kk  ("A_kk",   N, Blk, Blk); 
    Kokkos::View<value_type***, HpT> A_h   ("A_h",    N, Blk, Blk); /// host problem backup for validation
    Kokkos::View<value_type***, SpT> A_d   ("A_d",    N, Blk, Blk); /// device problem backup for validation
      
    /// Eigenvalues: ith problem real eigenvalues E(i,0,ALL), imag eigenvalues E(i,1,ALL)
    /// - MKL should use layout two contiguous vectors for real and imag.
    /// - KokkosKernels does not require a specific layout but here we follow the MKL storage format.
    Kokkos::View<value_type***, Kokkos::LayoutRight, HpT> E_mkl("E_mkl", N, 2, Blk);
    Kokkos::View<value_type***, Kokkos::LayoutRight, SpT> E_kk ("E_kk",  N, 2, Blk);

    /// Eigenvectors: left eigenvectors V(i,0,ALL,ALL), right eigenvectors V(i,1,ALL,ALL)
    Kokkos::View<value_type****, Kokkos::LayoutRight, HpT> V_mkl("V_mkl", N, 2, Blk, Blk); 
    Kokkos::View<value_type****, SpT>                      V_kk ("V_kk",  N, 2, Blk, Blk); 

    /// Workspace
#if defined (TEST_MKL)
    /// define work space required from mkl
    int lwork_mkl = -1;
    {    
      double work_query;
      LAPACKE_dgeev_work(LAPACK_ROW_MAJOR, 
                         'V', 'V', 
                         Blk, 
                         (double*)&A(0,0,0), Blk,
                         &E_mkl(0,0,0), &E_mkl(0,1,0),
                         NULL, Blk,
                         NULL, Blk,
                         &work_query,
                         lwork_mkl);
      lwork_mkl = int(work_query);
    }
    Kokkos::View<value_type**, Kokkos::LayoutRight, SpT> W_mkl("W_mkl", N, lwork_mkl); /// workspace for mkl
#endif
    /// for now, use work space for every batch instance, later we can use memory pool or team policy shared memory
    const int wlen = 2*Blk*Blk + Blk*5; 
    Kokkos::View<value_type**, Kokkos::LayoutRight, SpT> W_kk("W_kk", N, wlen); // workspace for kokkos

    /// subview pattern to extract strides information
    /// here we do not use high level interfacie through kokkos view but pointer interface
    auto aa_mkl = Kokkos::subview(A_mkl, 0, Kokkos::ALL(), Kokkos::ALL());
    auto aa_kk  = Kokkos::subview(A_kk,  0, Kokkos::ALL(), Kokkos::ALL());

    auto ee_mkl = Kokkos::subview(E_mkl, 0, 0, Kokkos::ALL());
    auto ee_kk  = Kokkos::subview(E_kk,  0, 0, Kokkos::ALL());

    auto vv_mkl = Kokkos::subview(V_mkl,  0, 0, Kokkos::ALL(), Kokkos::ALL());
    auto vv_kk  = Kokkos::subview(V_kk,   0, 0, Kokkos::ALL(), Kokkos::ALL());

#if defined(TEST_MKL)    
    const int mkl_as0 = aa_mkl.stride(0), mkl_as1 = aa_mkl.stride(1);
#endif     
    const int  kk_as0 = aa_kk.stride(0),   kk_as1 = aa_kk.stride(1);

#if defined(TEST_MKL)    
    const int mkl_es = ee_mkl.stride(0);
#endif     
    const int  kk_es = ee_kk.stride(0);

#if defined(TEST_MKL)    
    const int mkl_vs0 = vv_mkl.stride(0), mkl_vs1 = vv_mkl.stride(1);
#endif     
    const int  kk_vs0 = vv_kk.stride(0),   kk_vs1 = vv_kk.stride(1);

    /// randomize input matrices
    Kokkos::Random_XorShift64_Pool<SpT> random(13245);
    Kokkos::fill_random(A_h, random, value_type(1.0));
    Kokkos::deep_copy(A_d, A_h);

    typedef Kokkos::Schedule<Kokkos::Static> ScheduleType;
    Kokkos::RangePolicy<SpT,ScheduleType> policy(0, N);

    /// timing
#if defined (TEST_MKL)
    double t_mkl(0);
#endif
    double t_kk(0);

    const int niter_beg = -2, niter_end = 3;

    ///
    /// MKL dgeev
    ///
#if defined (TEST_MKL)
    for (int iter=niter_beg;iter<niter_end;++iter) {
      Kokkos::deep_copy(A_mkl, A_h);
      Kokkos::deep_copy(E_mkl, zero);
      Kokkos::deep_copy(V_mkl, zero);

      timer.reset();
      Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int i) {
          LAPACKE_dgeev_work(LAPACK_ROW_MAJOR, 
                             'V', 'V', 
                             Blk, 
                             (double*)&A_mkl(i,0,0), mkl_as0,
                             &E_mkl(i,0,0), &E_mkl(i,1,0),
                             &V_mkl(i,0,0,0), mkl_vs0,
                             &V_mkl(i,1,0,0), mkl_vs0,
                             &W_mkl(i,0), lwork_mkl);
        });
      Kokkos::fence();
      t_mkl += (iter >= 0)*timer.seconds(); 
    }
    printf("MKL           Eigensolver Per Problem Time: %e seconds\n", (t_mkl/double(niter_end*N)));
#endif 

    ///
    /// KokkosKernels Testing
    ///
    for (int iter=niter_beg;iter<niter_end;++iter) {
      Kokkos::deep_copy(A_kk, A_d);
      Kokkos::deep_copy(E_kk, zero);
      Kokkos::deep_copy(V_kk, zero);
      
      timer.reset();
      Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int i) {
          const int r_val = KokkosBatched::
            SerialEigendecompositionInternal::invoke(Blk,
                                                     &A_kk(i,0,0), kk_as0, kk_as1,
                                                     &E_kk(i,0,0), kk_es, // real eigenvalues
                                                     &E_kk(i,0,1), kk_es, // imag eigenvalues
                                                     &V_kk(i,0,0,0), kk_vs0, kk_vs1, // left eigenvectors
                                                     &V_kk(i,1,0,0), kk_vs0, kk_vs1, // right eigenvectors
                                                     &W_kk(i,0), wlen);
        });
      Kokkos::fence();
      t_kk += (iter >= 0)*timer.seconds(); 
    }
    printf("KokkosBatched Eigensolver Per Problem Time: %e seconds\n", (t_kk/double(niter_end*N)));
  }
  Kokkos::finalize();
  
  return 0;
}
