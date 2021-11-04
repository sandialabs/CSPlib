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


#ifndef __TEST_MATRIX_HPP__
#define __TEST_MATRIX_HPP__
#include <fstream>
#include "Kokkos_Core.hpp"
#include "Kokkos_Random.hpp"

namespace TestCSP {
  struct TestMatrix {
    using A_value_type_3d_view_kokkos = Kokkos::View<value_type***, SpT>;  
    using A_value_type_3d_view_host   = Kokkos::View<value_type***, Kokkos::LayoutRight,HpT>;  
    
    int _N, _Blk;

    A_value_type_3d_view_kokkos _A_kokkos;
    A_value_type_3d_view_host _A_host;

    void setRandomMatrix(const int N, const int Blk) {
      _N = N;
      _Blk = Blk; 

      _A_kokkos = A_value_type_3d_view_kokkos("A_mat_kokkos", N, Blk, Blk);
      _A_host    = A_value_type_3d_view_host ("A_mat_host",   N, Blk, Blk);

      const value_type one(1.0);
      Kokkos::Random_XorShift64_Pool<HpT> random(13245);      
      Kokkos::fill_random(_A_host, random, one);

      auto A_kokkos_host = Kokkos::create_mirror_view(Kokkos::HostSpace(), _A_kokkos);
      Kokkos::RangePolicy<HpT> policy(0, _N);
      Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int i) {
          for (int j=0,jend=A_kokkos_host.extent(1);j<jend;++j) 
            for (int k=0,kend=A_kokkos_host.extent(2);k<kend;++k)
              A_kokkos_host(i,j,k) = _A_host(i,j,k);
        });
      Kokkos::deep_copy(_A_kokkos, A_kokkos_host);
    }

    void setMatrixFromFile(const char *name, const int N = -1, const int Blk = -1) {
      A_value_type_3d_view_host A_tmp;
      {
        bool reading_matrix = false;
        int count = 0, nrows = 0, ncols = 0;
        std::ifstream infile(name);      
        if (!infile.is_open()) {
          printf("Error: opening file %s\n", name);
        }
        for (std::string line;getline(infile, line);) {
          std::vector<std::string> content;
          std::string buf;
          std::stringstream ss(line);
          while (ss >> buf)
            content.push_back(buf);
          if (content.size() > 1) { // start of row matrix
            reading_matrix = true;          
            ncols = content.size();
            ++nrows;
          } else {
            if (reading_matrix == true) {
              ++count;
              nrows = 0;
            }
            reading_matrix = false;
          }
        }
        printf("# of Jacobian matrices = %d, nrows = %d, ncols = %d\n", count, ncols, ncols);
        A_tmp = A_value_type_3d_view_host("A_mat_tmp", count, ncols, ncols);

        _N = N < 0 ? count : N;
        _Blk = ncols;        
        printf("# of testing matrices = %d, nrows = %d, ncols = %d\n", _N, _Blk, _Blk);
      }

      {
        bool reading_matrix = false;
        int count = 0, nrows = 0;
        std::ifstream infile(name);      
        if (!infile.is_open()) {
          printf("Error: opening file %s\n", name);
        }
        for (std::string line;getline(infile, line);) {
          std::vector<std::string> content;
          std::string buf;
          std::stringstream ss(line);
          while (ss >> buf)
            content.push_back(buf);
          if (content.size() > 1) { // start of row matrix
            reading_matrix = true;          
            for (int j=0,jend=content.size();j<jend;++j) 
              A_tmp(count, nrows, j) = std::stod(content.at(j));
            ++nrows;
          } else {
            if (reading_matrix == true) {
              ++count;
              nrows = 0;
            }
            reading_matrix = false;
          }
        }
#if 0 // debugging print
        for (int k=0,kend=A_tmp.extent(0);k<kend;++k) {
          printf("k = %d\n", k);
          for (int i=0,iend=A_tmp.extent(1);i<iend;++i) {
            for (int j=0,jend=A_tmp.extent(2);j<jend;++j)
              printf(" %e ", A_tmp(k,i,j));
            printf("\n");
          }
        }
#endif
      }

      {
        _A_kokkos = A_value_type_3d_view_kokkos("A_mat_kokkos", _N, _Blk, _Blk);
        _A_host   = A_value_type_3d_view_host  ("A_mat_host",   _N, _Blk, _Blk);

        const int M = A_tmp.extent(0);
        for (int k=0;k<_N;++k) {
          const int kk = k%M; // tmp index
          for (int i=0;i<_Blk;++i)
            for (int j=0;j<_Blk;++j)
              _A_host(k,i,j) = A_tmp(kk,i,j);
        }

#if 0 // debugging print
        for (int k=0,kend=_A_host.extent(0);k<kend;++k) {
          printf("k = %d\n", k);
          for (int i=0,iend=_A_host.extent(1);i<iend;++i) {
            for (int j=0,jend=_A_host.extent(2);j<jend;++j)
              printf(" %e ", _A_host(k,i,j));
            printf("\n");
          }
        }
#endif

        auto A_kokkos_host = Kokkos::create_mirror_view(Kokkos::HostSpace(), _A_kokkos);
        Kokkos::RangePolicy<HpT> policy(0, _N);
        Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int i) {
            for (int j=0,jend=A_kokkos_host.extent(1);j<jend;++j) 
              for (int k=0,kend=A_kokkos_host.extent(2);k<kend;++k)
                A_kokkos_host(i,j,k) = _A_host(i,j,k);
          });
        Kokkos::deep_copy(_A_kokkos, A_kokkos_host);
      }
    }
    
    int getBatchsize() const { return _N; }
    int getBlocksize() const { return _Blk; }

    A_value_type_3d_view_kokkos getProblemKokkos() const { return _A_kokkos; }
    A_value_type_3d_view_host   getProblemHost() const { return _A_host; }
  };
}

#endif
