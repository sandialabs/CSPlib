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


#ifndef INDEX_BATCH_CSP
#define INDEX_BATCH_CSP
#include "Tines.hpp"
#include "util.hpp"
#include "index_kokkos.hpp"
#include "CSPlib_ConfigDefs.h"

class CSPIndexBatch
{

 private:
   using real_type = typename Tines::ats<double>::scalar_type;

   using host_exec_space = Kokkos::DefaultHostExecutionSpace;
   using exec_space = Kokkos::DefaultExecutionSpace;
   using host_device_type = typename Tines::UseThisDevice<host_exec_space>::type;
   using device_type      = typename Tines::UseThisDevice<exec_space>::type;

   using policy_type = Kokkos::TeamPolicy<exec_space>;

   using index_csplib_host = CSP::IndexComputation<host_device_type>;
   using index_csplib_device = CSP::IndexComputation<device_type>;

   using real_type_1d_view = Tines::value_type_1d_view<real_type,device_type>;
   using real_type_2d_view = Tines::value_type_2d_view<real_type,device_type>;
   using real_type_3d_view = Tines::value_type_3d_view<real_type,device_type>;

   using ordinal_type_1d_view = Tines::value_type_1d_view<int,device_type>;

   using real_type_1d_view_host = typename real_type_1d_view::HostMirror;
   using real_type_2d_view_host = typename real_type_2d_view::HostMirror;
   using real_type_3d_view_host = typename real_type_3d_view::HostMirror;

   template<typename ViewType>
    struct DualViewType {
      ViewType _dev;
      typename ViewType::HostMirror _host;
    };

    using real_type_1d_dual_view = DualViewType<real_type_1d_view>;
    using real_type_2d_dual_view = DualViewType<real_type_2d_view>;
    using real_type_3d_dual_view = DualViewType<real_type_3d_view>;

   real_type_3d_view _A, _B, _S; //inputs
   ordinal_type_1d_view _M; //inputs
   real_type_2d_view _RoP; //inputs
   real_type_3d_view _Beta, _Alpha, _Gamma; // work
   real_type_2d_view  _work; // work
   real_type_3d_dual_view _ParticipationIndex, _FastImportanceIndex, _SlowImportanceIndex ; //outputs
   int _nBatch, _n_variables, _n_processes;
   int _FastImportanceIndex_need_sync, _SlowImportanceIndex_need_sync, _ParticipationIndex_need_sync;

#if defined(CSPLIB_MESUARE_WALL_TIME)
    Kokkos::Timer timer;
    FILE* fs;
#endif

   enum {
      NeedSyncToDevice = 1,
      NeedSyncToHost = -1,
      NoNeedSync = 0
    };

 public:
   // create class before loop
  CSPIndexBatch(
         const real_type_3d_view& A,
         const real_type_3d_view& B,
         const real_type_3d_view& S,
         const real_type_2d_view& RoP,
         const ordinal_type_1d_view& M
        ) :
         _A(A),
         _B(B),
         _S(S),
         _M(M),
         _RoP(RoP),
         _nBatch(A.extent(0)),
         _n_variables(A.extent(1)),
         _n_processes(_S.extent(2))
          {
            printf("_nBatch %d, _n_variables %d, _n_processes %d\n", _nBatch, _n_variables, _n_processes );
#if defined(CSPLIB_MESUARE_WALL_TIME)
            fs = fopen("CSPLIB_IndexKokkos.out", "w");
            fprintf(fs, "{\n");
            fprintf(fs, "%s: %d, \n","\"Number of state vectors\"", _nBatch);
            fprintf(fs, "\"Index Class\": \n {\n");
#endif
          }

 ~CSPIndexBatch();

void evalBeta(const ordinal_type& team_size=ordinal_type(-1),
              const ordinal_type& vector_size=ordinal_type(-1));
void evalAlpha(const ordinal_type& team_size=ordinal_type(-1),
               const ordinal_type& vector_size=ordinal_type(-1));
void evalGamma(const ordinal_type& team_size=ordinal_type(-1),
               const ordinal_type& vector_size=ordinal_type(-1));
void evalParticipationIndex(const ordinal_type& team_size=ordinal_type(-1),
                            const ordinal_type& vector_size=ordinal_type(-1));
void createBetaView();
void createAlphaView();
void createGammaView();
void createWorkView();
void createParticipationIndexView();
void createSlowImportanceIndexView();
void evalImportanceIndexSlow(const ordinal_type& team_size=ordinal_type(-1),
                             const ordinal_type& vector_size=ordinal_type(-1));
void createFastImportantIndexView();
void evalImportanceIndexFast(const ordinal_type& team_size=ordinal_type(-1),
                             const ordinal_type& vector_size=ordinal_type(-1));
void freeSlowImportanceIndexView();
void freeFastImportantIndexView();
void freeParticipationIndexView();
void getImportanceIndexSlow(
     std::vector<std::vector< std::vector<double> > > &Islow_jk );
void getImportanceIndexFast(
     std::vector<std::vector< std::vector< double > > > &Ifast );
void getParticipationIndex(
     std::vector<std::vector< std::vector< double > > > &PartIndex );

//
real_type_3d_view_host getParticipationIndex();
real_type_3d_view_host getImportanceIndexSlow();
real_type_3d_view_host getImportanceIndexFast();
void freeBetaView();
void freeAlphaView();
void freeGammaView();
void freeWorkView();


}; // end of CSPIndex
#endif  //end of header guard
