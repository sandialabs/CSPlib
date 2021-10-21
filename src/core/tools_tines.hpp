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


#ifndef TOOLS_TINES
#define TOOLS_TINES

#include "Tines.hpp"

template<typename SpT>
struct ComputeReducedJacobian{

public:

  using value_type_2d_view = Kokkos::View<double**, Kokkos::LayoutRight,SpT>;
  using value_type_1d_view = Kokkos::View<double*, Kokkos::LayoutRight,SpT>;

  inline
  static int getWorkSpaceSize(const int nEqns, const int nConstrains  )
  {
    const auto jacComp_workspace_size =  nEqns*nEqns +
      3*nEqns*nConstrains + nConstrains*nConstrains;
    //A [m,n] => nConstrainsxnConstrains
    //B[m,rhs] => nConstrains x nEqns
    // X[n,rhs]
    const int m = nConstrains, n = m;
    const int WSNewton = m * n + n * n + 2*(m < n ? m : n) + n * nEqns;
    int worksize = WSNewton + jacComp_workspace_size;

    return worksize;  }

  template<typename RealType3DViewType>
  inline
  static void runBatch(const RealType3DViewType &Jac,
                       const RealType3DViewType &redJac,
                       const int nEqns,
                       const int nConstrains) {
    //

    const int N   = Jac.extent(0);
    using policy_type = Kokkos::TeamPolicy<SpT>;
    policy_type policy(N, Kokkos::AUTO());
    const int level = 1;
    const int per_team_extent = getWorkSpaceSize(nEqns, nConstrains);

    const int per_team_scratch =
      Tines::ScratchViewType<value_type_1d_view>::shmem_size(per_team_extent);

    policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

    Kokkos::parallel_for(
			 "CSP::EigendecompositionKokkos:runBatch",
			 policy,
			 KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
			   const int i = member.league_rank();

			   const value_type_2d_view jac_at_i =
			     Kokkos::subview(Jac, i, Kokkos::ALL(), Kokkos::ALL());
			   //
			   const value_type_2d_view redJac_at_i =
			     Kokkos::subview(redJac, i, Kokkos::ALL(), Kokkos::ALL());
			   //
			  Tines::ScratchViewType<value_type_1d_view> work(member.team_scratch(level),
							    per_team_extent);

			   team_invoke(member, jac_at_i, nEqns, nConstrains, work, redJac_at_i);


			 });

  }
  template<typename MemberType,
           typename RealType1DViewType,
           typename RealType2DViewType>
  KOKKOS_INLINE_FUNCTION static void
  team_invoke(const MemberType& member,
	      //inputs
	      const RealType2DViewType& Jac,
	      const int nEqns,
	      const int nConstrains,
	      //workspaces
	      const RealType1DViewType& work,
	      //output
	      const RealType2DViewType& redJac)
  {

    // auto nEqns = kmcd.nSpec + 3;
    auto wptr = work.data();
    auto gu   = RealType2DViewType(wptr, nEqns, nEqns);
    wptr+=nEqns*nEqns;
    auto gv   = RealType2DViewType(wptr, nEqns, nConstrains);
    wptr+=nEqns*nConstrains;
    auto fu   = RealType2DViewType(wptr, nConstrains, nEqns);
    wptr+=nEqns*nConstrains;
    auto fv   = RealType2DViewType(wptr, nConstrains, nConstrains);
    wptr+=nConstrains*nConstrains;
    auto vu   = RealType2DViewType(wptr, nConstrains, nEqns);
    wptr+=nEqns*nConstrains;

    int wlen;
    Tines::SolveLinearSystem::workspace(fv, fu, wlen);

    auto work_linear_solver = RealType1DViewType(wptr, wlen);
    wptr+=wlen;

    Kokkos::parallel_for(
			 Kokkos::TeamThreadRange(member, nEqns), [&](const int& k) {
			   Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, nEqns),
						[&](const int& i) {
						  gu(k , i) = Jac(k,i);
						});
			 });

    Kokkos::parallel_for(
			 Kokkos::TeamThreadRange(member,nEqns), [&](const int& k) {
			   Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, nConstrains),
						[&](const int& i) {
						  gv(k , i) = Jac(k, i + nEqns);
						});
			 });

    Kokkos::parallel_for(
			 Kokkos::TeamThreadRange(member, nConstrains), [&](const int& k) {
			   Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, nEqns),
						[&](const int& i) {
						  //negative sign is needed, check notes on DAE
						  fu(k , i) = -Jac(k + nEqns , i );
						});
			 });

    Kokkos::parallel_for(
			 Kokkos::TeamThreadRange(member, nConstrains), [&](const int& k) {
			   Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, nConstrains),
						[&](const int& i) {
						  fv(k , i) = Jac(k + nEqns, i + nEqns);
						});
			 });

    member.team_barrier();

    bool is_valid(true);
    /// sanity check
    Tines::CheckNanInf::invoke(member, fv, is_valid);

    if (is_valid) {
      // solve linear system Ax= B A(Fv) B(Fu) X(dvdu)
      /// solve the equation: dx = -J^{-1} f(x);
      // dx = vu J = fv f=fu
      int matrix_rank(0);
      Tines::SolveLinearSystem::invoke(member,
				       fv, vu, fu, work_linear_solver, matrix_rank);
    } else{
      printf("Jacabian has Nan or Inf \n");
    }

    member.team_barrier();

    Kokkos::parallel_for(
			 Kokkos::TeamThreadRange(member, nEqns),
			 [&](const int &i) {
			   Kokkos::parallel_for(
						Kokkos::ThreadVectorRange(member, nEqns),
						[&](const int &j) {
						  double val(0);
						  for (int k = 0; k < nConstrains; k++) {
						    val +=  gv(i,k)*vu(k,j);
						  }
						  redJac(i,j) = gu(i,j) + val   ;//
						});
			 });

    member.team_barrier();

  }
};

namespace CSP {
namespace Test {
static inline bool
compareFiles(const std::string& filename1, const std::string& filename2)
{
std::ifstream f1(filename1);
std::string s1((std::istreambuf_iterator<char>(f1)),
               std::istreambuf_iterator<char>());
std::ifstream f2(filename2);
std::string s2((std::istreambuf_iterator<char>(f2)),
               std::istreambuf_iterator<char>());
return (s1.compare(s2) == 0);
}

static inline bool
compareFilesValues(const std::string& filename1, const std::string& filename2)
{
using real_type = double;

std::ifstream f1(filename1);
std::string s1((std::istreambuf_iterator<char>(f1)),
               std::istreambuf_iterator<char>());
std::ifstream f2(filename2);
std::string s2((std::istreambuf_iterator<char>(f2)),
               std::istreambuf_iterator<char>());

bool passTest(true);
if (s1.compare(s2) != 0) {

  using ats = Tines::ats<real_type>;

  std::ifstream file1(filename1);
  std::ifstream file2(filename2);
  real_type max_relative_error(0);
  real_type max_absolute_error(0);
  real_type save_value1(0);
  real_type save_value2(0);

  if (file1.is_open() && file2.is_open()) {
    //header
    std::string line;
    std::getline(file1, line);
    std::istringstream iss1(line);

    std::string line2;
    std::getline(file2, line2);
    std::istringstream iss2(line2);

    real_type value1, value2, diff;
    while (file1 >> value1 && file2 >> value2) {

      diff = ats::abs(value1 - value2) ;
      if (diff > max_absolute_error)
      {
        max_absolute_error = diff;
        max_relative_error = diff/value2;
        save_value1= value1;
        save_value2= value2;

      }
    }

  } else {
    printf("test : Could not open %s -> Abort !\n", filename1.c_str());
    printf("test : Could not open %s -> Abort !\n", filename2.c_str());
    return (false);
  }
  printf("Files are not exactly the same \n");
  printf("Maximum relative error : %15.10e \n", max_relative_error );
  printf("Maximum absolute error : %15.10e \n", max_absolute_error );
  printf("Current value :  %15.10e Reference value : %15.10e \n",save_value1 ,save_value2 );
  const real_type margin = 100, threshold = ats::epsilon() * margin;
  if ( ats::abs(max_relative_error) < threshold) {
    printf("PASS with threshold: %15.10e \n", threshold );
    passTest=true;
  } else {
    printf("FAIL with threshold: %15.10e \n", threshold );
    passTest=false;
  }

}

return (passTest);
}

} // end Test
} // end CSP


#endif  //end of header guard
