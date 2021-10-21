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


#ifndef EIGENDECOMPOSITION_KOKKOS_CSP
#define EIGENDECOMPOSITION_KOKKOS_CSP

#include "Tines.hpp"


//#define EIGENDECOMPOSITION_KOKKOS_CSP_TIMER
typedef double value_type;
typedef int ordinal_type;
typedef Kokkos::DefaultExecutionSpace SpT;
typedef Kokkos::DefaultHostExecutionSpace HpT;

namespace CSP {

  struct ScopeGuard {
    ScopeGuard(int argc, char** argv) { Kokkos::initialize(argc, argv); }
    ~ScopeGuard() { Kokkos::finalize(); }
  };

  /// use host space for checking
  struct TestCheck {
    using value_type_2d_view = Tines::value_type_2d_view<value_type,Tines::UseThisDevice<HpT>::type>;
    using value_type_3d_view = Tines::value_type_3d_view<value_type,Tines::UseThisDevice<HpT>::type>;

    using complex_value_type_2d_view = Tines::value_type_2d_view<Kokkos::complex<value_type>,Tines::UseThisDevice<HpT>::type>;
    using complex_value_type_3d_view = Tines::value_type_3d_view<Kokkos::complex<value_type>,Tines::UseThisDevice<HpT>::type>;

    int _N, _Blk;

    value_type_3d_view _A_problem;
    value_type_2d_view _Er;
    value_type_2d_view _Ei;
    value_type_3d_view _V;

    complex_value_type_2d_view _Ec;
    complex_value_type_3d_view _Vc;
    complex_value_type_3d_view _Ac;
    complex_value_type_3d_view _Rc;

    struct ConvertToComplexTag {};
    struct CheckRightEigenvectorTag {};

    template<typename MemberType>
    KOKKOS_INLINE_FUNCTION
    void operator()(const ConvertToComplexTag &, const MemberType &member) const {
      const int i = member.league_rank();

      // for convenience, create a complex eigenvalues and eigenvectors
      auto er = Kokkos::subview(_Er, i, Kokkos::ALL());
      auto ei = Kokkos::subview(_Ei, i, Kokkos::ALL());
      auto V  = Kokkos::subview(_V,  i, Kokkos::ALL(), Kokkos::ALL());

      auto ec = Kokkos::subview(_Ec, i, Kokkos::ALL());
      auto Vc = Kokkos::subview(_Vc, i, Kokkos::ALL(), Kokkos::ALL());

      Tines::EigendecompositionToComplex
	::invoke(member,
		 er, ei, V,
		 ec, Vc);
    }

    template<typename MemberType>
    KOKKOS_INLINE_FUNCTION
    void operator()(const CheckRightEigenvectorTag &, const MemberType &member) const {
      const int i = member.league_rank();

      auto Ap = Kokkos::subview(_A_problem, i, Kokkos::ALL(), Kokkos::ALL());
      auto Ac = Kokkos::subview(_Ac,        i, Kokkos::ALL(), Kokkos::ALL());
      auto Rc = Kokkos::subview(_Rc,        i, Kokkos::ALL(), Kokkos::ALL());

      auto e  = Kokkos::subview(_Ec       , i, Kokkos::ALL());
      auto Vc = Kokkos::subview(_Vc       , i, Kokkos::ALL(), Kokkos::ALL());

      double rel_err(0);
      Tines::Copy::invoke(member, Ap, Ac);
      Tines::EigendecompositionValidateRightEigenPairs
	::invoke(member,
		 Ac, e, Vc, Rc,
		 rel_err);
    }

    template<typename MViewType>
    double computeNormSquared(const MViewType &M) {
      double norm = 0;
      const auto Blk(_Blk);
      Kokkos::parallel_reduce
	(Kokkos::RangePolicy<HpT>(0,_N),
	 KOKKOS_LAMBDA(const int &k, double &update) {
	  for (int i=0;i<Blk;++i)
	    for (int j=0;j<Blk;++j) {
	      const auto val = Tines::ats<typename MViewType::non_const_value_type>::abs(M(k,i,j));
	      update += val*val;
	    }
	}, norm);
      return norm;
    }

    bool checkTest(double tol = 1e-6) {
      // reconstruct matrix and compute diff
      Kokkos::parallel_for(Kokkos::TeamPolicy<HpT,ConvertToComplexTag>(_N, Kokkos::AUTO), *this);
      Kokkos::fence();

      const double q = _N;

      const double norm_ref = computeNormSquared(_A_problem);

      Kokkos::parallel_for(Kokkos::TeamPolicy<HpT,CheckRightEigenvectorTag>(_N, Kokkos::AUTO), *this);
      Kokkos::fence();

      const double norm_right = computeNormSquared(_Rc);
      const bool right_pass = std::sqrt(norm_right/norm_ref/q) < tol;

      printf(" --- A*VR - VR*E    : ref norm %e, diff %e\n", norm_ref, norm_right);

      return right_pass;
    }

    template<typename AViewType,
	     typename EViewType,
	     typename VViewType>
    TestCheck(const int N, const int Blk,
	      const AViewType &A_problem,
	      const EViewType &Er,
	      const EViewType &Ei,
	      const VViewType &V)
      : _N(N),
	_Blk(Blk),
	_A_problem(),
	_Er(),
	_Ei(),
	_V(),
	_Ec(),
	_Vc(),
	_Ac(),
	_Rc() {
      _A_problem = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A_problem);
      _Er = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Er);
      _Ei = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Ei);
      _V  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), V);

      _Ec = complex_value_type_2d_view("Ec", N, Blk);
      _Vc = complex_value_type_3d_view("Vc", N, Blk, Blk);
      _Ac = complex_value_type_3d_view("Ac", N, Blk, Blk);
      _Rc = complex_value_type_3d_view("Rc", N, Blk, Blk);
    }

  };

}

/// We already use SpT in the above
template<typename ExecSpaceType>
struct EigendecompositionKokkos {

  using value_type_2d_view = Tines::value_type_2d_view<value_type,typename Tines::UseThisDevice<ExecSpaceType>::type>;
  using value_type_3d_view = Tines::value_type_3d_view<value_type,typename Tines::UseThisDevice<ExecSpaceType>::type>;

public:

  inline
  static void eval(const value_type_3d_view &_A,
                   std::vector<std::vector <value_type> >& eig_val_real,
                   std::vector<std::vector <value_type> >& eig_val_imag,
                   std::vector < std::vector<std::vector <value_type> > >& eig_vec_R)
  {
    const int nBatch = _A.extent(0);
    const int Nvars =  _A.extent(1);

    /// we do not want to touch the given matrix
    value_type_3d_view A("A", nBatch, Nvars, Nvars);
    Kokkos::deep_copy(A,_A);

    value_type_2d_view er("real eigen values", nBatch, Nvars );
    value_type_2d_view ei("imag eigen values", nBatch, Nvars );

    value_type_3d_view V("V", nBatch, Nvars, Nvars);

    //work spaces
    const int wsize = 3*Nvars*Nvars + 2*Nvars;
    value_type_2d_view w ("w",  nBatch, wsize);
    Tines::SolveEigenvaluesNonSymmetricProblemDevice<ExecSpaceType>
      ::invoke(ExecSpaceType(), A, er, ei, V, w);

    eig_val_real =
      std::vector<std::vector<value_type>>(nBatch,
					   std::vector<value_type>(Nvars,0.0));

    eig_val_imag =
      std::vector<std::vector<value_type>>(nBatch,
					   std::vector<value_type>(Nvars,0.0));

    eig_vec_R = std::vector< std::vector< std::vector< value_type > > >
      (nBatch, std::vector<std::vector<value_type> > (Nvars,
						      std::vector<value_type>(Nvars,0)));

    Tines::convertToStdVector(eig_val_real, er);
    Tines::convertToStdVector(eig_val_imag, ei);
    Tines::convertToStdVector(eig_vec_R, V);
  }
};

struct EigenSolver{

  using value_type_3d_view      = Tines::value_type_3d_view<value_type,Tines::UseThisDevice<SpT>::type>;
  using value_type_3d_view_host = Tines::value_type_3d_view<value_type,Tines::UseThisDevice<HpT>::type>;

  inline
  static void evalDevice(const value_type_3d_view &A,
			 std::vector<std::vector <value_type> >& eig_val_real,
			 std::vector<std::vector <value_type> >& eig_val_imag,
			 std::vector < std::vector<std::vector <value_type> > >& eig_vec_R)
  {
    EigendecompositionKokkos<SpT>::eval(A,
					eig_val_real,
					eig_val_imag,
					eig_vec_R);

  }

  inline
  static void evalDevice(std::vector < std::vector<std::vector <value_type> > >& A,
       std::vector<std::vector <value_type> >& eig_val_real,
       std::vector<std::vector <value_type> >& eig_val_imag,
       std::vector < std::vector<std::vector <value_type> > >& eig_vec_R)
  {

    value_type_3d_view A_kokkos;
    Tines::convertToKokkos(A_kokkos, A);

    EigendecompositionKokkos<SpT>::eval(A_kokkos,
          eig_val_real,
          eig_val_imag,
          eig_vec_R);

  }

  inline
  static void evalHost(const value_type_3d_view_host &A,
		       std::vector<std::vector <value_type> >& eig_val_real,
		       std::vector<std::vector <value_type> >& eig_val_imag,
		       std::vector < std::vector<std::vector <value_type> > >& eig_vec_R)
  {
    EigendecompositionKokkos<HpT>::eval(A,
					eig_val_real,
					eig_val_imag,
					eig_vec_R);

  }

  inline
  static void evalHost(std::vector < std::vector<std::vector <value_type> > >& A,
           std::vector<std::vector <value_type> >& eig_val_real,
           std::vector<std::vector <value_type> >& eig_val_imag,
           std::vector < std::vector<std::vector <value_type> > >& eig_vec_R)
  {
    value_type_3d_view_host  A_kokkos;
    Tines::convertToKokkos(A_kokkos, A);

    EigendecompositionKokkos<HpT>::eval(A_kokkos,
          eig_val_real,
          eig_val_imag,
          eig_vec_R);

  }

};


#endif  //end of header guard
