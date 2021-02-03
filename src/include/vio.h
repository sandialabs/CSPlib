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


//===============================================================
// overload the << operator to output a std::vector of any type
// outputs full std::vector contents starting from beginning to end
// thus a std::vector like (x1,x2,...,xn) is output as :
// x1 x2 ... xn
// elements are separated by one space, and no newline at end

template <typename TElem>
  std::ostream& operator<<(std::ostream& os, const std::vector<TElem>& v) {
  for (auto x : v)
    os << x << " ";
  return os;
}
//===============================================================
//===============================================================
// overload the << operator to output a std::vector of std::vectors of any type
// outputs full std::vector-std::vector contents starting from beginning to end
// thus a std::vector-std::vector like (x11,x12,...,x1n,x21,x22,...,x2n,...,xm1,xm2,...,xmn) is output as :
// x11 x12 ... x1n
// x21 x22 ... x2n
//      ...
// xm1 xm2 ... xmn
// elements are separated by one space, and with newline at end of each line

template <typename TElem>
  std::ostream& operator<<(std::ostream& os, const std::vector<std::vector<TElem> >& vv) {
  for (auto v : vv){
    for (auto x : v)
      os << x << " ";
    os << std::endl;
  }
  return os;
}
//===============================================================
