#pragma once

#include <sstream>

#include "core/pack/GetDims.h"
#include "core/pack/GetInds.h"

namespace keops {

// Print formula to string

template < class F >
::std::string PrintFormula() {
  ::std::stringstream str;
  str << "Variables : ";
  using Vars0 = typename F::template VARS< 0 >;
  using Dims0 = GetDims< Vars0 >;
  using Inds0 = GetInds< Vars0 >;
  for (int k = 0; k < Vars0::SIZE; k++)
    str << "x" << Inds0::VAL(k) << " (dim=" << Dims0::VAL(k) << "), ";
  using Vars1 = typename F::template VARS< 1 >;
  using Dims1 = GetDims< Vars1 >;
  using Inds1 = GetInds< Vars1 >;
  for (int k = 0; k < Vars1::SIZE; k++)
    str << "y" << Inds1::VAL(k) << " (dim=" << Dims1::VAL(k) << "), ";
  using Vars2 = typename F::template VARS< 2 >;
  using Dims2 = GetDims< Vars2 >;
  using Inds2 = GetInds< Vars2 >;
  for (int k = 0; k < Vars2::SIZE; k++)
    str << "p" << Inds2::VAL(k) << " (dim=" << Dims2::VAL(k) << "), ";
  str << ::std::endl;
  str << "Formula = ";
  F::PrintId(str);
  return str.str();
}

// other version, allowing to write PrintFormula(f) where f is instance of F
template < class F >
::std::string PrintFormula(F f) {
  return PrintFormula< F >();
}

// Print reduction to string
template < class F >
::std::string PrintReduction() {
  ::std::stringstream str;
  F::PrintId(str);
  return str.str();
}

}
