#pragma once

#include <sstream>

#include "core/pack/UnivPack.h"
#include "core/formulas/constants/Zero.h"

namespace keops {


// A constant integer value, defined using the IntConstant<N> syntax.

template < int N > struct IntConstant_Impl;
template < int N > struct IntConstant_Alias;
template < int N >
using IntConstant = typename IntConstant_Alias<N>::type;

template < int N >
struct IntConstant_Impl {
  static const int DIM = 1;

  static void PrintId(::std::stringstream& str) {
    str << N;
  }

  template< class A, class B >
  using Replace = IntConstant< N >;

  using AllTypes = univpack< IntConstant< N > >;

  template < int CAT >      // Whatever CAT...
  using VARS = univpack<>;  // there's no variable used in there.

  // Evaluation is easy : simply fill *out = out[0] with N.
  template < class INDS, typename... ARGS >
  static DEVICE INLINE void Eval(__TYPE__* out, ARGS... args) {
    *out = N;
  }

  // There is no gradient to accumulate on V, whatever V.
  template < class V, class GRADIN >
  using DiffT = Zero<V::DIM>;
};

// Simplification rule

// base class, redirects to implementation
template < int N >
struct IntConstant_Alias {
  using type = IntConstant_Impl<N>;
};

// 0 = 0
template<>
struct IntConstant_Alias<0> {
  using type = Zero<1>;
};

#define IntCst(N) KeopsNS<IntConstant<N>>()

}