#pragma once

#include <sstream>

#include "core/pack/UnivPack.h"
#include "core/utils/TypesUtils.h"
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

  static void PrintId(::std::stringstream& str) { str << N; }

  template< class A, class B >
  using Replace = IntConstant< N >;

  template< class A1, class B1, class A2, class B2 >
  using ReplaceVars2 = IntConstant< N >;

  // N.B we comment out AutoFactorize and AllTypes in all code as of oct 2020 to speed up compile time
  // using AllTypes = univpack< IntConstant< N > >;

  template < int CAT >      // Whatever CAT...
  using VARS = univpack<>;  // there's no variable used in there.

  // Evaluation is easy : simply fill *out = out[0] with N.
  template < class INDS, typename TYPE, typename... ARGS >
  static DEVICE INLINE void Eval(TYPE* out, ARGS... args) {
    *out = cast_to<TYPE>((float)N);
  }

  // There is no gradient to accumulate on V, whatever V.
  template < class V, class GRADIN >
  using DiffT = Zero<V::DIM>;


  template < int DIMCHK >
  using CHUNKED_VERSION = IntConstant< N >;

  static const bool IS_CHUNKABLE = true;

  template < int DIMCHK >
  using CHUNKED_FORMULAS = univpack<>;

  static const int NUM_CHUNKED_FORMULAS = 0;

  template < int IND >
  using POST_CHUNK_FORMULA = IntConstant< N >;

  template < int CAT >
  using CHUNKED_VARS = univpack<>;

  template < int CAT >
  using NOTCHUNKED_VARS = univpack<>;

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
