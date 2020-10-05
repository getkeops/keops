#pragma once

#include <sstream>
#include "core/utils/TypesUtils.h"
#include "core/pack/UnivPack.h"

namespace keops {


// A "zero" vector of size _DIM
// Declared using the   Zero<DIM>   syntax.
template < int _DIM >
struct Zero {
  static const int DIM = _DIM;

  static void PrintId(::std::stringstream& str) {
    str << "0(DIM=" << DIM << ")";
  }

  template<class A, class B>
  using Replace = Zero<DIM>;

  template<class A1, class B1, class A2, class B2>
  using ReplaceVars2 = Zero<DIM>;

  // N.B we comment out AutoFactorize and AllTypes in all code as of oct 2020 to speed up compile time
  // using AllTypes = univpack<Zero<DIM>>;

  template < int CAT >      // Whatever CAT...
  using VARS = univpack<>;  // there's no variable used in there.

  // Evaluation is easy : simply fill-up *out with zeros.
  template < class INDS, typename TYPE, typename... ARGS >
  static DEVICE INLINE void Eval(TYPE* out, ARGS... args) {
    VectAssign<DIM>(out, 0.0f);
  }

  // There is no gradient to accumulate on V, whatever V.
  template < class V, class GRADIN >
  using DiffT = Zero<V::DIM>;


  template < int DIMCHK >
  using CHUNKED_VERSION = Zero<DIMCHK>;

  static const bool IS_CHUNKABLE = true;

  template < int DIMCHK >
  using CHUNKED_FORMULAS = univpack<>;

  static const int NUM_CHUNKED_FORMULAS = 0;

  template < int IND >
  using POST_CHUNK_FORMULA = Zero<DIM>;

  template < int CAT >
  using CHUNKED_VARS = univpack<>;

  template < int CAT >
  using NOTCHUNKED_VARS = univpack<>;

};

#define Zero(D) KeopsNS<Zero<D>>()

}
