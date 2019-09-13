#pragma once

#include <sstream>
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

  using AllTypes = univpack<Zero<DIM>>;

  template < int CAT >      // Whatever CAT...
  using VARS = univpack<>;  // there's no variable used in there.

  // Evaluation is easy : simply fill-up *out with zeros.
  template < class INDS, typename... ARGS >
  static DEVICE INLINE void Eval(__TYPE__* out, ARGS... args) {
    for(int k=0; k<DIM; k++)
      out[k] = 0;
  }

  // There is no gradient to accumulate on V, whatever V.
  template < class V, class GRADIN >
  using DiffT = Zero<V::DIM>;
};

#define Zero(D) KeopsNS<Zero<D>>()

}