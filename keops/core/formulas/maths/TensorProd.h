#pragma once

#include "core/autodiff/BinaryOp.h"
#include "core/formulas/maths/Add.h"
#include "core/formulas/maths/MatVecMult.h"
#include "core/formulas/maths/VecMatMult.h"

#include "core/pre_headers.h"

namespace keops {

/////////////////////////////////////////////////////////////////////////
////      Tensor product      a x b^T                                ////
/////////////////////////////////////////////////////////////////////////

template<class A, class B>
struct TensorProd : BinaryOp<TensorProd, A, B> {
  // A is vector of size n, B is vector of size p,
  // output is vector of size n*p understood as a matrix n x p

  static const int DIM = A::DIM * B::DIM;

  static void PrintIdString(::std::stringstream &str) {
    str << "(x)";
  }
#if C_CONTIGUOUS // row major
  static DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *inA, __TYPE__ *inB) {
        int q = 0;
#pragma unroll
        for (int k = 0; k < A::DIM; k++) {
#pragma unroll
            for (int l = 0; l < B::DIM; l++, q++)
                out[q] = inA[k] * inB[l];
        }
    }
#else // column major
  static DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *inA, __TYPE__ *inB) {
    int q = 0;
#pragma unroll
    for (int i = 0; i < A::DIM; i++) {
#pragma unroll
      for (int j = 0; j < B::DIM; j++, q++)
        out[A::DIM * j + i] = inA[i] * inB[j];
    }
  }
#endif

  template<class V, class GRADIN>
  using DiffTA = typename A::template DiffT<V, GRADIN>;

  template<class V, class GRADIN>
  using DiffTB = typename B::template DiffT<V, GRADIN>;

  template<class V, class GRADIN>
  using DiffT = Add<DiffTA<V, MatVecMult<GRADIN, B>>, DiffTB<V, VecMatMult<A, GRADIN>>>;

};

#define TensorProd(f,g) KeopsNS<TensorProd<decltype(InvKeopsNS(f)),decltype(InvKeopsNS(g))>>()

}
