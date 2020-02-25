#pragma once

#include <sstream>
#include <assert.h>

#include "core/autodiff/BinaryOp.h"
#include "core/formulas/maths/Add.h"
#include "core/formulas/maths/TensorProd.h"
#include "core/formulas/maths/VecMatMult.h"

#include "core/pre_headers.h"

namespace keops {


/////////////////////////////////////////////////////////////////////////
////      Matrix-vector product      A x b                           ////
/////////////////////////////////////////////////////////////////////////

template<class A, class B>
struct MatVecMult: BinaryOp<MatVecMult, A, B> {
  // A is vector of size n*p, interpreted as matrix, B is vector of size p, interpreted as column vector
  // output is vector of size n

  static_assert(A::DIM % B::DIM == 0, "Dimensions of A and B are not compatible for matrix-vector product");

  static const int DIM = A::DIM / B::DIM;

  static void PrintIdString(::std::stringstream &str) { str << "x"; }

#if C_CONTIGUOUS //row major

  template < typename TYPE > 
  static DEVICE INLINE void Operation(TYPE *out, TYPE *inA, TYPE *inB) {
        int q = 0;
        #pragma unroll
        for (int i = 0; i < DIM; i++) {
            out[i] = cast_to<TYPE>(0.0f);
            #pragma unroll
            for (int k = 0; k < B::DIM; k++, q++)
                out[i] += inA[q] * inB[k];
        }
    }

#else // column major

  template < typename TYPE > 
  static DEVICE INLINE void Operation(TYPE *out, TYPE *inA, TYPE *inB) {
    #pragma unroll
    for (int i = 0; i < DIM; i++) {
      out[i] = cast_to<TYPE>(0.0f);
      #pragma unroll
      for (int k = 0; k < B::DIM; k++)
        out[i] += inA[k * DIM + i] * inB[k];
    }
  }

#endif

  template<class V, class GRADIN>
  using DiffTA = typename A::template DiffT<V, GRADIN>;

  template<class V, class GRADIN>
  using DiffTB = typename B::template DiffT<V, GRADIN>;

  template<class V, class GRADIN>
  using DiffT = Add<DiffTA<V, TensorProd<GRADIN, B>>, DiffTB<V, VecMatMult<GRADIN, A>>>;

};

#define MatVecMult(f,g) KeopsNS<MatVecMult<decltype(InvKeopsNS(f)),decltype(InvKeopsNS(g))>>()

}
