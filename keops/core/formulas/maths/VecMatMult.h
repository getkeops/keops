#pragma once

#include <sstream>
#include <assert.h>

#include "core/utils/keops_math.h"
#include "core/utils/TypesUtils.h"
#include "core/autodiff/BinaryOp.h"
#include "core/formulas/maths/Add.h"
#include "core/formulas/maths/MatVecMult.h"
#include "core/formulas/maths/TensorProd.h"
#include "core/pre_headers.h"

namespace keops {

/////////////////////////////////////////////////////////////////////////
////     Vector-matrix product           b x A                       ////
/////////////////////////////////////////////////////////////////////////

template<class A, class B>
struct MatVecMult;
template<class A, class B>
struct TensorProd;

template<class B, class A>
struct VecMatMult : BinaryOp<VecMatMult, B, A> {

  // A is vector of size n*p, interpreted as matrix, B is vector of size n, interpreted as row vector
  // output is vector of size p

  static_assert(A::DIM % B::DIM == 0, "Dimensions of A and B are not compatible for vector-matrix product");

  static const int DIM = A::DIM / B::DIM;

  static void PrintIdString(::std::stringstream &str) { str << "x"; }

#if C_CONTIGUOUS //row major

  template < typename TYPE >
  static DEVICE INLINE void Operation(TYPE *out, TYPE *inB, TYPE *inA) {
        #pragma unroll
        for (int i = 0; i < DIM; i++) {
            out[i] = cast_to<TYPE>(0.0f);
	    #pragma unroll
            for (int k = 0; k < B::DIM; k++)
                out[i] = keops_fma(inB[k], inA[DIM * k + i], out[i]);
        }
    }

#else // column major

  template < typename TYPE >
  static DEVICE INLINE void Operation(TYPE *out, TYPE *inB, TYPE *inA) {
    int q = 0;
    #pragma unroll
    for (int i = 0; i < DIM; i++) {
      out[i] = cast_to<TYPE>(0.0f);
      #pragma unroll
      for (int k = 0; k < B::DIM; k++, q++)
        out[i] = keops_fma(inB[k], inA[q], out[i]);
    }
  }

#endif

  template<class V, class GRADIN>
  using DiffTA = typename A::template DiffT<V, GRADIN>;

  template<class V, class GRADIN>
  using DiffTB = typename B::template DiffT<V, GRADIN>;

  template<class V, class GRADIN>
  using DiffT = Add<DiffTA<V, TensorProd<B, GRADIN>>, DiffTB<V, MatVecMult<A, GRADIN>>>;

};

#define VecMatMult(f,g) KeopsNS<VecMatMult<decltype(InvKeopsNS(f)),decltype(InvKeopsNS(g))>>()

}
