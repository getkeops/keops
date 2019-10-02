#pragma once

#include <sstream>

#include "core/autodiff/UnaryOp.h"
#include "core/formulas/constants/IntConst.h"
#include "core/formulas/maths/Mult.h"
#include "core/formulas/maths/Scal.h"
#include "core/formulas/maths/Square.h"

#include "core/pre_headers.h"

namespace keops {

//////////////////////////////////////////////////////////////
////      INVERSE : Inv<F>                                ////
//////////////////////////////////////////////////////////////

//template < class F >
//using Inv = Pow<F,-1>;

template<class F>
struct Inv : UnaryOp<Inv, F> {

  static const int DIM = F::DIM;

  static void PrintIdString(::std::stringstream &str) {
    str << "Inv";
  }

  static DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outF) {
#pragma unroll
    for (int k = 0; k < DIM; k++) {
#ifdef __CUDA_ARCH__
#if USE_DOUBLE
      out[k] = 1 / outF[k];           // there is no fast divide for cuda and double
#else
      out[k] = fdividef(1.0, outF[k]);
#endif
#else
      out[k] = 1 / outF[k];
#endif
    }
  }

  template<class V, class GRADIN>
  using DiffTF = typename F::template DiffT<V, GRADIN>;

  // [\partial_V (F)**2].gradin = F * [\partial_V F].gradin
  template<class V, class GRADIN>
  using DiffT = Scal<IntConstant<-1>, DiffTF<V, Mult<Square<Inv<F>>, GRADIN>>>;

};

#define Inv(f) KeopsNS<Inv<decltype(InvKeopsNS(f))>>()

}
