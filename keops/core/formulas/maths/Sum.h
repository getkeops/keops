#pragma once

#include <assert.h>

#include "core/autodiff/UnaryOp.h"
#include "core/formulas/maths/SumT.h"

#include "core/pre_headers.h"
namespace keops {

//////////////////////////////////////////////////////////////
////                 SUM : Sum< F >                       ////
//////////////////////////////////////////////////////////////

template<class F, int D>
struct SumT;

template<class F>
struct Sum : UnaryOp<Sum, F> {

  static const int DIM = 1;

  static void PrintIdString(::std::stringstream &str) {
    str << "Sum";
  }

  static DEVICE INLINE
  void Operation(__TYPE__ *out, __TYPE__ *outF) {
    *out = 0;
#if USE_HALF
#pragma unroll
    for (int k = 0; k < F::DIM; k++)
#if GPU_ON
      *out = __hadd(*out,outF[k]);
#else
      *out = *out + outF[k];
#endif
#else
#pragma unroll
    for (int k = 0; k < F::DIM; k++)
      *out += outF[k];
#endif
  }

  template<class V, class GRADIN>
  using DiffT = typename F::template DiffT<V, SumT<GRADIN, F::DIM>>;

};

#define Sum(p) KeopsNS<Sum<decltype(InvKeopsNS(p))>>()

}
