#pragma once

#include <assert.h>

#include "core/Pack.h"
#include "core/autodiff.h"

namespace keops {

//////////////////////////////////////////////////////////////
////                 SUM : Sum< F >                       ////
//////////////////////////////////////////////////////////////

template<class F, int D>
struct SumT;

template<class F>
struct Sum : UnaryOp<Sum, F> {

  static const int DIM = 1;

  static void PrintIdString(std::stringstream &str) {
    str << "Sum";
  }

  static HOST_DEVICE INLINE
  void Operation(__TYPE__ *out, __TYPE__ *outF) {
    *out = 0;
#pragma unroll
    for (int k = 0; k < F::DIM; k++)
      *out += outF[k];
  }

  template<class V, class GRADIN>
  using DiffT = typename F::template DiffT<V, SumT<GRADIN, F::DIM>>;

};



//////////////////////////////////////////////////////////////
////        Transpose of Sum : SumT< F >                   ////
//////////////////////////////////////////////////////////////

template<class F, int D>
struct SumT : UnaryOp<SumT, F, D> {

  static_assert(F::DIM == 1, "Dimension of input must be 1 for SumT");

  static const int DIM = D;

  static void PrintIdString(std::stringstream& str) { str << "SumT"; }

  static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outF) {
#pragma unroll
    for (int k = 0; k < DIM; k++)
      out[k] = *outF;
  }

  template<class V, class GRADIN>
  using DiffT = typename F::template DiffT<V, Sum<GRADIN>>;

};

}
