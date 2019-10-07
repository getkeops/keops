#pragma once

#include <sstream>
#include <math.h>

#include "core/autodiff/BinaryOp.h"
#include "core/formulas/constants/IntConst.h"
#include "core/formulas/constants/Zero.h"

#include "core/pre_headers.h"

namespace keops {

//////////////////////////////////////////////////////////////
////         Element-wise Modulo : Mod< FA,FB>            ////
//////////////////////////////////////////////////////////////


template<class FA, class FB>
struct Mod : BinaryOp<Mod, FA, FB> {
  // FA and FB are vectors with same size, Output has the same size
  static const int DIM = FA::DIM;
  static_assert(FB::DIM == DIM, "Dimensions of FA and FB must be the same for Mod.");

  static void PrintIdString(::std::stringstream &str) {
    str << "%";
  }

  static DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outA, __TYPE__ *outB) {
#pragma unroll
    for (int k = 0; k < DIM; k++)
      out[k] = fmod(outA[k], outB[k]);
  }

  // N.B.: the gradient could be implemented if needed.
  template < class V, class GRADIN >
  using DiffT = Zero<V::DIM>;

};


template < class FA, class FB >
KeopsNS <Mod< FA, FB>> operator%(KeopsNS <FA> fa, KeopsNS <FB> fb) {
  return KeopsNS < Mod < FA, FB >> ();
}
#define Mod(fa, fb) KeopsNS<Mod<decltype(InvKeopsNS(fa)),decltype(InvKeopsNS(fb))>>()

}
