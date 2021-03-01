#pragma once

#include <assert.h>

#include "core/pack/CondType.h"
#include "core/autodiff/BinaryOp.h"
#include "core/formulas/constants/Zero.h"
#include "core/formulas/constants/IntConst.h"
#include "core/formulas/maths/Sum.h"
#include "core/formulas/maths/Scal.h"
#include "core/utils/keops_math.h"

#include "core/formulas/maths/Square.h"
#include "core/formulas/maths/Divide.h"
#include "core/formulas/maths/Mult.h"
#include "core/formulas/maths/Subtract.h"


namespace keops {

//////////////////////////////////////////////////////////////
////              Atan2 :  Atan2< FA, Fb >                ////
//////////////////////////////////////////////////////////////

//using namespace keops;

template < class FA, class FB >
struct Atan2_Impl : BinaryOp< Atan2_Impl, FA, FB > {

  // Output dim = FA::DIM = FB::DIM
  static const int DIM = FA::DIM;
  static_assert(DIM == FB::DIM, "Dimensions must be the same for Atan2");

  static void PrintIdString(::std::stringstream &str) { str << "Atan2"; }

  template < typename TYPE >
  static DEVICE INLINE void Operation(TYPE* out, TYPE* outA, TYPE* outB) {
    #pragma unroll
    for (int k = 0; k < DIM; k++) {
      out[k] = keops_atan2(outA[k], outB[k]);
    }
  }

  // [ \partial_V Atan2(A, B) ] . gradin = [ -(A / A^2 + B^2) . \partial_V A ] . gradin  + [ (B / A^2 + B^2) . \partial_V B ] . gradin
  template < class V, class GRADIN>
  using partial_FA = typename FA::template DiffT< V, Mult< Divide<FA, Add<Square<FA>, Square<FB>> >, GRADIN> >;
  template < class V, class GRADIN>
  using partial_FB = typename FB::template DiffT< V, Mult< Divide<FB, Add<Square<FA>, Square<FB>> >, GRADIN> >;

  template < class V, class GRADIN >
  using DiffT = Subtract<partial_FB<V, GRADIN>, partial_FA<V, GRADIN>>;

};

#define Atan2(fa, fb) KeopsNS<Atan2_Impl<decltype(InvKeopsNS(fa)), decltype(InvKeopsNS(fb))>>()

}
