#pragma once

#include "core/utils/keops_math.h"
#include "core/autodiff/VectorizedScalarUnaryOp.h"
#include "core/formulas/maths/Sin.h"
#include "core/formulas/maths/Minus.h"
#include "core/formulas/maths/Mult.h"
#include "core/formulas/maths/Inv.h"
#include "core/formulas/maths/Sqrt.h"
#include "core/formulas/maths/Subtract.h"
#include "core/formulas/constants/IntConst.h"
#include "core/formulas/maths/Square.h"


namespace keops {

//////////////////////////////////////////////////////////////
////                ARCCOSINE :  Acos< F >                ////
//////////////////////////////////////////////////////////////



template<class F>
struct Acos : VectorizedScalarUnaryOp<Acos, F> {

  static void PrintIdString(::std::stringstream &str) { str << "Acos"; }

  template < typename TYPE >
  struct Operation_Scalar {
    DEVICE INLINE void operator() (TYPE &out, TYPE &outF) {
          out = keops_acos(outF);
    }
  };

  template<class V, class GRADIN>
  using DiffT = typename F::template DiffT<V, Mult<Minus<Inv<Sqrt<Subtract<IntConstant<1>, Square<F>>>>>, GRADIN>>;

};

#define Acos(f) KeopsNS<Acos<decltype(InvKeopsNS(f))>>()

}
