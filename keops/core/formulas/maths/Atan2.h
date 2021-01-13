#pragma once

#include "core/utils/keops_math.h"
#include "core/formulas/maths/Atan.h"
#include "core/formulas/maths/Divide.h"
#include "core/autodiff/VectorizedScalarBinaryOp.h"


namespace keops {

//////////////////////////////////////////////////////////////
////                ATAN2 :  Atan2< F, G >                ////
//////////////////////////////////////////////////////////////



template<class F, class G>
struct Atan2 : VectorizedScalarBinaryOp<Atan2, F, G> {

  static void PrintIdString(::std::stringstream &str) { str << "Atan2"; }

  template < typename TYPE >
  struct Operation_Scalar {
    DEVICE INLINE void operator() (TYPE &out, TYPE &outF, TYPE &outG) {
          out = keops_atan2(outF, outG);
    }
  };
    
  using AtanEq = Atan < Divide < G, F > >;

  template<class V, class GRADIN>
  using DiffT = typename AtanEq::template DiffT<V,GRADIN>;

};

#define Atan2(f) KeopsNS<Atan2<decltype(InvKeopsNS(f))>>()

}
