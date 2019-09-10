#pragma once

#include "core/pack/CondType.h"
#include "core/formulas/maths/Scal.h"
#include "core/formulas/maths/Mult.h"

#include "core/pre_headers.h"

namespace keops {

// small hack to be able to use the * operator for both
// Scal and Mult depending on dimension in the new syntax

template<class FA, class FB>
using ScalOrMult = CondType<Mult <FA, FB>,
CondType<Scal < FB, FA>, CondType<Scal< FA, FB>, Mult<FA, FB>, FA::DIM == 1>, FB::DIM == 1>,
FA::DIM == FB::DIM>;

template < class FA, class FB >
KeopsNS<ScalOrMult<FA,FB>> operator*(KeopsNS<FA> fa, KeopsNS<FB> fb) {
  return KeopsNS<ScalOrMult<FA,FB>>();
}
#define ScalOrMult(fa,fb) KeopsNS<ScalOrMult<decltype(InvKeopsNS(fa)),decltype(InvKeopsNS(fb))>>()

}