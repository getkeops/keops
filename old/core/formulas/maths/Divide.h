#pragma once

#include "core/formulas/maths/ScalOrMult.h"
#include "core/formulas/maths/Inv.h"

#include "core/pre_headers.h"

namespace keops {



//////////////////////////////////////////////////////////////
////      DIVIDE : Divide<A,B> is A/B                     ////
//////////////////////////////////////////////////////////////

template<class FA, class FB>
using Divide = ScalOrMult<FA, Inv<FB>>;

template < class FA, class FB >
KeopsNS<Divide<FA,FB>> operator/(KeopsNS<FA> fa, KeopsNS<FB> fb) {
  return KeopsNS<Divide<FA,FB>>();
}
#define Divide(fa,fb) KeopsNS<Divide<decltype(InvKeopsNS(fa)),decltype(InvKeopsNS(fb))>>()
}