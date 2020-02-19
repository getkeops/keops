#pragma once

#include "core/pack/CondType.h"
#include "core/formulas/maths/Scal.h"
#include "core/formulas/maths/Mult.h"

#include "core/pre_headers.h"

namespace keops {

// small hack to be able to use the * operator for both
// Scal and Mult depending on dimension in the new syntax

template < class FA, class FB, class = void > struct ScalOrMult_Impl;

template < class FA, class FB >
struct ScalOrMult_Impl < FA, FB, typename std::enable_if < FA::DIM==FB::DIM >::type > {
    using type = typename Mult_Alias<FA,FB>::type;
};

template < class FA, class FB >
struct ScalOrMult_Impl < FA, FB, typename std::enable_if < (FA::DIM==1) && (FB::DIM>1) >::type > {
    using type = typename Scal_Alias<FA,FB>::type;
};

template < class FA, class FB >
struct ScalOrMult_Impl < FA, FB, typename std::enable_if < (FB::DIM==1) && (FA::DIM>1) >::type > {
    using type = typename Scal_Alias<FB,FA>::type;
};

template < class FA, class FB >
using ScalOrMult = typename ScalOrMult_Impl<FB,FA>::type;

template < class FA, class FB >
KeopsNS<ScalOrMult<FA,FB>> operator*(KeopsNS<FA> fa, KeopsNS<FB> fb) {
  return KeopsNS<ScalOrMult<FA,FB>>();
}
#define ScalOrMult(fa,fb) KeopsNS<ScalOrMult<decltype(InvKeopsNS(fa)),decltype(InvKeopsNS(fb))>>()

}
