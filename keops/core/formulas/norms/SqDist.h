#pragma once

#include "core/formulas/maths/Subtract.h"
#include "core/formulas/norms/SqNorm2.h"
#include "core/pre_headers.h"

namespace keops {

//////////////////////////////////////////////////////////////
////      SQUARED DISTANCE : SqDist<A,B>                  ////
//////////////////////////////////////////////////////////////

template < class X, class Y >
using SqDist = SqNorm2<Subtract<X,Y>>;

#define SqDist(f,g) KeopsNS<SqDist<decltype(InvKeopsNS(f)),decltype(InvKeopsNS(g))>>()

}
