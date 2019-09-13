#pragma once

#include "core/formulas/maths/Subtract.h"
#include "core/formulas/norms/WeightedSqNorm.h"
#include "core/pre_headers.h"

namespace keops {

//////////////////////////////////////////////////////////////
////   WEIGHTED SQUARED DISTANCE : WeightedSqDist<S,A,B>  ////
//////////////////////////////////////////////////////////////

template < class S, class X, class Y >
using WeightedSqDist = WeightedSqNorm< S, Subtract<X,Y>>;

#define WeightedSqDist(s,f,g) KeopsNS<WeightedSqDist<decltype(InvKeopsNS(s)), decltype(InvKeopsNS(f)),decltype(InvKeopsNS(g))>>()

}