#pragma once

#include "core/formulas/norms/WeightedSqNorm.h"

namespace keops {

//////////////////////////////////////////////////////////////
////   WEIGHTED SQUARED DISTANCE : WeightedSqDist<S,A,B>  ////
//////////////////////////////////////////////////////////////

template < class S, class X, class Y >
using WeightedSqDist = WeightedSqNorm< S, Subtract<X,Y>>;

}