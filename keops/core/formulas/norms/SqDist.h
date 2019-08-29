#pragma once

#include "core/formulas/norms/SqNorm2.h"

namespace keops {

//////////////////////////////////////////////////////////////
////      SQUARED DISTANCE : SqDist<A,B>                  ////
//////////////////////////////////////////////////////////////

template < class X, class Y >
using SqDist = SqNorm2<Subtract<X,Y>>;


}
