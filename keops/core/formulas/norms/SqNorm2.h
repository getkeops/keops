#pragma once

#include "core/formulas/norms/Scalprod.h"

namespace keops {

//////////////////////////////////////////////////////////////
////         SQUARED L2 NORM : SqNorm2< F >               ////
//////////////////////////////////////////////////////////////

// Simple alias
template < class F >
using SqNorm2 = Scalprod<F,F>;

}