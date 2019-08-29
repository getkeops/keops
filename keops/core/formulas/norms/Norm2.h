#pragma once

#include "core/formulas/maths/Sqrt.h"
#include "core/formulas/norms/Scalprod.h"

namespace keops {

//////////////////////////////////////////////////////////////
////           L2 NORM :   ||F||                          ////
//////////////////////////////////////////////////////////////

// Simple alias
template < class F >
using Norm2 = Sqrt<Scalprod<F,F>>;

}
