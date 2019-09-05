#pragma once

#include "core/formulas/maths/Sqrt.h"
#include "core/formulas/norms/Scalprod.h"
#include "core/pre_headers.h"

namespace keops {

//////////////////////////////////////////////////////////////
////           L2 NORM :   ||F||                          ////
//////////////////////////////////////////////////////////////

// Simple alias
template < class F >
using Norm2 = Sqrt<Scalprod<F,F>>;

#define Norm2(f) KeopsNS<Norm2<decltype(InvKeopsNS(f))>>()

}
