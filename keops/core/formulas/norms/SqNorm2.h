#pragma once

#include "core/formulas/norms/Scalprod.h"
#include "core/pre_headers.h"

namespace keops {

//////////////////////////////////////////////////////////////
////         SQUARED L2 NORM : SqNorm2< F >               ////
//////////////////////////////////////////////////////////////

// Simple alias
template < class F >
using SqNorm2 = Scalprod<F,F>;

#define SqNorm2(f) KeopsNS<SqNorm2<decltype(InvKeopsNS(f))>>()

}