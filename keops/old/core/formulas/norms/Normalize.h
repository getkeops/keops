#pragma once

#include "core/formulas/maths/Scal.h"
#include "core/formulas/maths/Rsqrt.h"
#include "core/formulas/norms/SqNorm2.h"
#include "core/pre_headers.h"

namespace keops {
//////////////////////////////////////////////////////////////
////       NORMALIZE :   F / ||F||                        ////
//////////////////////////////////////////////////////////////

// Simple alias
template < class F >
using Normalize = Scal<Rsqrt<SqNorm2<F>>,F>;

#define Normalize(f) KeopsNS<Normalize<decltype(InvKeopsNS(f))>>()

}