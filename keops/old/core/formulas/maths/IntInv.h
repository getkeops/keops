#pragma once

#include "core/formulas/constants/IntConst.h"
#include "core/formulas/maths/Inv.h"

#include "core/pre_headers.h"

namespace keops {

//////////////////////////////////////////////////////////////
////      INVERSE OF INTEGER CONSTANT : Inv<N> is 1/N     ////
//////////////////////////////////////////////////////////////

// remark : there is currently no way to get a fixed real number directly...

template<int N>
using IntInv = Inv<IntConstant<N>>;

#define IntInv(N) KeopsNS<IntInv<N>>()
}
