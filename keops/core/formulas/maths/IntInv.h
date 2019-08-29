#pragma once

#include "core/formulas/constants.h"
#include "core/formulas/maths/Inv.h"

namespace keops {

//////////////////////////////////////////////////////////////
////      INVERSE OF INTEGER CONSTANT : Inv<N> is 1/N     ////
//////////////////////////////////////////////////////////////

// remark : there is currently no way to get a fixed real number directly...

template<int N>
using IntInv = Inv<IntConstant<N>>;

}