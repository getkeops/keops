#pragma once

#include "core/formulas/complex/ComplexSquareAbs.h"
#include "core/formulas/maths/Sqrt.h"

namespace keops {


/////////////////////////////////////////////////////////////////////////
////      ComplexAbs                           ////
/////////////////////////////////////////////////////////////////////////

template<class F>
using ComplexAbs = Sqrt < ComplexSquareAbs < F > >;

#define ComplexAbs(f) KeopsNS<ComplexAbs<decltype(InvKeopsNS(f))>>()

}
