#pragma once

#include <sstream>
#include <assert.h>

#include "core/autodiff/UnaryOp.h"
#include "core/formulas/maths/Atan2.h"
#include "core/formulas/complex/ComplexReal.h"
#include "core/formulas/complex/ComplexImag.h"

#include "core/utils/keops_math.h"
#include "core/pre_headers.h"

namespace keops {


/////////////////////////////////////////////////////////////////////////
////      ComplexAngle                           ////
/////////////////////////////////////////////////////////////////////////

template<class F>
using ComplexAngle = Atan2 < ComplexImag<F>, ComplexReal<F> >;

#define ComplexAngle(f) KeopsNS<ComplexAngle<decltype(InvKeopsNS(f))>>()

}
