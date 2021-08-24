#pragma once

#include "core/formulas/complex/Real2Complex.h"
#include "core/formulas/complex/ComplexMult.h"
#include "core/formulas/complex/ComplexSquareAbs.h"
#include "core/formulas/complex/Conj.h"
#include "core/formulas/maths/Inv.h"

namespace keops {


/////////////////////////////////////////////////////////////////////////
////      ComplexDivide                           ////
/////////////////////////////////////////////////////////////////////////

template<class F, class G>
using ComplexDivide = ComplexMult < Real2Complex< Inv< ComplexSquareAbs<G> > >, ComplexMult< F, Conj<G> > >;

#define ComplexDivide(f,g) KeopsNS<ComplexDivide<decltype(InvKeopsNS(f)),decltype(InvKeopsNS(g))>>()

}
