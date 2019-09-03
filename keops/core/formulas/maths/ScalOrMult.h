#pragma once

#include "core/Pack.h"
#include "core/formulas/maths/maths.h"
#include "core/formulas/maths/Scal.h"
#include "core/formulas/maths/Mult.h"

namespace keops {

// small hack to be able to use the * operator for both
// Scal and Mult depending on dimension in the new syntax

template<class FA, class FB>
using ScalOrMult = CondType<Mult <FA, FB>,
CondType<Scal < FB, FA>, CondType<Scal< FA, FB>, Mult<FA, FB>, FA::DIM == 1>, FB::DIM == 1>,
FA::DIM == FB::DIM>;

}