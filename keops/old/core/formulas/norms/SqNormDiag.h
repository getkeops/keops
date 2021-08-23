#pragma once

#include "core/formulas/maths/Mult.h"
#include "core/formulas/maths/Square.h"
#include "core/formulas/maths/Sum.h"

namespace keops {

// Anisotropic (but diagonal) norm, if S::DIM == A::DIM:
// SqNormDiag<S,A> = sum_i s_i*a_i*a_i
template < class FS, class FA >
using SqNormDiag = Sum<Mult<FS,Square<FA>>>;

}
