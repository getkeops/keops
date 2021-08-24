#pragma once

#include "core/formulas/maths/Scal.h"
#include "core/formulas/norms/SqNorm2.h"

namespace keops {

//////////////////////////////////////////////////////////////
////          ISOTROPIC NORM :   SqNormIso< S,A >         ////
//////////////////////////////////////////////////////////////

// Isotropic norm, if S is a scalar:
// SqNormIso<S,A> = S * <A,A> = S * sum_i a_i*a_i
template<class FS, class FA>
using SqNormIso = Scal<FS,SqNorm2<FA>>;

}
