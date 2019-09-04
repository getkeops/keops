#pragma once

#include "core/formulas/maths/maths.h"
#include "core/formulas/maths/Exp.h"
#include "core/formulas/maths/Scal.h"
#include "core/formulas/maths/Log.h"

namespace keops {

//////////////////////////////////////////////////////////////
////             POWER OPERATOR : Powf< A, B >            ////
//////////////////////////////////////////////////////////////

template<class FA, class FB>
using Powf = Exp<Scal<FB, Log<FA>>>;

}