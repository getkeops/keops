#pragma once

#include "core/formulas/maths/ScalOrMult.h"
#include "core/formulas/maths/Inv.h"

namespace keops {



//////////////////////////////////////////////////////////////
////      DIVIDE : Divide<A,B> is A/B                     ////
//////////////////////////////////////////////////////////////

template<class FA, class FB>
using Divide = ScalOrMult<FA, Inv<FB>>;


}