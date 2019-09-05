#pragma once

#include "core/formulas/maths/Exp.h"
#include "core/formulas/maths/Scal.h"
#include "core/formulas/maths/Log.h"

#include "core/pre_headers.h"

namespace keops {

//////////////////////////////////////////////////////////////
////             POWER OPERATOR : Powf< A, B >            ////
//////////////////////////////////////////////////////////////

template<class FA, class FB>
using Powf = Exp<Scal<FB, Log<FA>>>;

#define Powf(fa,fb) KeopsNS<Powf<decltype(InvKeopsNS(fa)),decltype(InvKeopsNS(fb))>>()

}