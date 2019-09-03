#pragma once

#include "core/Pack.h"
#include "core/autodiff.h"
#include "core/formulas/maths/maths.h"

namespace keops {

//////////////////////////////////////////////////////////////////////////////////////////////
////      Standard basis of R^DIM : < (1,0,0,...) , (0,1,0,...) , ... , (0,...,0,1) >     ////
//////////////////////////////////////////////////////////////////////////////////////////////

template<int DIM, int I = 0>
struct StandardBasis_Impl {
  using EI = ElemT<IntConstant<1>, DIM, I>;
  using type = typename StandardBasis_Impl<DIM, I + 1>::type::template PUTLEFT<EI>;
};

template<int DIM>
struct StandardBasis_Impl<DIM, DIM> {
  using type = univpack<>;
};

template<int DIM>
using StandardBasis = typename StandardBasis_Impl<DIM>::type;

}