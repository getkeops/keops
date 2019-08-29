#pragma once

#include "core/Pack.h"
#include "core/autodiff.h"
#include "core/formulas/maths/StandardBasis.h"
#include "core/formulas/maths/Concat.h"

namespace keops {

/////////////////////////////////////////////////////////////////////////
////      Matrix of gradient operator (=transpose of jacobian)       ////
/////////////////////////////////////////////////////////////////////////


template<class F, class V>
struct GradMatrix_Impl {
  using IndsTempVars = GetInds<typename F::template VARS<3>>;
  using GRADIN = Var<1 + IndsTempVars::MAX, F::DIM, 3>;
  using packGrads = IterReplace <Grad<F, V, GRADIN>, GRADIN, StandardBasis<F::DIM>>;
  using type = IterBinaryOp<Concat, packGrads>;
};

template<class F, class V>
using GradMatrix = typename GradMatrix_Impl<F, V>::type;

}
