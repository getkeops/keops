#pragma once

#include "core/pack/UnivPack.h"
#include "core/pack/GetInds.h"
#include "core/pack/IterReplace.h"
#include "core/autodiff/Grad.h"
#include "core/formulas/maths/ElemT.h"
#include "core/formulas/maths/Concat.h"
#include "core/formulas/constants/IntConst.h"
#include "core/pre_headers.h"

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


#define GradMatrix(f,g) KeopsNS<GradMatrix<decltype(InvKeopsNS(f)),decltype(InvKeopsNS(g))>>()

}
