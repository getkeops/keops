#pragma once

#include "core/autodiff/Var.h"

namespace keops {

//////////////////////////////////////////////////////////////
////      GRADIENT OPERATOR  : Grad< F, V, Gradin >       ////
//////////////////////////////////////////////////////////////

// Defines [\partial_V F].gradin function
// Symbolic differentiation is a straightforward recursive operation,
// provided that the operators have implemented their DiffT "compiler methods":
template < class F, class V, class GRADIN >
using Grad = typename F::template DiffT<V,GRADIN>;

// same with additional saved forward variable. This is only used for taking gradients of reductions operations.
template < class F, class V, class GRADIN, class FO >
using Grad_WithSavedForward = typename F::template DiffT<V,GRADIN,FO>;

// Defines [\partial_V F].gradin with gradin defined as a new variable with correct
// category, dimension and index position.
// This will work only when taking gradients of reduction operations (otherwise F::CAT
// is not defined so it will not compile). The position is the only information which
// is not available in the C++ code, so it needs to be provided by the user.
// Note additional variable to input saved forward
template < class F, class V, int I >
using GradFromPos = Grad_WithSavedForward<F,V,Var<I,F::DIM,F::CAT>,Var<I+1,F::DIM,F::CAT>>;

}