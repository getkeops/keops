#pragma once

#include "core/autodiff/Var.h"
#include "core/pack/Val.h"
#include "core/pack/IndVal.h"
#include "core/pack/GetInds.h"
#include "core/pack/ConcatPack.h"
#include "core/pre_headers.h"


namespace keops {

//////////////////////////////////////////////////////////////
////      GRADIENT OPERATOR  : Grad< F, V, Gradin >       ////
//////////////////////////////////////////////////////////////

// Defines [\partial_V F].gradin function
// Symbolic differentiation is a straightforward recursive operation,
// provided that the operators have implemented their DiffT "compiler methods":
template< class F, class V, class GRADIN >
using Grad = typename F::template DiffT< V, GRADIN >;

// same with additional saved forward variable. This is only used for taking gradients of reductions operations.
template< class F, class V, class GRADIN, class FO >
using Grad_WithSavedForward = typename F::template DiffT< V, GRADIN, FO >;

// Defines [\partial_V F].gradin with gradin defined as a new variable with correct
// category, dimension and index position.
// This will work only when taking gradients of reduction operations (otherwise F::CAT
// is not defined so it will not compile). The position is the only information which
// is not available in the C++ code, so it needs to be provided by the user.
// Note additional variable to input saved forward
template< class F, class V, int I >
using GradFromPos = Grad< F, V, Var< I, F::DIM, F::CAT > >;

template< class F, class V, int I >
using GradFromPos_WithSavedForward = Grad_WithSavedForward< F, V, Var< I, F::DIM, F::CAT >, Var< I + 1, F::DIM, F::CAT > >;

template< class F, int IND_V >
using VARSS = ConcatPacks<ConcatPacks<typename F::F::template VARS<0>, typename F::F::template VARS<1>>, typename F::F::template VARS<2> >;

template< class F, int IND_V, int I >
using GradFromInd = Grad< F, Val< VARSS< F, IND_V >, IndVal< GetInds< VARSS< F, IND_V > >, IND_V >::value >, Var< I, F::DIM, F::CAT > >;

template< class F, int IND_V, int I >
using GradFromInd_WithSavedForward = Grad_WithSavedForward< F, Val< VARSS< F, IND_V >, IndVal< GetInds< VARSS< F, IND_V > >, IND_V >::value >, Var< I, F::DIM, F::CAT >, Var< I + 1, F::DIM, F::CAT > >;

#define Grad(F, V, GRADIN)  KeopsNS< Grad< decltype(InvKeopsNS(F)), decltype(InvKeopsNS(V)), decltype(InvKeopsNS(GRADIN)) > >()
#define Grad_WithSavedForward(F, V, GRADIN, FO)  KeopsNS< Grad_WithSavedForward< decltype(InvKeopsNS(F)), decltype(InvKeopsNS(V)), decltype(InvKeopsNS(GRADIN)), decltype(InvKeopsNS(FO)) > >()

#define GradFromPos(F, V, I)  KeopsNS< GradFromPos< decltype(InvKeopsNS(F)), decltype(InvKeopsNS(V)), I > >()
#define GradFromPos_WithSavedForward(F, V, I)  KeopsNS< GradFromPos_WithSavedForward< decltype(InvKeopsNS(F)), decltype(InvKeopsNS(V)), I > >()

#define GradFromInd(F, Ind_V, I)  KeopsNS< GradFromInd< decltype(InvKeopsNS(F)), Ind_V, I > >()
#define GradFromInd_WithSavedForward(F, Ind_V, I)  KeopsNS< GradFromInd_WithSavedForward< decltype(InvKeopsNS(F)), Ind_V, I > >()
}
