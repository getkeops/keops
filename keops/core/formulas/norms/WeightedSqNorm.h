#pragma once

#include "core/pack/CondType.h"
#include "core/formulas/maths/TensorProd.h"
#include "core/formulas/maths/Mult.h"
#include "core/formulas/maths/Sum.h"
#include "core/formulas/norms/SqNormDiag.h"
#include "core/formulas/norms/SqNormIso.h"
#include "core/pre_headers.h"


namespace keops {

///////////////////////////////////////////////////////////////////////////////////////
////             Fully anisotropic norm, if S::DIM == A::DIM * A::DIM:             ////
///////////////////////////////////////////////////////////////////////////////////////

// SymSqNorm<A,X> = sum_{ij} a_ij * x_i*x_j
template < class A, class X >
using SymSqNorm = Sum<Mult<A,TensorProd<X,X>>>;

// WeightedSqNorm<A,X> : redirects to SqNormIso, SqNormDiag or SymSqNorm
// depending on dimension of A
template < class A, class X >
using WeightedSqNorm = CondType< SqNormIso<A,X>,
CondType< SqNormDiag<A,X>, SymSqNorm<A,X>, A::DIM==X::DIM >,
A::DIM == 1  >;

#define WeightedSqNorm(s,f)   KeopsNS<WeightedSqNorm<decltype(InvKeopsNS(s)), decltype(InvKeopsNS(f))>>()



}
