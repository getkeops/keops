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
// depending on dimension of A. We use a template "Impl" structure
// and specialize for the three cases identified by int_id template
// parameter (int_id=0,1 or 2)
template < class A, class X, int int_id > struct WeightedSqNorm_Impl {};

template < class A, class X >
struct WeightedSqNorm_Impl<A,X,0> {
    using type = SqNormIso<A,X>;
};

template < class A, class X >
struct WeightedSqNorm_Impl<A,X,1> {
    using type = SqNormDiag<A,X>;
};

template < class A, class X >
struct WeightedSqNorm_Impl<A,X,2> {
    using type = SymSqNorm<A,X>;
};

// now we redirect to the appropriate spectialization
// A::DIM==1      =>  int_id=0  =>  SqNormIso<A,X>
// A::DIM==X::DIM =>  int_id=1  =>  SqNormDiag<A,X>
// otherwise      =>  int_id=2  =>  SymSqNorm<A,X>
template < class A, class X >
using WeightedSqNorm = typename WeightedSqNorm_Impl<A,X,(!(A::DIM==1))*(1+!(A::DIM==X::DIM))>::type;

#define WeightedSqNorm(s,f)   KeopsNS<WeightedSqNorm<decltype(InvKeopsNS(s)), decltype(InvKeopsNS(f))>>()



}
