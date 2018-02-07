#pragma once

#include <iostream>
#include <assert.h>

#include "core/Pack.h"
#include "core/autodiff.h"

#include "core/formulas/constants.h"
#include "core/formulas/maths.h"

/*
 * The file where the elementary norm-related operators are defined.
 * Available norms and scalar products are :
 *
 *   < .,. >, | . |^2, | .-. |^2 :
 *      Scalprod<FA,FB> 			: scalar product between FA and FB
 *      SqNorm2<F>					: alias for Scalprod<F,F>
 *      SqDist<A,B>					: alias for SqNorm2<Subtract<A,B>>
 *
 */




//////////////////////////////////////////////////////////////
////           SCALAR PRODUCT :   Scalprod< A,B >         ////
//////////////////////////////////////////////////////////////



template < class FA, class FB >
struct ScalprodImpl : BinaryOp<ScalprodImpl,FA,FB> {
    // Output dimension = 1, provided that FA::DIM = FB::DIM
    static const int DIMIN = FA::DIM;
    static_assert(DIMIN==FB::DIM,"Dimensions must be the same for Scalprod");
    static const int DIM = 1;

    static void PrintIdString() { cout << ","; }
    
    static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outA, __TYPE__ *outB) {
    		*out = 0;
            for(int k=0; k<DIMIN; k++)
            	*out += outA[k]*outB[k];
	}

    // <A,B> is scalar-valued, so that gradin is necessarily a scalar.
    // [\partial_V <A,B>].gradin = gradin * ( [\partial_V A].B + [\partial_V B].A )
    template < class V, class GRADIN >
    using DiffT = Scal < GRADIN , Add < typename FA::template DiffT<V,FB> , typename FB::template DiffT<V,FA> > >;
};


template < class FA, class FB >
struct ScalprodAlias {
    using type = ScalprodImpl<FA,FB>;
};

// Three simple optimizations :

// <A,0> = 0
template < class FA, int DIM >
struct ScalprodAlias<FA,Zero<DIM>> {
    static_assert(DIM==FA::DIM,"Dimensions must be the same for Scalprod");
    using type = Zero<1>;
};

// <0,B> = 0
template < class FB, int DIM >
struct ScalprodAlias<Zero<DIM>,FB> {
    static_assert(DIM==FB::DIM,"Dimensions must be the same for Scalprod");
    using type = Zero<1>;
};

// <0,0> = 0
template < int DIM1, int DIM2 >
struct ScalprodAlias<Zero<DIM1>,Zero<DIM2>> {
    static_assert(DIM1==DIM2,"Dimensions must be the same for Scalprod");
    using type = Zero<1>;
};




//////////////////////////////////////////////////////////////
////         SQUARED L2 NORM : SqNorm2< F >               ////
//////////////////////////////////////////////////////////////

// Simple alias
template < class F >
using SqNorm2 = Scalprod<F,F>;

//////////////////////////////////////////////////////////////
////      SQUARED DISTANCE : SqDist<A,B>                  ////
//////////////////////////////////////////////////////////////

template < class X, class Y >
using SqDist = SqNorm2<Subtract<X,Y>>;




