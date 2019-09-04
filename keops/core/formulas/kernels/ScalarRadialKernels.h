#pragma once

#include "core/formulas/constants.h"
#include "core/formulas/maths/Exp.h"
#include "core/formulas/maths/Scal.h"
#include "core/formulas/maths/Minus.h"
#include "core/formulas/maths/Add.h"
#include "core/formulas/maths/Inv.h"
#include "core/formulas/maths/Sqrt.h"
#include "core/formulas/norms/Scalprod.h"
#include "core/formulas/norms/SqDist.h"


/*
 * The file where the most useful kernel-related operators are defined.
 *
 * Available kernel-related routines are :
 *   Radial functions :
 *      GaussFunction<R2,C>                     : = exp( - C * R2 )
 *      CauchyFunction<R2,C>                    : = 1/( 1 +  R2 * C )
 *      LaplaceFunction<R2,C>                   : = exp( - sqrt( C * R2 ) )
 *      InverseMultiquadricFunction<R2,C>       : = (1/C + R2)^(-1/2)
 *
 *   Utility functions :
 *      ScalarRadialKernel<F,DIMPOINT,DIMVECT>  : which builds a function
 *                                                (x_i,y_j,b_j) -> F_s( |x_i-y_j|^2 ) * b_j from
 *                                                a radial function F<S,R2> -> ...,
 *                                                a "point" dimension DIMPOINT (x_i and y_j)
 *                                                a "vector" dimension DIMVECT (b_j and output)
 *
 *   Radial Kernel operators : inline expressions w.r.t. x_i = X_0, y_j = Y_1, b_j = Y_2
 *      GaussKernel<DIMPOINT,DIMVECT>                  : uses GaussFunction
 *      CauchyKernel<DIMPOINT,DIMVECT>                 : uses CauchyFunction
 *      LaplaceKernel<DIMPOINT,DIMVECT>                : uses LaplaceFunction
 *      InverseMultiquadricKernel<DIMPOINT,DIMVECT>    : uses InverseMultiquadricFunction
 *
 */

namespace keops {

//////////////////////////////////////////////////////////////
////                 RADIAL FUNCTIONS                     ////
//////////////////////////////////////////////////////////////

template < class R2, class C >
using GaussFunction = Exp<Scal<C,Minus<R2>>>;

template < class R2, class C >
using CauchyFunction = Inv<Add<IntConstant<1>,Scal<C,R2>>>;

template < class R2, class C >
using LaplaceFunction = Exp<Minus<Scal<C,Sqrt<R2>>>>;

template < class R2, class C >
using InverseMultiquadricFunction = Inv<Sqrt<Add< Inv<C>,R2>>>;

template < class R2, class C, class W >
using SumGaussFunction = Scalprod<W,Exp<Scal<Minus<R2>,C>>>;

//////////////////////////////////////////////////////////////
////                 SCALAR RADIAL KERNELS                ////
//////////////////////////////////////////////////////////////

// Utility function

// for some reason the following variadic template version should work but the nvcc compiler does not like it :
//template < class X, class Y, class B, template<class,class...> class F, class... PARAMS >
//using ScalarRadialKernel = Scal<F<SqDist<X,Y>,PARAMS...>,B>;

// so we use two distinct ScalarRadialKernel aliases, depending on the number of parameters :

template < class X, class Y, class B, template<class,class> class F, class PARAMS >
using ScalarRadialKernel_1 = Scal<F<SqDist<X,Y>,PARAMS>,B>;

template < class X, class Y, class B, template<class,class,class> class F, class PARAMS1, class PARAMS2 >
using ScalarRadialKernel_2 = Scal<F<SqDist<X,Y>,PARAMS1,PARAMS2>,B>;

}
