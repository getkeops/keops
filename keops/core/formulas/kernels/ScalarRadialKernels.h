#pragma once

#include "core/formulas/constants/IntConst.h"
#include "core/formulas/maths/Exp.h"
#include "core/formulas/maths/Scal.h"
#include "core/formulas/maths/Minus.h"
#include "core/formulas/maths/Add.h"
#include "core/formulas/maths/Inv.h"
#include "core/formulas/maths/Sqrt.h"
#include "core/formulas/norms/Scalprod.h"
#include "core/formulas/norms/SqDist.h"


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
