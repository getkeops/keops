#pragma once

#include "core/formulas/kernels/ScalarRadialKernels.h"
#include "core/pre_headers.h"


namespace keops {



template < class C, class W, class X, class Y, class B >
using SumGaussKernel = ScalarRadialKernel_2<X,Y,B,SumGaussFunction,C,W>;


#define SumGaussKernel(C,W,X,Y,B) KeopsNS<SumGaussKernel<decltype(InvKeopsNS(C)),decltype(InvKeopsNS(W)),decltype(InvKeopsNS(X)),decltype(InvKeopsNS(Y)),decltype(InvKeopsNS(B))>>()

}

