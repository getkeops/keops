#pragma once


#include "core/formulas/kernels/ScalarRadialKernels.h"
#include "core/pre_headers.h"


namespace keops {



template < class C, class X, class Y, class B >
using CauchyKernel = ScalarRadialKernel_1<X,Y,B,CauchyFunction,C>;

#define CauchyKernel(C,X,Y,B) KeopsNS<CauchyKernel<decltype(InvKeopsNS(C)),decltype(InvKeopsNS(X)),decltype(InvKeopsNS(Y)),decltype(InvKeopsNS(B))>>()

}
