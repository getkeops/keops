#pragma once


#include "core/formulas/kernels/ScalarRadialKernels.h"
#include "core/pre_headers.h"


namespace keops {



template < class C, class X, class Y, class B >
using LaplaceKernel = ScalarRadialKernel_1<X,Y,B,LaplaceFunction,C>;

#define LaplaceKernel(C,X,Y,B) KeopsNS<LaplaceKernel<decltype(InvKeopsNS(C)),decltype(InvKeopsNS(X)),decltype(InvKeopsNS(Y)),decltype(InvKeopsNS(B))>>()

}
