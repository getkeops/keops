#pragma once


#include "core/formulas/kernels/ScalarRadialKernels.h"
#include "core/pre_headers.h"


namespace keops {



template < class C, class X, class Y, class B >
using InverseMultiquadricKernel = ScalarRadialKernel_1<X,Y,B,InverseMultiquadricFunction,C>;

#define InverseMultiquadricKernel(C,X,Y,B) KeopsNS<InverseMultiquadricKernel<decltype(InvKeopsNS(C)),decltype(InvKeopsNS(X)),decltype(InvKeopsNS(Y)),decltype(InvKeopsNS(B))>>()

}

