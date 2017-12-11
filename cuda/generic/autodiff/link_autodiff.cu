// (echo "#define FORMULA Scal<Square<Scalprod<X<2,4>,Y<3,4>>>,GaussKernel<P<0>,X<0,3>,Y<1,3>,Y<4,3>>>" ; cat link_autodiff.cu) | nvcc -x cu -std=c++11 -Xcompiler -fPIC -shared -o link_autodiff.so -I $PWD -

#include "GpuConv2D.cu"
#include "autodiff.h"

#include <stdarg.h>
#include <stdio.h>

template < int N, int DIM >
using X = Var<N,DIM,0>;

template < int N, int DIM >
using Y = Var<N,DIM,1>;

template < int N >
using P = Param<N>;

// sum over j : gamma_i = sum_j F(X_i,Y_j)
extern "C" int GpuConv(float* params, int nx, int ny, float* gamma, float** args) {
    return GpuConv2D(Generic<FORMULA>::sEval(), params, nx, ny, gamma, args);
}

extern "C" int CpuConv(float* params, int nx, int ny, float* gamma, float** args) {
    return CpuConv(Generic<FORMULA>::sEval(), params, nx, ny, gamma, args);
}

// sum over i : gamma_j = sum_i F(X_i,Y_j)
extern "C" int GpuTransConv(float* params, int nx, int ny, float* gamma, float** args) {
    return GpuConv2D(Generic<FORMULA,1>::sEval(), params, ny, nx, gamma, args);
}

extern "C" int CpuTransConv(float* params, int nx, int ny, float* gamma, float** args) {
    return CpuConv(Generic<FORMULA,1>::sEval(), params, ny, nx, gamma, args);
}


