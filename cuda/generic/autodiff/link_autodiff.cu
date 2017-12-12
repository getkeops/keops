// use "compile" file for compilation

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

#include "kernel_library.h"

// sum over j : gamma_i = sum_j F(X_i,Y_j)
extern "C" int GpuConv(__TYPE__* params, int nx, int ny, __TYPE__* gamma, __TYPE__** args) 
{
	return GpuConv2D(Generic<FORMULA>::sEval(), params, nx, ny, gamma, args);
}

extern "C" int CpuConv(__TYPE__* params, int nx, int ny, __TYPE__* gamma, __TYPE__** args) 
{
	return CpuConv(Generic<FORMULA>::sEval(), params, nx, ny, gamma, args);
}

// sum over i : gamma_j = sum_i F(X_i,Y_j)
extern "C" int GpuTransConv(__TYPE__* params, int nx, int ny, __TYPE__* gamma, __TYPE__** args) 
{
	return GpuConv2D(Generic<FORMULA,1>::sEval(), params, ny, nx, gamma, args);
}

extern "C" int CpuTransConv(__TYPE__* params, int nx, int ny, __TYPE__* gamma, __TYPE__** args) 
{
	return CpuConv(Generic<FORMULA,1>::sEval(), params, ny, nx, gamma, args);
}


