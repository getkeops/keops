
// (echo "#define FORMULA Scal<Square<Scalprod<X<2,4>,Y<3,4>>>,GaussKernel<P<0>,X<0,3>,Y<1,3>,Y<4,3>>>" ; cat link_autodiff.cu) | nvcc -x cu -std=c++11 -Xcompiler -fPIC -shared -o link_autodiff.so -I $PWD -

// ./compile "Scal<Square<Scalprod<X<2,4>,Y<3,4>>>,GaussKernel<P<0>,X<0,3>,Y<1,3>,Y<4,3>>>"
// ./compile "GaussKernel<P<0>,X<0,3>,Y<1,3>,Y<4,3>>"
// ./compile "X<0,3>"

// nvcc -std=c++11 -Xcompiler -fPIC -shared -o link_autodiff.so link_autodiff.cu

#include "GpuConv2D.cu"
#include "autodiff.h"

template < int N, int DIM >
using X = Var<N,DIM,0>;

template < int N, int DIM >
using Y = Var<N,DIM,1>;

template < int N >
using P = Param<N>;

extern "C" int Conv(float ooSigma2, float* x, float* y, float* u, float* v, float* beta, float* gamma, int nx, int ny) 
{
	float params[1];
	params[0] = ooSigma2;
	return GpuConv2D(Generic<FORMULA>::sEval(), params, nx, ny, gamma, x, y, u, v, beta); 
}

