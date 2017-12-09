
// nvcc -std=c++11 -Xcompiler -fPIC -shared -o simple_autodiff.so simple_autodiff.cu


#include "GpuConv2D.cu"
#include "autodiff.h"

// define variables
using X = Var<0,3>; 	// X is the first variable and represents a 3D vector
using Y = Var<1,3>; 	// Y is the second variable and represents a 3D vector
using U = Var<2,4>; 	// U is the third variable and represents a 4D vector
using V = Var<3,4>; 	// V is the fourth variable and represents a 4D vector
using Beta = Var<4,3>;	// Beta is the fifth variable and represents a 3D vector
using C = Param<0>;		// C is the first extra parameter

// define F = <U,V>^2 * exp(-C*|X-Y|^2) * Beta in usual notations
using F = Scal<Square<Scalprod<U,V>>,GaussKernel<C,X,Y,Beta>>;

using FUNCONVF = typename Generic<F>::sEval;

extern "C" int FConv(float ooSigma2, float* x, float* y, float* u, float* v, float* beta, float* gamma, int nx, int ny) 
{
	float params[1];
	params[0] = ooSigma2;
	return GpuConv2D(FUNCONVF(), params, nx, ny, gamma, x, y, u, v, beta); 
}


// now define the gradient wrt X
using Eta = Var<5,F::DIM>;	// new variable is in sixth position and is input of gradient
using GX = Grad<F,X,Eta>;

using FUNCONVGX = typename Generic<GX>::sEval;

extern "C" int GXConv(float ooSigma2, float* x, float* y, float* u, float* v, float* beta, float* eta, float* gamma, int nx, int ny) 
{
	float params[1];
	params[0] = ooSigma2;
	return GpuConv2D(FUNCONVGX(), params, nx, ny, gamma, x, y, u, v, beta, eta); 
}


// now define the gradient wrt Y. There may be still a problem with indices in this part...
using GY = Grad<F,Y,Eta>;

// since Y is a j variable, all i variables become j variables and conversely : this is why we put 1 as second template argument after GY :
using FUNCONVGY = typename Generic<GY,1>::sEval;

extern "C" int GYConv(float ooSigma2, float* x, float* y, float* u, float* v, float* beta, float* eta, float* gamma, int nx, int ny) 
{
	float params[1];
	params[0] = ooSigma2;
	return GpuConv2D(FUNCONVGY(), params, nx, ny, gamma, x, y, u, v, beta, eta); 
}



