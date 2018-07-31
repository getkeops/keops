// nvcc -I.. -Wno-deprecated-gpu-targets -DCUDA_BLOCK_SIZE=192 -D__TYPE__=float -std=c++11 -Xcompiler -fPIC -shared -o build/simple_autodiff.so simple_autodiff.cu

#include "core/formulas/constants.h"
#include "core/formulas/maths.h"
#include "core/formulas/kernels.h"
#include "core/formulas/norms.h"
#include "core/formulas/factorize.h"

#include "core/GpuConv2D.cu"

using namespace keops;


// define variables
using X = Var<1,3>; 	// X is the second variable and represents a 3D vector
using Y = Var<2,3>; 	// Y is the third variable and represents a 3D vector
using U = Var<3,4>; 	// U is the fourth variable and represents a 4D vector
using V = Var<4,4>; 	// V is the fifth variable and represents a 4D vector
using Beta = Var<5,3>;	// Beta is the sixth variable and represents a 3D vector
using C = Param<0,1>;	// C is the first variable and is a scalar parameter

// define F = <U,V>^2 * exp(-C*|X-Y|^2) * Beta in usual notations
using F = Scal<Square<Scalprod<U,V>>,Scal<Exp<Scal<C,Minus<SqNorm2<Subtract<X,Y>>>>>,Beta>>;

using FUNCONVF = SumReduction<F>;

extern "C" int FConv(float ooSigma2, float* x, float* y, float* u, float* v, float* beta, float* gamma, int nx, int ny) {
    float params[1];
    params[0] = ooSigma2;
    return FUNCONVF::Eval<GpuConv2D_FromHost>(nx, ny, gamma, params, x, y, u, v, beta);
}


// now define the gradient wrt XX
using Eta = Var<6,FUNCONVF::DIM>;	// new variable is in seventh position and is input of gradient

using FUNCONVGX = Grad<FUNCONVF,X,Eta>;

extern "C" int GXConv(float ooSigma2, float* x, float* y, float* u, float* v, float* beta, float* eta, float* gamma, int nx, int ny) {
    float params[1];
    params[0] = ooSigma2;
    return FUNCONVGX::Eval<GpuConv2D_FromHost>(nx, ny, gamma, params, x, y, u, v, beta, eta);
}


// now define the gradient wrt Y. 

using FUNCONVGY = Grad<FUNCONVF,Y,Eta>;

extern "C" int GYConv(float ooSigma2, float* x, float* y, float* u, float* v, float* beta, float* eta, float* gamma, int nx, int ny) {
    float params[1];
    params[0] = ooSigma2;
    return FUNCONVGY::Eval<GpuConv2D_FromHost>(ny, nx, params, gamma, x, y, u, v, beta, eta);
}



