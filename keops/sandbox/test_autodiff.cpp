// test convolution with autodiff
// compile with
//		g++ -I.. -D__TYPE__=float -std=c++11 -O2 -o build/test_autodiff test_autodiff.cpp 

// we define an arbitrary function using available blocks,
// then test its convolution on the CPU, then get its gradient and test again the convolution

// Here we build the function F(x,y,u,v,beta,C) = <u,v>^2 * exp(-C*|x-y|^2) * beta
// where x, y, beta are 3D vectors, and u, v are 4D vectors, C is a scalar parameter
// and the convolution is gamma_i = sum_j F(x_i,y_j,u_i,v_j,beta_j,C)
// then we define G(x,y,u,v,beta,C,eta) = gradient of F with respect to x, with new input variable eta (3D)
// and the new convolution is gamma_i = sum_j G(x_i,y_j,u_i,v_j,beta_j,C,eta_i)
#include <cstdlib>

#include <stdio.h>
#include <assert.h>
#include <vector>
#include <ctime>
#include <algorithm>
#include <iostream>

#include "core/formulas/constants.h"
#include "core/formulas/maths.h"
#include "core/formulas/kernels.h"
#include "core/formulas/norms.h"
#include "core/formulas/factorize.h"

#include "../core/CpuConv.cpp"


using namespace keops;

__TYPE__ floatrand() {
    return ((__TYPE__)std::rand())/RAND_MAX-.5;    // random value between -.5 and .5
}

template < class V > void fillrandom(V& v) {
    generate(v.begin(), v.end(), floatrand);    // fills vector with random values
}

int main() {
    // In this part we define the symbolic variables of the function
    using X = Var<0,3,0>; 	// X is the first variable and represents a 3D vector
    using Y = Var<1,3,1>; 	// Y is the second variable and represents a 3D vector
    using U = Var<2,4,0>; 	// U is the third variable and represents a 4D vector
    using V = Var<3,4,1>; 	// V is the fourth variable and represents a 4D vector
    using Beta = Var<4,3,1>;	// Beta is the fifth variable and represents a 3D vector
    using C = Param<5,1>;	// C is the sixth variable and represents a scalar (1D vector)

    // symbolic expression of the function ------------------------------------------------------
    
    // here we define F = <U,V>^2 * exp(-C*|X-Y|^2) * Beta in usual notations
    using F = SumReduction<Scal<Norm2<U>,Scal<Square<Scalprod<U,V>>, Scal<Exp<Scal<C,Minus<SqNorm2<Subtract<X,Y>>>>>,Beta>>>>;

    // gradient with respect to X ---------------------------------------------------------------
    using Eta = Var<6,F::DIM,0>; // new variable is in seventh position and is input of gradient
    
    using GX = Grad<F,X,Eta>;

    using GY = Grad<F,Y,Eta>;

    // now we test ------------------------------------------------------------------------------

    int Nx=5000, Ny=2000;

    std::vector<__TYPE__> vf(Nx*F::DIM);    fillrandom(vf); __TYPE__ *f = vf.data();
    std::vector<__TYPE__> vx(Nx*X::DIM);    fillrandom(vx); __TYPE__ *x = vx.data();
    std::vector<__TYPE__> vy(Ny*Y::DIM);    fillrandom(vy); __TYPE__ *y = vy.data();
    std::vector<__TYPE__> vu(Nx*U::DIM);    fillrandom(vu); __TYPE__ *u = vu.data();
    std::vector<__TYPE__> vv(Ny*V::DIM);    fillrandom(vv); __TYPE__ *v = vv.data();
    std::vector<__TYPE__> vb(Ny*Beta::DIM); fillrandom(vb); __TYPE__ *b = vb.data();
    
    std::vector<__TYPE__> rescpu(Nx*F::DIM);

    __TYPE__ params[1];
    __TYPE__ Sigma = 1;
    params[0] = 1.0/(Sigma*Sigma);
  
    clock_t begin, end;

    std::cout << "testing function F" << std::endl;

    begin = clock();
    F::Eval<CpuConv>(Nx, Ny, f, x, y, u, v, b, params);
    end = clock();
    std::cout << "time for CPU computation : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    rescpu = vf;

    std::vector<__TYPE__> ve(Nx*Eta::DIM); fillrandom(ve); __TYPE__ *e = ve.data();

    std::cout << "testing function GX" << std::endl;

    begin = clock();
    GX::Eval<CpuConv>(Nx, Ny, f, x, y, u, v, b, params, e);
    end = clock();
    std::cout << "time for CPU computation : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    rescpu = vf;

    // gradient wrt Y, which is a "j" variable.

    rescpu.resize(Ny*GY::DIM); 
    vf.resize(Ny*GY::DIM);
    f = vf.data();

    std::cout << "testing function GY" << std::endl;

    begin = clock();
    GY::Eval<CpuConv>(Nx, Ny, f, x, y, u, v, b, params, e);
    end = clock();
    std::cout << "time for CPU computation : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    rescpu = vf;

}



