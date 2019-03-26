// test convolution with the new syntax
// compile with
//		g++ -I.. -D__TYPE__=float -std=c++11 -O2 -o build/test_newsyntax test_newsyntax.cpp 

// we define an arbitrary function using available blocks and with the new syntax
// then test its convolution on the CPU

// Here we build the function F(x,y,u,v,beta,C) = <u,v>^2 * exp(-C*|x-y|^2) * beta
// where x, y, beta are 3D vectors, and u, v are 4D vectors, C is a scalar parameter
// and the convolution is gamma_i = sum_j F(x_i,y_j,u_i,v_j,beta_j,C)

#include <stdio.h>
#include <assert.h>
#include <vector>
#include <ctime>
#include <algorithm>
#include <iostream>

#include "core/formulas/newsyntax.h"

#include "core/CpuConv.cpp"
#include "core/reductions/sum.h"

using namespace keops;

__TYPE__ floatrand() {
    return ((__TYPE__) std::rand())/RAND_MAX-.5;    // random value between -.5 and .5
}

template < class V > void fillrandom(V& v) {
    generate(v.begin(), v.end(), floatrand);    // fills vector with random values
}


int main() {
    // In this part we define the symbolic variables of the function
    auto X = Vi(0,3); 	// X is the first variable and represents a 3D vector
    auto Y = Vj(1,3); 	// Y is the second variable and represents a 3D vector
    auto U = Vi(2,4); 	// U is the third variable and represents a 4D vector
    auto V = Vj(3,4); 	// V is the fourth variable and represents a 4D vector
    auto B = Vj(4,3);	// B is the fifth variable and represents a 3D vector
    auto C = Pm(5,1);	// C is the sixth variable and represents a scalar (1D vector)

    // symbolic expression of the function ------------------------------------------------------
    
    // here we define F = <u,v>^2 * exp(-c*|x-y|^2) * b in usual notations
    auto g = Square((U|V))*Exp(-C*SqDist(X,Y))*B;
    using F = decltype(InvKeopsNS(g));
    //using F = decltype(InvKeopsNS(g-g+IntCst(2)*g-g-g));

    using FUNCONVF = SumReduction<F>;

   // now we test ------------------------------------------------------------------------------

    int Nx=5000, Ny=2000;

    std::vector<__TYPE__> vf(Nx*F::DIM);    fillrandom(vf); __TYPE__ *f = vf.data();
    std::vector<__TYPE__> vx(Nx*X.DIM);    fillrandom(vx); __TYPE__ *x = vx.data();
    std::vector<__TYPE__> vy(Ny*Y.DIM);    fillrandom(vy); __TYPE__ *y = vy.data();
    std::vector<__TYPE__> vu(Nx*U.DIM);    fillrandom(vu); __TYPE__ *u = vu.data();
    std::vector<__TYPE__> vv(Ny*V.DIM);    fillrandom(vv); __TYPE__ *v = vv.data();
    std::vector<__TYPE__> vb(Ny*B.DIM); fillrandom(vb); __TYPE__ *b = vb.data();
    
    std::vector<__TYPE__> rescpu(Nx*F::DIM);

    __TYPE__ params[1];
    __TYPE__ Sigma = 1;
    params[0] = 1.0/(Sigma*Sigma);

    
    clock_t begin, end;

    std::cout << "testing function F : " << std::endl;
    std::cout << PrintFormula<F>();
    std::cout << std::endl;

    std::cout << "minimal number of arguments : " << std::endl;
    std::cout << FUNCONVF::NMINARGS;
    std::cout << std::endl;

    begin = clock();
    Eval<FUNCONVF,CpuConv>::Run(Nx, Ny, f, x, y, u, v, b, params);
    end = clock();
    std::cout << "time for CPU computation : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    rescpu = vf;

}



