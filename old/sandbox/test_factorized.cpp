// test convolution using factorized formula
// compile with
//		g++ -I.. -D__TYPE__=float -std=c++14 -O3 -o build/test_factorized test_factorized.cpp

// we define an arbitrary function F,
// then use a factorized version FF of the same function and test
// 

#include <stdio.h>
#include <assert.h>
#include <vector>
#include <ctime>
#include <algorithm>
#include <iostream>

#include <keops_includes.h>

using namespace keops;

__TYPE__ floatrand() {
    return ((__TYPE__) std::rand())/RAND_MAX-.5;    // random value between -.5 and .5
}

template < class V > void fillrandom(V& v) {
    generate(v.begin(), v.end(), floatrand);    // fills vector with random values
}

int main() {
    std::cout << std::endl << "N.B. AutoFactorize is deactivated in all code as of oct 2020 to speed up compile time." << std::endl << std::endl;
	
	/*
    // symbolic variables of the function
    using X = Var<1,3,0>; 	// X is the first variable and represents a 3D vector
    using Y = Var<2,3,1>; 	// Y is the second variable and represents a 3D vector
    using B = Var<3,3,1>;	// B is the fifth variable and represents a 3D vector
    using U = Var<4,3,0>;
    using V = Var<5,3,1>;
    using C = Param<0,1>;		// C is the first extra parameter

    // symbolic expression of the function : 3rd order gradient with respect to X, X and Y of the Gauss kernel
    using F = Grad<Grad<Grad<GaussKernel<C,X,Y,B>,X,U>,X,U>,Y,V>;

    std::cout << std::endl << "Function F : " << std::endl;
    std::cout << PrintFormula<F>();
    std::cout << std::endl << std::endl;

    using FF = AutoFactorize<F>;

    std::cout << "Function FF = factorized version of F :" << std::endl;    
    std::cout << PrintFormula<FF>();

    using FUNCONVF = Sum_Reduction<F>;
    using FUNCONVFF = Sum_Reduction<FF>;

    // now we test ------------------------------------------------------------------------------

    std::cout << std::endl << std::endl << "Testing F" << std::endl;

    int Nx=500, Ny=200;

    std::vector<__TYPE__> vf(Nx*F::DIM);    fillrandom(vf); __TYPE__ *f = vf.data();
    std::vector<__TYPE__> vx(Nx*X::DIM);    fillrandom(vx); __TYPE__ *x = vx.data();
    std::vector<__TYPE__> vy(Ny*Y::DIM);    fillrandom(vy); __TYPE__ *y = vy.data();
    std::vector<__TYPE__> vu(Nx*U::DIM);    fillrandom(vu); __TYPE__ *u = vu.data();
    std::vector<__TYPE__> vv(Ny*V::DIM);    fillrandom(vv); __TYPE__ *v = vv.data();
    std::vector<__TYPE__> vb(Ny*B::DIM); fillrandom(vb); __TYPE__ *b = vb.data();

    std::vector<__TYPE__> rescpu1(Nx*F::DIM), rescpu2(Nx*F::DIM);

    __TYPE__ params[1];
    __TYPE__ Sigma = 1;
    params[0] = 1.0/(Sigma*Sigma);

    clock_t begin, end;

    begin = clock();
    Eval<FUNCONVF,CpuConv>::Run(Nx, Ny, f, params, x, y, b, u, v);
    end = clock();
    std::cout << "time for CPU computation : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    rescpu1 = vf;



    /// testing FF

    std::cout << std::endl << std::endl << "Testing FF" << std::endl;

    begin = clock();
    Eval<FUNCONVFF,CpuConv>::Run(Nx, Ny, f, params, x, y, b, u, v);
    end = clock();
    std::cout << "time for CPU computation : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    rescpu2 = vf;

    // display values
    std::cout << "rescpu1 = ";
    for(int i=0; i<5; i++)
        std::cout << rescpu1[i] << " ";
    std::cout << std::endl << "rescpu2 = ";
    for(int i=0; i<5; i++)
        std::cout << rescpu2[i] << " ";

    // display mean of errors
    __TYPE__ s = 0;
    for(int i=0; i<Nx*F::DIM; i++)
        s += std::abs(rescpu1[i]-rescpu2[i]);
    std::cout << std::endl << "mean abs error =" << s/Nx << std::endl;
	
	*/

}


