// test convolution using specific formula for Gauss kernel
// compile with
//		g++ -I.. -D__TYPE__=float -std=c++14 -O3 -o build/test_specific test_specific.cpp

// we compare a generic implementation of the Gauss kernel vs the specific
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

__TYPE__ floatone() {
    return ((__TYPE__) 1.0);    
}

template < class V > void fillones(V& v) {
    generate(v.begin(), v.end(), floatone);    // fills vector with ones
}

int main() {

    // symbolic variables of the function
    using X = Var<1,3,0>; 	// X is the first variable and represents a 3D vector
    using Y = Var<2,3,1>; 	// Y is the second variable and represents a 3D vector
    using B = Var<3,3,1>;	// B is the third variable and represents a 3D vector
    using C = Param<0,1>;	// C is the first extra parameter

    // symbolic expression of the function : Gauss kernel
    using F = GaussKernel<C,X,Y,B>;

    std::cout << std::endl << "Function F : generic Gauss kernel :" << std::endl;
    std::cout << PrintFormula<F>();
    std::cout << std::endl << std::endl;

    using SF = GaussKernel_specific<C,X,Y,B>;

    std::cout << "Function SF = specific Gauss kernel :" << std::endl;    
    std::cout << PrintFormula<SF>();

    using FUNCONVF = Sum_Reduction<F>;
    using FUNCONVSF = Sum_Reduction<SF>;

    // now we test ------------------------------------------------------------------------------

    std::cout << std::endl << std::endl << "Testing F" << std::endl;

    int Nx=5000, Ny=5000;

    std::vector<__TYPE__> vf(Nx*F::DIM);    fillrandom(vf); __TYPE__ *f = vf.data();
    std::vector<__TYPE__> vx(Nx*X::DIM);    fillrandom(vx); __TYPE__ *x = vx.data();
    std::vector<__TYPE__> vy(Ny*Y::DIM);    fillrandom(vy); __TYPE__ *y = vy.data();
    std::vector<__TYPE__> vb(Ny*B::DIM);    fillrandom(vb); __TYPE__ *b = vb.data();

    std::vector<__TYPE__> rescpu1(Nx*F::DIM), rescpu2(Nx*F::DIM);

    __TYPE__ params[1];
    __TYPE__ Sigma = 1;
    params[0] = 1.0/(Sigma*Sigma);

    clock_t begin, end;

    begin = clock();
    Eval<FUNCONVF,CpuConv>::Run(Nx, Ny, f, params, x, y, b);
    end = clock();
    std::cout << "time for CPU computation : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    rescpu1 = vf;

    /// testing SF

    std::cout << std::endl << std::endl << "Testing SF" << std::endl;

    begin = clock();
    Eval<FUNCONVSF,CpuConv>::Run(Nx, Ny, f, params, x, y, b);
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

    // gradient with respect to X ---------------------------------------------------------------
    using Eta = Var<4,F::DIM,0>; // new variable is in seventh position and is input of gradient
	using FUNCONVGX = Grad<FUNCONVF,X,Eta>;
	using FUNCONVSGX = Grad<FUNCONVSF,X,Eta>;
    std::vector<__TYPE__> ve(Nx*Eta::DIM); fillrandom(ve); __TYPE__ *e = ve.data();

    std::cout << "testing gradient wrt X of F" << std::endl;

    begin = clock();
    Eval<FUNCONVGX,CpuConv>::Run(Nx, Ny, f, params, x, y, b, e);
    end = clock();
    std::cout << "time for CPU computation : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    rescpu1 = vf;

    std::cout << "testing gradient wrt X of SF" << std::endl;

    begin = clock();
    Eval<FUNCONVSGX,CpuConv>::Run(Nx, Ny, f, params, x, y, b, e);
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
    s = 0;
    for(int i=0; i<Nx*F::DIM; i++)
        s += std::abs(rescpu1[i]-rescpu2[i]);
    std::cout << std::endl << "mean abs error =" << s/Nx << std::endl;

	
}


