// test convolution
// compile with
//		g++ -I.. -D__TYPE__=float -std=c++11 -O3 -o build/test_argkmin test_argkmin.cpp
// 

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

#include "core/CpuConv.cpp"
#include "core/reductions/kmin.h"

using namespace keops;

__TYPE__ floatrand() {
    return ((__TYPE__) std::rand())/RAND_MAX-.5;    // random value between -.5 and .5
}

template < class V > void fillrandom(V& v) {
    generate(v.begin(), v.end(), floatrand);    // fills vector with random values
}

#define DIMPOINT 3
#define DIMVECT 1

int main() {

    // symbolic expression of the function : linear combination of 4 gaussians
    using C = _P<0,1>;
    using X = _X<1,DIMPOINT>;
    using Y = _Y<2,DIMPOINT>;
    using B = _Y<3,DIMVECT>;
    
    using F = GaussKernel<C,X,Y,B>;

    std::cout << std::endl << "Function F : " << std::endl;
    std::cout << PrintFormula<F>();
    std::cout << std::endl << std::endl;

    using FUNCONVF = ArgKMin_Reduction<F,3>;

    // now we test ------------------------------------------------------------------------------

    std::cout << "Testing ArgKMin reduction" << std::endl;

    int Nx=5000, Ny=2000;
        
    std::vector<__TYPE__> vf(Nx*FUNCONVF::DIM);    fillrandom(vf); __TYPE__ *f = vf.data();
    std::vector<__TYPE__> vx(Nx*DIMPOINT);    fillrandom(vx); __TYPE__ *x = vx.data();
    std::vector<__TYPE__> vy(Ny*DIMPOINT);    fillrandom(vy); __TYPE__ *y = vy.data();
    std::vector<__TYPE__> vb(Ny*DIMVECT); fillrandom(vb); __TYPE__ *b = vb.data();

    std::vector<__TYPE__> rescpu1(Nx*FUNCONVF::DIM);

    __TYPE__ oos2[1] = {.5};

    clock_t begin, end;

    begin = clock();
    Eval<FUNCONVF,CpuConv>::Run(Nx, Ny, f, oos2, x, y, b);
    end = clock();

    std::cout << "time for CPU computation : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    rescpu1 = vf;

    // display values
    std::cout << "rescpu1 = ";
    for(int i=0; i<5; i++)
        std::cout << rescpu1[i] << " ";
    std::cout << "..." << std::endl << std::endl;
    
}



