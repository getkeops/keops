// test convolution
// compile with
//		g++ -I.. -D__TYPE__=float -std=c++14 -O3 -o build/test_sumgauss test_sumgauss.cpp
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

#define DIMPOINT 3
#define DIMVECT 2

int main() {

    // symbolic expression of the function : linear combination of 4 gaussians
    using C = _P<0,4>;
    using W = _P<1,4>;
    using X = _X<2,DIMPOINT>;
    using Y = _Y<3,DIMPOINT>;
    using B = _Y<4,DIMVECT>;
    using F = SumGaussKernel<C,W,X,Y,B>;

    std::cout << std::endl << "Function F : " << std::endl;
    std::cout << PrintFormula<F>();
    std::cout << std::endl << std::endl;

    using FUNCONVF = Sum_Reduction<F>;

    // now we test ------------------------------------------------------------------------------

    std::cout << "Testing F" << std::endl;

    int Nx=5000, Ny=2000;
        
    std::vector<__TYPE__> vf(Nx*FUNCONVF::DIM);    fillrandom(vf); __TYPE__ *f = vf.data();
    std::vector<__TYPE__> vx(Nx*DIMPOINT);    fillrandom(vx); __TYPE__ *x = vx.data();
    std::vector<__TYPE__> vy(Ny*DIMPOINT);    fillrandom(vy); __TYPE__ *y = vy.data();
    std::vector<__TYPE__> vb(Ny*DIMVECT); fillrandom(vb); __TYPE__ *b = vb.data();

    std::vector<__TYPE__> rescpu1(Nx*FUNCONVF::DIM);

    __TYPE__ oos2s[4] = {.5,.25,.1,1.0};
    __TYPE__ weights[4] = {1.0,-2.0,-.5,3.2};

    clock_t begin, end;

    begin = clock();
    Eval<FUNCONVF,CpuConv>::Run(Nx, Ny, f, oos2s, weights, x, y, b);
    end = clock();
    std::cout << "time for CPU computation : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    rescpu1 = vf;


    // display values
    std::cout << "rescpu1 = ";
    for(int i=0; i<5; i++)
        std::cout << rescpu1[i] << " ";
    std::cout << "..." << std::endl << std::endl;
    
    // compare with combination of 4 convolutions
    std::cout << "Comparing with combination of 4 convolutions" << std::endl;
    using C0 = _P<0,1>;
    using X0 = _X<1,DIMPOINT>;
    using Y0 = _Y<2,DIMPOINT>;
    using B0 = _Y<3,DIMVECT>;
    using F0 = GaussKernel<C0,X0,Y0,B0>;
    using FUNCONVF0 = Sum_Reduction<F0>;
    std::vector<__TYPE__> vf0(Nx*FUNCONVF::DIM);    fillrandom(vf0); __TYPE__ *f0 = vf0.data();
    std::vector<__TYPE__> vf1(Nx*FUNCONVF::DIM);    fillrandom(vf1); __TYPE__ *f1 = vf1.data();
    std::vector<__TYPE__> vf2(Nx*FUNCONVF::DIM);    fillrandom(vf2); __TYPE__ *f2 = vf2.data();
    std::vector<__TYPE__> vf3(Nx*FUNCONVF::DIM);    fillrandom(vf3); __TYPE__ *f3 = vf3.data();
    std::vector<__TYPE__> rescpu2(Nx*FUNCONVF::DIM);
    begin = clock();
    Eval<FUNCONVF0,CpuConv>::Run(Nx, Ny, f0, oos2s, x, y, b);
    Eval<FUNCONVF0,CpuConv>::Run(Nx, Ny, f1, oos2s+1, x, y, b);
    Eval<FUNCONVF0,CpuConv>::Run(Nx, Ny, f2, oos2s+2, x, y, b);
    Eval<FUNCONVF0,CpuConv>::Run(Nx, Ny, f3, oos2s+3, x, y, b);
    for(int i=0; i<Nx*FUNCONVF::DIM; i++)
        f[i] = weights[0]*f0[i]+weights[1]*f1[i]+weights[2]*f2[i]+weights[3]*f3[i];
    end = clock();
    std::cout << "time for CPU computation : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
    rescpu2 = vf;

    // display values
    std::cout << "rescpu2 = ";
    for(int i=0; i<5; i++)
        std::cout << rescpu2[i] << " ";
    std::cout << "..." << std::endl;

    // display mean of errors
    __TYPE__ s = 0;
    for(int i=0; i<Nx*FUNCONVF::DIM; i++)
        s += std::abs(rescpu1[i]-rescpu2[i]);
    std::cout << std::endl << "mean abs error = " << s/Nx << std::endl << std::endl;

    
    
    
    std::cout << "Testing Gradient of F" << std::endl;
    
    using E = _X<5,DIMVECT>;
    using G = Grad<F,X,E>;

    std::cout << std::endl << "Function G : " << std::endl;
    std::cout << PrintFormula<G>();
    std::cout << std::endl << std::endl;

    std::vector<__TYPE__> vg(Nx*G::DIM);    fillrandom(vg); __TYPE__ *g = vg.data();
    std::vector<__TYPE__> ve(Nx*DIMVECT);    fillrandom(ve); __TYPE__ *e = ve.data();

    using FUNCONVG = Sum_Reduction<G>;
    begin = clock();
    Eval<FUNCONVG,CpuConv>::Run(Nx, Ny, g, oos2s, weights, x, y, b, e);
    end = clock();
    std::cout << "time for CPU computation : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
    
    rescpu1 = vg;
    // display values
    std::cout << "rescpu1 = ";
    for(int i=0; i<5; i++)
        std::cout << rescpu1[i] << " ";
    std::cout << "..." << std::endl << std::endl;
    
    std::cout << "Comparing with combination of 4 convolutions" << std::endl;
    using E0 = _X<4,DIMVECT>;
    using G0 = Grad<F0,X0,E0>;
    using FUNCONVG0 = Sum_Reduction<G0>;
    std::vector<__TYPE__> vg0(Nx*G0::DIM);    fillrandom(vg0); __TYPE__ *g0 = vg0.data();
    std::vector<__TYPE__> vg1(Nx*G0::DIM);    fillrandom(vg1); __TYPE__ *g1 = vg1.data();
    std::vector<__TYPE__> vg2(Nx*G0::DIM);    fillrandom(vg2); __TYPE__ *g2 = vg2.data();
    std::vector<__TYPE__> vg3(Nx*G0::DIM);    fillrandom(vg3); __TYPE__ *g3 = vg3.data();
    begin = clock();
    Eval<FUNCONVG0,CpuConv>::Run(Nx, Ny, g0, oos2s, x, y, b, e);
    Eval<FUNCONVG0,CpuConv>::Run(Nx, Ny, g1, oos2s+1, x, y, b, e);
    Eval<FUNCONVG0,CpuConv>::Run(Nx, Ny, g2, oos2s+2, x, y, b, e);
    Eval<FUNCONVG0,CpuConv>::Run(Nx, Ny, g3, oos2s+3, x, y, b, e);
    for(int i=0; i<Nx*G0::DIM; i++)
        g[i] = weights[0]*g0[i]+weights[1]*g1[i]+weights[2]*g2[i]+weights[3]*g3[i];
    end = clock();
    std::cout << "time for CPU computation : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
    rescpu2 = vg;
    
    // display values
    std::cout << "rescpu2 = ";
    for(int i=0; i<5; i++)
        std::cout << rescpu2[i] << " ";
    std::cout << "..." << std::endl;
    
    // display mean of errors
    s = 0;
    for(int i=0; i<Nx*G::DIM; i++)
        s += std::abs(rescpu1[i]-rescpu2[i]);
    std::cout << std::endl << "mean abs error = " << s/Nx << std::endl << std::endl;
}



