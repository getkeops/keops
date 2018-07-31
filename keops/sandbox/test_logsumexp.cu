// test convolution
// compile with
//		nvcc -I.. -DCUDA_BLOCK_SIZE=192 -Wno-deprecated-gpu-targets -D__TYPE__=float -std=c++11 -O2 -o build/test_logsumexp test_logsumexp.cu
// 

#include <stdio.h>
#include <assert.h>
#include <vector>
#include <cuda.h>
#include <ctime>
#include <algorithm>
#include <iostream>

#include "core/formulas/constants.h"
#include "core/formulas/maths.h"
#include "core/formulas/kernels.h"
#include "core/formulas/norms.h"
#include "core/formulas/factorize.h"

#include "core/CpuConv.cpp"
#include "core/GpuConv1D.cu"
#include "core/GpuConv2D.cu"
#include "core/reductions/log_sum_exp.h"
#include "core/reductions/sum.h"

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

    using FUNCONVF = LogSumExpReduction<F>;

    using ExpF = Exp<F>;

    std::cout << std::endl << "Function ExpF : " << std::endl;
    std::cout << PrintFormula<ExpF>();
    std::cout << std::endl << std::endl;

    using FUNCONVEXPF = SumReduction<ExpF>;

    // now we test ------------------------------------------------------------------------------

    std::cout << "Testing LogSumExp reduction" << std::endl;

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
    
    std::cout << "Testing Log of Sum reduction of Exp" << std::endl;

    std::vector<__TYPE__> rescpu2(Nx*FUNCONVEXPF::DIM);

    begin = clock();
    Eval<FUNCONVEXPF,CpuConv>::Run(Nx, Ny, f, oos2, x, y, b);
    for(int i=0; i<vf.size(); i++)
    	vf[i] = log(vf[i]);
    end = clock();
    std::cout << "time for CPU computation : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    rescpu2 = vf;


    // display values
    std::cout << "rescpu2 = ";
    for(int i=0; i<5; i++)
        std::cout << rescpu2[i] << " ";
    std::cout << "..." << std::endl << std::endl;

    // display mean of errors
    __TYPE__ s = 0;
    for(int i=0; i<Nx*FUNCONVF::DIM; i++)
        s += std::abs(rescpu1[i]-rescpu2[i]);
    std::cout << std::endl << "mean abs error = " << s/Nx << std::endl << std::endl;

    Eval<FUNCONVF,GpuConv1D_FromHost>::Run(Nx, Ny, f, oos2, x, y, b);	// first dummy call to Gpu

    begin = clock();
    Eval<FUNCONVF,GpuConv1D_FromHost>::Run(Nx, Ny, f, oos2, x, y, b);
    end = clock();
    std::cout << "time for GPU computation (1D scheme) : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    std::vector<__TYPE__> resgpu1(Nx*FUNCONVF::DIM);
    resgpu1 = vf;

    // display values
    std::cout << "resgpu1 = ";
    for(int i=0; i<5; i++)
        std::cout << resgpu1[i] << " ";
    std::cout << "..." << std::endl << std::endl;
    
    begin = clock();
    Eval<FUNCONVF,GpuConv2D_FromHost>::Run(Nx, Ny, f, oos2, x, y, b);
    end = clock();
    std::cout << "time for GPU computation (2D scheme) : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    std::vector<__TYPE__> resgpu2(Nx*FUNCONVF::DIM);
    resgpu2 = vf;

    // display values
    std::cout << "resgpu2 = ";
    for(int i=0; i<5; i++)
        std::cout << resgpu2[i] << " ";
    std::cout << "..." << std::endl << std::endl;
    

    std::cout << "Testing Gradient of LogSumExp reduction" << std::endl;

    using E = _X<4,DIMVECT>;
    using GX = Grad<FUNCONVF,X,E>;

    std::vector<__TYPE__> ve(Nx*DIMPOINT); fillrandom(ve); __TYPE__ *e = ve.data();

    rescpu1.resize(Nx*GX::DIM); 
    vf.resize(Nx*GX::DIM);
    f = vf.data();
   
    begin = clock();
    Eval<GX,CpuConv>::Run(Nx, Ny, f, oos2, x, y, b, e);
    end = clock();
    std::cout << "time for CPU computation : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    rescpu1 = vf;

    std::cout << "rescpu1 = ";
    for(int i=0; i<5; i++)
        std::cout << rescpu1[i] << " ";
    std::cout << "..." << std::endl << std::endl;

    resgpu1.resize(Nx*GX::DIM); 
   
    begin = clock();
    Eval<GX,GpuConv1D_FromHost>::Run(Nx, Ny, f, oos2, x, y, b, e);
    end = clock();
    std::cout << "time for GPU computation (1D scheme) : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    resgpu1 = vf;

    std::cout << "resgpu1 = ";
    for(int i=0; i<5; i++)
        std::cout << resgpu1[i] << " ";
    std::cout << "..." << std::endl << std::endl;

    resgpu2.resize(Nx*GX::DIM); 
    vf.resize(Nx*GX::DIM);
    f = vf.data();
   
    begin = clock();
    Eval<GX,GpuConv1D_FromHost>::Run(Nx, Ny, f, oos2, x, y, b, e);
    end = clock();
    std::cout << "time for GPU computation (2D scheme) : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    resgpu2 = vf;

    std::cout << "resgpu2 = ";
    for(int i=0; i<5; i++)
        std::cout << resgpu2[i] << " ";
    std::cout << "..." << std::endl << std::endl;

    std::cout << "Testing Gradient of Log of Sum reduction of Exp" << std::endl;
	using GX2 = Grad<FUNCONVEXPF,X,E>;
    begin = clock();
	Eval<GX2,GpuConv1D_FromHost>::Run(Nx, Ny, f, oos2, x, y, b, e);
    for(int i=0; i<vf.size(); i++)
    	vf[i] = vf[i]/exp(rescpu2[i/GX::DIM]);
    end = clock();
    std::cout << "time for GPU computation (1D scheme) : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    resgpu2 = vf;
    std::cout << "rescpu2 = ";
    for(int i=0; i<5; i++)
        std::cout << resgpu2[i] << " ";
    std::cout << "..." << std::endl << std::endl;

    // display mean of errors
    for(int i=0; i<Nx*GX::DIM; i++)
        s += std::abs(rescpu1[i]-resgpu2[i]);
    std::cout << std::endl << "mean abs error = " << s/Nx << std::endl << std::endl;



}



