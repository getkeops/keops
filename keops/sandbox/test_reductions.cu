// Example of reductions using KeOps C++ / Cuda routines.
// This code performs the following computations :
//       res_i = Red_j F(i,j)
// where : 
//    - F is a Gaussian kernel F(i,j) = exp(-c*|x_i-y_j|^2)*b_j 
//    - the x_i are Nx=10000 points in R^3 
//    - the y_j are Ny=10000 points in R^3
//    - c is a scalar parameter
//    - the b_j are Ny vectors in R^2 
//    - Red_j represents a reduction operation, computed for each index i over the j indices
//        This reduction is vectorized, meaning that reduction is applied separately to each coordinate
//        of F(i,j).
//        Reductions tested hare are the following :
//        * Sum : returns the sum of F(i,j) values,
//        * Min : returns the minimum of F(i,j) values,
//        * ArgMin : returns the indices of minimal F(i,j) values,
//        * MinArgMin : returns minimal values and corresponding indices,
//        * ArgKMin : returns the indices of K first minimal values of F(i,j) (here K=3), 
//        * KMinArgKMin : returns values and indices of K first minimal values of F(i,j) (here K=3), 
//
// This program runs on CPU ; see the file test_reductions.cu for the equivalent program on GPU.
//
// This example can be compiled with the command
//		nvcc -I.. -DCUDA_BLOCK_SIZE=192 -DMAXTHREADSPERBLOCK0=1024 -DSHAREDMEMPERBLOCK0=49152 -Wno-deprecated-gpu-targets -D__TYPE__=float -std=c++11 -O3 -o build/test_reductions test_reductions.cu
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
#include "core/reductions/sum.h"
#include "core/reductions/min.h"
#include "core/reductions/kmin.h"

using namespace keops;

__TYPE__ floatrand() {
    return ((__TYPE__) std::rand())/RAND_MAX-.5;    // random value between -.5 and .5
}

template < class V > void fillrandom(V& v) {
    generate(v.begin(), v.end(), floatrand);    // fills vector with random values
}

void DispValues(__TYPE__ *x, int N, int dim) {
  std::cout << std::endl;
  int k = 0;
  for(int i=0; i<N; i++) {
    for(int d=0; d<dim; d++) {
      std::cout << x[k] << " ";
      k++;
    }
    std::cout << std::endl;
  }
  for(int d=0; d<dim; d++)
    std::cout << "... ";
  std::cout << std::endl << std::endl;
}

#define DIMPOINT 3
#define DIMVECT 2

template < class RED > void DoTest(std::string red_id, int Nx, int Ny, __TYPE__ *oos2, __TYPE__ *x, __TYPE__ *y, __TYPE__ *b) {
    std::cout << "Testing " << red_id << " reduction" << std::endl;

    std::vector<__TYPE__> vres(Nx*RED::DIM);    fillrandom(vres); __TYPE__ *res = vres.data();

    clock_t begin, end;

    begin = clock();
    Eval<RED,CpuConv>::Run(Nx, Ny, res, oos2, x, y, b);
    end = clock();

    std::cout << "time for CPU computation : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    std::vector<__TYPE__> rescpu(Nx*RED::DIM);
    rescpu = vres;
    std::cout << "output = ";
    DispValues(rescpu,5,RED::DIM);

    Eval<RED,GpuConv1D_FromHost>::Run(Nx, Ny, res, oos2, x, y, b);	// first dummy call to Gpu

    begin = clock();
    Eval<RED,GpuConv1D_FromHost>::Run(Nx, Ny, res, oos2, x, y, b);
    end = clock();

    std::cout << "time for GPU computation (1D scheme) : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    std::vector<__TYPE__> resgpu1(Nx*RED::DIM);
    resgpu1 = vres;
    std::cout << "output = ";
    DispValues(resgpu1,5,RED::DIM);

    begin = clock();
    Eval<RED,GpuConv2D_FromHost>::Run(Nx, Ny, res, oos2, x, y, b);
    end = clock();

    std::cout << "time for GPU computation (2D scheme) : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    std::vector<__TYPE__> resgpu2(Nx*RED::DIM);
    resgpu2 = vres;
    std::cout << "output = ";
    DispValues(resgpu2,5,RED::DIM);

}

int main() {

    // symbolic expression of the function : a gaussian kernel
    using C = _P<0,1>;
    using X = _X<1,DIMPOINT>;
    using Y = _Y<2,DIMPOINT>;
    using B = _Y<3,DIMVECT>;
    
    using F = GaussKernel<C,X,Y,B>;

    std::cout << std::endl << "Function F : " << std::endl;
    std::cout << PrintFormula<F>();
    std::cout << std::endl << std::endl;

    // now we test ------------------------------------------------------------------------------

    int Nx=5000, Ny=2000;
        
    
    std::vector<__TYPE__> vx(Nx*DIMPOINT);    fillrandom(vx); __TYPE__ *x = vx.data();
    std::vector<__TYPE__> vy(Ny*DIMPOINT);    fillrandom(vy); __TYPE__ *y = vy.data();
    std::vector<__TYPE__> vb(Ny*DIMVECT); fillrandom(vb); __TYPE__ *b = vb.data();

    __TYPE__ oos2[1] = {.5};

    DoTest < Sum_Reduction<F> >("Sum",Nx, Ny, oos2, x, y, b);
    DoTest < Min_Reduction<F> >("Min",Nx, Ny, oos2, x, y, b);
    DoTest < ArgMin_Reduction<F> >("ArgMin",Nx, Ny, oos2, x, y, b);
    DoTest < Min_ArgMin_Reduction<F> >("Min_ArgMin",Nx, Ny, oos2, x, y, b);
    DoTest < ArgKMin_Reduction<F,3> >("ArgKMin",Nx, Ny, oos2, x, y, b);
    DoTest < KMin_ArgKMin_Reduction<F,3> >("KMin_ArgKMin",Nx, Ny, oos2, x, y, b);
}



