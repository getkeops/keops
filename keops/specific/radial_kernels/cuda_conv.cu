#include <stdio.h>
#include <assert.h>
#include <cuda.h>

#include <iostream>
#include <vector>
#include <algorithm>

#include <keops_includes.h>
#include "core/mapreduce/GpuConv1D.cu"

#include "specific/radial_kernels/radial_kernels.h"
#include "specific/radial_kernels/cuda_conv.cx"
#include "specific/radial_kernels/cuda_tile.cx"

#define TOTO_DIM 128
#define BLOCK_SIZE 64
#define TILE_SIZE 32

__TYPE__ floatrand() {
    return ((__TYPE__) std::rand())/RAND_MAX-.5;    // random value between -.5 and .5
}

template < class V > void fillrandom(V& v) {
    generate(v.begin(), v.end(), floatrand);    // fills vector with random values
}
using namespace keops;

int main() {

    // now we test ------------------------------------------------------------------------------

    int Nx=10000, Ny=10000;

    // here we define actual data for all variables and feed it it with random values
    std::vector<__TYPE__> vx(Nx*TOTO_DIM); fillrandom(vx); __TYPE__ *px = vx.data();
    std::vector<__TYPE__> vy(Ny*TOTO_DIM); fillrandom(vy); __TYPE__ *py = vy.data();
    std::vector<__TYPE__> vb(Ny*TOTO_DIM); fillrandom(vb); __TYPE__ *pb = vb.data();

    // also a vector for the output
    //std::vector<__TYPE__> vres(Nx*DIM); fillrandom(vres); __TYPE__ *pres = vres.data();

    // and three vectors to get copies of the output, for comparing Cpu vs Gpu/1D vs Gpu/2D computations
    std::vector<__TYPE__> resgpu3D(Nx*TOTO_DIM), resgpu2D(Nx*TOTO_DIM), resgpu1D(Nx*TOTO_DIM);

    // parameter variable
    __TYPE__ ooSigma = 1.0;

    clock_t begin, end;

    std::cout << "blank run" << std::endl;
    KernelGpuEvalConv<__TYPE__,GaussF, TOTO_DIM, BLOCK_SIZE>(ooSigma, px, py, pb, resgpu1D.data(), TOTO_DIM, TOTO_DIM, Nx, Ny);


    begin = clock();
    KernelGpuEvalConv<__TYPE__,GaussF, TOTO_DIM, BLOCK_SIZE>(ooSigma, px, py, pb, resgpu1D.data(), TOTO_DIM, TOTO_DIM, Nx, Ny);
    end = clock();
    std::cout << "time for GPU computation (1D scheme, specific) : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;



    begin = clock();
    KernelGpuEvalTile<__TYPE__,GaussF, TOTO_DIM, BLOCK_SIZE, TILE_SIZE>(ooSigma, px, py, pb, resgpu3D.data(), TOTO_DIM, TOTO_DIM, Nx, Ny);
    end = clock();
    std::cout << "time for GPU computation (1D tiled scheme, specific) : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;








/*
    // In this part we define the symbolic variables of the function
    auto p = Pm(0,1);	 // p is the first variable and is a scalar parameter
    auto x = Vi(1,TOTO_DIM); 	 // x is the second variable and represents a 3D vector, "i"-indexed.
    auto y = Vj(2,TOTO_DIM); 	 // y is the third variable and represents a 3D vector, "j"-indexed.
    auto beta = Vj(3,TOTO_DIM); // beta is the sixth variable and represents a 3D vector, "j"-indexed.
    __TYPE__ params[1];
    params[0] = ooSigma;
    // symbolic expression of the function ------------------------------------------------------

    // here we define f = <u,v>^2 * exp(-p*|x-y|^2) * beta in usual notations
    auto f = Exp(-p*SqNorm2(x-y)) * beta;

    // We define the reduction operation on f. Here a sum reduction, performed over the "j" index, and resulting in a "i"-indexed variable
    auto Sum_f = Sum_Reduction(f,0);  // 0 means output of reduction will be "i"-indexed (0 means"i", 1 means "j")


    begin = clock();
    EvalRed<GpuConv1D_FromHost>(Sum_f, Nx, Ny, resgpu2D.data(), params, px, py, pb);
    end = clock();
    std::cout << "time for GPU computation (1D scheme, generic) : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
*/

    for(int i=0; i<50; i++)
        std::cout << i << " = " << resgpu2D[i] << " " << resgpu1D[i] << " " << resgpu3D[i] << std::endl;

    // display mean of errors
    __TYPE__ s = 0;
    for(int i=0; i<Nx*TOTO_DIM; i++)
        s += std::abs(resgpu1D[i]-resgpu3D[i]);
    std::cout << "mean abs error 2D =" << s << std::endl;


    return 0;

}