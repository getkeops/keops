// test convolution with autodiff
// compile with
//		nvcc -I.. -Wno-deprecated-gpu-targets -std=c++11 -O2 -o build/test_fromdevice test_fromdevice.cu

// testing "from device" convolution, i.e. convolution which is performed on the device
// directly from device data

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <vector>
#include <ctime>
#include <algorithm>

// fix some Gpu properties
// These values should be fine, but you can check them with GetGpuProps.cu program
#ifndef MAXIDGPU
  #define MAXIDGPU 0 // (= number of Gpu devices - 1)
  #define CUDA_BLOCK_SIZE 192
  #define MAXTHREADSPERBLOCK0 1024 
  #define SHAREDMEMPERBLOCK0 49152
#endif 

#ifndef __TYPE__
  #define __TYPE__ float
#endif

#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include <thrust/device_vector.h>
#include <thrust/random.h>

#include "core/formulas/constants.h"
#include "core/formulas/maths.h"
#include "core/formulas/kernels.h"
#include "core/formulas/norms.h"
#include "core/formulas/factorize.h"
#include "core/formulas/newsyntax.h"

#include "core/GpuConv1D.cu"
#include "core/GpuConv2D.cu"
#include "core/reductions/sum.h"

#define DIMPOINT 3
#define DIMVECT 2

struct GenRand
{
    __device__
    float operator () (int idx)
    {
        thrust::default_random_engine randEng;
        thrust::uniform_real_distribution<__TYPE__> uniDist(0,1);
        randEng.discard(idx);
printf("rand=%f\n",uniDist(randEng));
        return uniDist(randEng);
    }
};

void FillRandom(thrust::device_vector<__TYPE__> v) {
    thrust::transform(v.begin(),v.end(),v.begin(),GenRand());
}

using namespace keops;

int main() {

    int deviceID = 0;
    cudaSetDevice(deviceID);

    // symbolic expression of the function : a gaussian kernel
    auto c = Pm(0,1);
    auto x = Vi(1,DIMPOINT);
    auto y = Vj(2,DIMPOINT);
    auto beta = Vj(3,DIMVECT);
    
    auto f = Exp(-c*SqNorm2(x-y)) * beta; 

    std::cout << std::endl << "Function f : " << std::endl;
    std::cout << PrintFormula(f);
    std::cout << std::endl << std::endl;

    auto Sum_f = Sum_Reduction(f,0);

    // now we test ------------------------------------------------------------------------------

    int Nx=4000, Ny=60000;

    thrust::device_vector<__TYPE__> vres_d(Nx*Sum_f.DIM);
    __TYPE__ *res_d = thrust::raw_pointer_cast(vres_d.data());
    
    thrust::device_vector<__TYPE__> vparam_d(c.DIM);
    FillRandom(vparam_d);
    __TYPE__ *param_d = thrust::raw_pointer_cast(vparam_d.data());
    
    thrust::device_vector<__TYPE__> vx_d(Nx*x.DIM);
    FillRandom(vx_d);
    __TYPE__ *x_d = thrust::raw_pointer_cast(vx_d.data());
    
    thrust::device_vector<__TYPE__> vy_d(Ny*y.DIM);
    FillRandom(vy_d);
    __TYPE__ *y_d = thrust::raw_pointer_cast(vy_d.data());
    
    thrust::device_vector<__TYPE__> vb_d(Ny*beta.DIM);
    FillRandom(vb_d);
    __TYPE__ *b_d = thrust::raw_pointer_cast(vb_d.data());
    
    clock_t begin, end;

    std::cout << "blank run 1" << std::endl;
    begin = clock();
    EvalRed<GpuConv2D_FromDevice>(Sum_f,Nx, Ny, res_d, param_d, x_d, y_d, b_d);
    end = clock();
    std::cout << "time for blank run 1 : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    std::cout << "blank run 2" << std::endl;
    begin = clock();
    EvalRed<GpuConv2D_FromDevice>(Sum_f,Nx, Ny, res_d, param_d, x_d, y_d, b_d);
    end = clock();
    std::cout << "time for blank run 2 : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    std::cout << "testing function F" << std::endl;
    begin = clock();
    for(int i=0; i<200; i++)
        EvalRed<GpuConv2D_FromDevice>(Sum_f,Nx, Ny, res_d, param_d, x_d, y_d, b_d);
    end = clock();
    std::cout << "time for 200 GPU computations (2D scheme) : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    std::vector<__TYPE__> resgpu2D(Nx*Sum_f.DIM);
    cudaMemcpy(resgpu2D.data(), res_d, Nx*Sum_f.DIM*sizeof(__TYPE__), cudaMemcpyDeviceToHost);

    // display output
    std::cout << std::endl << "resgpu2D =";
    for(int i=0; i<10; i++)
      std::cout << " " << resgpu2D[i];
    std::cout << " ..." << std::endl;

    begin = clock();
    for(int i=0; i<200; i++)
        EvalRed<GpuConv1D_FromDevice>(Sum_f,Nx, Ny, res_d, param_d, x_d, y_d, b_d);
    end = clock();
    std::cout << "time for 200 GPU computations (1D scheme) : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    thrust::copy(vx_d.begin(), vx_d.begin()+10, std::ostream_iterator<__TYPE__>(std::cout, " "));

    std::vector<__TYPE__> resgpu1D(Nx*Sum_f.DIM);
    cudaMemcpy(resgpu1D.data(), res_d, Nx*Sum_f.DIM*sizeof(__TYPE__), cudaMemcpyDeviceToHost);

    // display output
    std::cout << std::endl << "resgpu1D =";
    for(int i=0; i<10; i++)
      std::cout << " " << resgpu1D[i];
    std::cout << " ..." << std::endl;

    // display mean of errors
    __TYPE__ s = 0;
    for(int i=0; i<Nx*Sum_f.DIM; i++)
        s += std::abs(resgpu1D[i]-resgpu2D[i]);
    std::cout << "mean abs error 1D/2D =" << s/Nx << std::endl;



}



