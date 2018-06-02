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

#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include "core/formulas/constants.h"
#include "core/formulas/maths.h"
#include "core/formulas/kernels.h"
#include "core/formulas/norms.h"
#include "core/formulas/factorize.h"

#include "core/GpuConv1D.cu"
#include "core/GpuConv2D.cu"

using namespace keops;

__TYPE__ floatrand() {
    return ((__TYPE__) std::rand())/RAND_MAX-.5;    // random value between -.5 and .5
}

template < class V > void fillrandom(V& v) {
    generate(v.begin(), v.end(), floatrand);    // fills vector with random values
}

int main() {

    int deviceID = 0;
    cudaSetDevice(deviceID);

    // In this part we define the symbolic variables of the function
    using X = Var<1,3,0>; 	// X is the second variable and represents a 3D vector
    using Y = Var<2,3,1>; 	// Y is the third variable and represents a 3D vector
    using Beta = Var<3,3,1>;	// Beta is the sixth variable and represents a 3D vector
    using C = Param<0,1>;		// C is the first variable and is a scalar parameter

    // symbolic expression of the function ------------------------------------------------------

    // here we define F = exp(-C*|X-Y|^2) * Beta in usual notations
    using F = Scal<Exp<Scal<C,Minus<SqNorm2<Subtract<X,Y>>>>>,Beta>;

    using FUNCONVF = typename Generic<F>::sEval;


    // now we test ------------------------------------------------------------------------------

    int Nx=4000, Ny=60000;

    __TYPE__ *f_d;
    cudaMalloc(&f_d, sizeof(__TYPE__)*(Nx*F::DIM));
	thrust::device_ptr<__TYPE__> f_d_thrust(f_d);
    thrust::fill(f_d_thrust, f_d_thrust + Nx*F::DIM, 3.4);
    
    __TYPE__ *param_d;
    cudaMalloc(&param_d, sizeof(__TYPE__)*C::DIM);
	thrust::device_ptr<__TYPE__> param_d_thrust(param_d);
    thrust::fill(param_d_thrust, param_d_thrust + C::DIM, 1.0);
    
    __TYPE__ *x_d;
    cudaMalloc(&x_d, sizeof(__TYPE__)*(Nx*X::DIM));
	thrust::device_ptr<__TYPE__> x_d_thrust(x_d);
    thrust::fill(x_d_thrust, x_d_thrust + Nx*X::DIM, 1.0);
    
    __TYPE__ *y_d;
    cudaMalloc(&y_d, sizeof(__TYPE__)*(Ny*Y::DIM));
	thrust::device_ptr<__TYPE__> y_d_thrust(y_d);
    thrust::fill(y_d_thrust, y_d_thrust + Ny*Y::DIM, 1.0);
    
    __TYPE__ *b_d;
    cudaMalloc(&b_d, sizeof(__TYPE__)*(Ny*Beta::DIM));
	thrust::device_ptr<__TYPE__> b_d_thrust(b_d);
    thrust::fill(b_d_thrust, b_d_thrust + Ny*Beta::DIM, 1.0);
    
    clock_t begin, end;

    begin = clock();
    end = clock();
    std::cout << "time for GPU initialization : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    std::cout << "blank run" << std::endl;
    begin = clock();
    GpuConv2D_FromDevice(FUNCONVF(), Nx, Ny, f_d, param_d, x_d, y_d, b_d);
    end = clock();
    std::cout << "time for blank run : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    std::cout << "testing function F" << std::endl;
    begin = clock();
    for(int i=0; i<200; i++)
        GpuConv2D_FromDevice(FUNCONVF(), Nx, Ny, f_d, param_d, x_d, y_d, b_d);
    end = clock();
    std::cout << "time for 200 GPU computations (2D scheme) : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    std::vector<__TYPE__> resgpu2D(Nx*F::DIM);     fill(resgpu2D.begin(),resgpu2D.end(),2.5);
    cudaMemcpy(resgpu2D.data(), f_d, Nx*F::DIM*sizeof(__TYPE__), cudaMemcpyDeviceToHost);

    // display output
    std::cout << std::endl << "resgpu2D =";
    for(int i=0; i<10; i++)
      std::cout << " " << resgpu2D[i];
    std::cout << " ..." << std::endl;

    begin = clock();
    for(int i=0; i<200; i++)
        GpuConv1D_FromDevice(FUNCONVF(), Nx, Ny, f_d, param_d, x_d, y_d, b_d);
    end = clock();
    std::cout << "time for 200 GPU computations (1D scheme) : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    std::vector<__TYPE__> resgpu1D(Nx*F::DIM);     fill(resgpu1D.begin(),resgpu1D.end(),3.4);
    cudaMemcpy(resgpu1D.data(), f_d, Nx*F::DIM*sizeof(__TYPE__), cudaMemcpyDeviceToHost);

    // display output
    std::cout << std::endl << "resgpu1D =";
    for(int i=0; i<10; i++)
      std::cout << " " << resgpu1D[i];
    std::cout << " ..." << std::endl;

    // display mean of errors
    __TYPE__ s = 0;
    for(int i=0; i<Nx*F::DIM; i++)
        s += abs(resgpu1D[i]-resgpu2D[i]);
    std::cout << "mean abs error 1D/2D =" << s/Nx << std::endl;


    cudaFree(f_d);
    cudaFree(param_d);
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(b_d);





}



