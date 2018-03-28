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

#define __TYPE__ double
#define CUDA_BLOCK_SIZE 192 

#include "core/GpuConv1D.cu"
#include "core/GpuConv2D.cu"

#include "core/autodiff.h"

#include "core/formulas/constants.h"
#include "core/formulas/maths.h"
#include "core/formulas/kernels.h"
#include "core/formulas/norms.h"
#include "core/formulas/factorize.h"

#include <thrust/device_ptr.h>
#include <thrust/fill.h>

using namespace std;



int main() {
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

    int Nx=5000, Ny=2000;
    
    __TYPE__ *f_d;
    cudaMalloc(&f_d, sizeof(__TYPE__)*(Nx*F::DIM));
    
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
    int deviceID = 1;
    cudaSetDevice(deviceID);
    end = clock();
    cout << "time for GPU initialization : " << double(end - begin) / CLOCKS_PER_SEC << endl;

    cout << "blank run" << endl;
    begin = clock();
    GpuConv2D_FromDevice(FUNCONVF(), Nx, Ny, f_d, param_d, x_d, y_d, b_d);
    end = clock();
    cout << "time for blank run : " << double(end - begin) / CLOCKS_PER_SEC << endl;

    cout << "testing function F" << endl;
    begin = clock();
    GpuConv2D_FromDevice(FUNCONVF(), Nx, Ny, f_d, param_d, x_d, y_d, b_d);
    end = clock();
    cout << "time for GPU computation (2D scheme) : " << double(end - begin) / CLOCKS_PER_SEC << endl;

	vector<__TYPE__> resgpu2D(Nx*F::DIM);
    cudaMemcpy(f_d, resgpu2D.data(), Nx*F::DIM*sizeof(__TYPE__*), cudaMemcpyDeviceToHost);

    begin = clock();
    GpuConv1D_FromDevice(FUNCONVF(), Nx, Ny, f_d, param_d, x_d, y_d, b_d);
    end = clock();
    cout << "time for GPU computation (1D scheme) : " << double(end - begin) / CLOCKS_PER_SEC << endl;

	vector<__TYPE__> resgpu1D(Nx*F::DIM);
    cudaMemcpy(f_d, resgpu1D.data(), Nx*F::DIM*sizeof(__TYPE__*), cudaMemcpyDeviceToHost);

    // display mean of errors
    __TYPE__ s = 0;
    for(int i=0; i<Nx*F::DIM; i++)
        s += abs(resgpu1D[i]-resgpu2D[i]);
    cout << "mean abs error 1D/2D =" << s/Nx << endl;





}



