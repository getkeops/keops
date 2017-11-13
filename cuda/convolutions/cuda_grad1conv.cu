/*
   This file is part of the libds by b. Charlier and J. Glaunes
*/

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include "radial_kernels.cx"
#include "cuda_grad1conv.cx"

#define UseCudaOnDoubles USE_DOUBLE_PRECISION


//////////////////////////////////////////////////////
/////////// CPU -> GPU -> CPU routines ///////////////
//////////////////////////////////////////////////////

template < typename TYPE, KernelFun KernelFp >
int KernelGpuGrad1Conv(TYPE ooSigma2,
                                TYPE* alpha_h, TYPE* x_h, TYPE* y_h, TYPE* beta_h, TYPE* gamma_h,
                                int dimPoint, int dimVect, int nx, int ny){

    // Data on the device.
    TYPE* x_d;
    TYPE* y_d;
    TYPE* alpha_d;
    TYPE* gamma_d;
    TYPE* beta_d;

    // Allocate arrays on device.
    cudaMalloc((void**)&x_d,     sizeof(TYPE)*(nx*dimPoint));
    cudaMalloc((void**)&y_d,     sizeof(TYPE)*(ny*dimPoint));
    cudaMalloc((void**)&alpha_d, sizeof(TYPE)*(nx*dimVect ));
    cudaMalloc((void**)&beta_d,  sizeof(TYPE)*(ny*dimVect ));
    cudaMalloc((void**)&gamma_d, sizeof(TYPE)*(nx*dimPoint));

    // Send data from host to device.
    cudaMemcpy(x_d,     x_h,     sizeof(TYPE)*(nx*dimPoint), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d,     y_h,     sizeof(TYPE)*(ny*dimPoint), cudaMemcpyHostToDevice);
    cudaMemcpy(alpha_d, alpha_h, sizeof(TYPE)*(nx*dimVect ), cudaMemcpyHostToDevice);
    cudaMemcpy(beta_d,  beta_h,  sizeof(TYPE)*(ny*dimVect ), cudaMemcpyHostToDevice);

    // compute on device.
    dim3 blockSize;
    blockSize.x = CUDA_BLOCK_SIZE; // number of threads in each block
    dim3 gridSize;
    gridSize.x =  nx / blockSize.x + (nx%blockSize.x==0 ? 0 : 1);
    
    // Copy-paste templating, allowing us to pass the DIMPOINT and DIMVECT at compilation time : 
    if(     dimPoint==1 && dimVect==1)
        KernelGpuGrad1ConvOnDevice<TYPE,1,1,KernelFp><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect)*sizeof(TYPE)>>>
            (ooSigma2, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==2 && dimVect==1)
        KernelGpuGrad1ConvOnDevice<TYPE,2,1,KernelFp><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect)*sizeof(TYPE)>>>
            (ooSigma2, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==3 && dimVect==1)
        KernelGpuGrad1ConvOnDevice<TYPE,3,1,KernelFp><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect)*sizeof(TYPE)>>>
            (ooSigma2, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==4 && dimVect==1)
        KernelGpuGrad1ConvOnDevice<TYPE,4,1,KernelFp><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect)*sizeof(TYPE)>>>
            (ooSigma2, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==2 && dimVect==2)
        KernelGpuGrad1ConvOnDevice<TYPE,2,2,KernelFp><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect)*sizeof(TYPE)>>>
            (ooSigma2, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==3 && dimVect==3)
        KernelGpuGrad1ConvOnDevice<TYPE,3,3,KernelFp><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect)*sizeof(TYPE)>>>
            (ooSigma2, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==4 && dimVect==4)
        KernelGpuGrad1ConvOnDevice<TYPE,4,4,KernelFp><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect)*sizeof(TYPE)>>>
            (ooSigma2, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else
    {
        printf("GaussGpuGrad1Conv error: dimensions of Gauss kernel not implemented in cuda\nYou probably just need a copy-paste in the conda_grad1conv.cu file !");
        cudaFree(x_d);
        cudaFree(y_d);
        cudaFree(alpha_d);
        cudaFree(gamma_d);
        cudaFree(beta_d);
        return(-1);
    }

    // block until the device has completed
    cudaThreadSynchronize();

    // Send data from device to host.
    cudaMemcpy(gamma_h, gamma_d, sizeof(TYPE)*(nx*dimPoint),cudaMemcpyDeviceToHost);

    // Free memory.
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(alpha_d);
    cudaFree(gamma_d);
    cudaFree(beta_d);

    return 0;
}

#if !(UseCudaOnDoubles) 

// Couldn't find a clean way to give a name to an explicit instantiation :-(
extern "C" int GaussGpuGrad1Conv(float ooSigma2, float* alpha_h, float* x_h, float* y_h, float* beta_h, float* gamma_h, int dimPoint, int dimVect, int nx, int ny) {
    return KernelGpuGrad1Conv<float,GaussFp>(ooSigma2, alpha_h, x_h, y_h, beta_h, gamma_h, dimPoint, dimVect, nx, ny);
}
extern "C" int LaplaceGpuGrad1Conv(float ooSigma2, float* alpha_h, float* x_h, float* y_h, float* beta_h, float* gamma_h, int dimPoint, int dimVect, int nx, int ny) {
    return KernelGpuGrad1Conv<float,LaplaceFp>(ooSigma2, alpha_h, x_h, y_h, beta_h, gamma_h, dimPoint, dimVect, nx, ny);
}
extern "C" int EnergyGpuGrad1Conv(float ooSigma2, float* alpha_h, float* x_h, float* y_h, float* beta_h, float* gamma_h, int dimPoint, int dimVect, int nx, int ny) {
    return KernelGpuGrad1Conv<float,EnergyFp>(ooSigma2, alpha_h, x_h, y_h, beta_h, gamma_h, dimPoint, dimVect, nx, ny);
}

#else
extern "C" int GaussGpuGrad1Conv(double ooSigma2, double* alpha_h, double* x_h, double* y_h, double* beta_h, double* gamma_h, int dimPoint, int dimVect, int nx, int ny) {
    return KernelGpuGrad1Conv<double,GaussFp>(ooSigma2, alpha_h, x_h, y_h, beta_h, gamma_h, dimPoint, dimVect, nx, ny);
}
extern "C" int LaplaceGpuGrad1Conv(double ooSigma2, double* alpha_h, double* x_h, double* y_h, double* beta_h, double* gamma_h, int dimPoint, int dimVect, int nx, int ny) {
    return KernelGpuGrad1Conv<double,LaplaceFp>(ooSigma2, alpha_h, x_h, y_h, beta_h, gamma_h, dimPoint, dimVect, nx, ny);
}
extern "C" int EnergyGpuGrad1Conv(double ooSigma2, double* alpha_h, double* x_h, double* y_h, double* beta_h, double* gamma_h, int dimPoint, int dimVect, int nx, int ny) {
    return KernelGpuGrad1Conv<double,EnergyFp>(ooSigma2, alpha_h, x_h, y_h, beta_h, gamma_h, dimPoint, dimVect, nx, ny);
}

#endif



void ExitFcn(void)
{
    cudaDeviceReset();
}
