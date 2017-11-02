/*
   This file is part of the libds by b. Charlier and J. Glaunes
*/

#include <stdio.h>
#include <assert.h>
#include "cuda_conv.cx"

#define UseCudaOnDoubles USE_DOUBLE_PRECISION

//////////////////////////////////////////////////////
/////////// CPU -> GPU -> CPU routines ///////////////
//////////////////////////////////////////////////////

template < typename TYPE, KernelFun KernelF >
int KernelGpuEvalConv(TYPE ooSigma2,
                                   TYPE* x_h, TYPE* y_h, TYPE* beta_h, TYPE* gamma_h,
                                   int dimPoint, int dimVect, int nx, int ny) {

    // Data on the device.
    TYPE* x_d;
    TYPE* y_d;
    TYPE* beta_d;
    TYPE* gamma_d;

    // Allocate arrays on device.
    cudaMalloc((void**)&x_d,     sizeof(TYPE)*(nx*dimPoint));
    cudaMalloc((void**)&y_d,     sizeof(TYPE)*(ny*dimPoint));
    cudaMalloc((void**)&beta_d,  sizeof(TYPE)*(ny*dimVect ));
    cudaMalloc((void**)&gamma_d, sizeof(TYPE)*(nx*dimVect ));

    // Set values to zeros
    cudaMemset(x_d,    0, sizeof(TYPE)*(nx*dimPoint));
    cudaMemset(y_d,    0, sizeof(TYPE)*(ny*dimPoint));
    cudaMemset(beta_d, 0, sizeof(TYPE)*(ny*dimVect ));
    cudaMemset(gamma_d,0, sizeof(TYPE)*(nx*dimVect ));

    // Send data from host to device.
    cudaMemcpy(x_d,    x_h,    sizeof(TYPE)*(nx*dimPoint), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d,    y_h,    sizeof(TYPE)*(ny*dimPoint), cudaMemcpyHostToDevice);
    cudaMemcpy(beta_d, beta_h, sizeof(TYPE)*(ny*dimVect ), cudaMemcpyHostToDevice);

    // Compute on device.
    dim3 blockSize;
    blockSize.x = CUDA_BLOCK_SIZE; // number of threads in each block
    dim3 gridSize;
    gridSize.x =  nx / blockSize.x + (nx%blockSize.x==0 ? 0 : 1);

    // Copy-paste templating, allowing us to pass the DIMPOINT and DIMVECT at compilation time :
    if(     dimPoint==1 && dimVect==1)
        KernelGpuConvOnDevice<TYPE,1,1,KernelF><<<gridSize,blockSize,blockSize.x*(dimVect+dimPoint)*sizeof(TYPE)>>>
        (ooSigma2, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==3 && dimVect==1)
        KernelGpuConvOnDevice<TYPE,3,1,KernelF><<<gridSize,blockSize,blockSize.x*(dimVect+dimPoint)*sizeof(TYPE)>>>
        (ooSigma2, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==2 && dimVect==1)
        KernelGpuConvOnDevice<TYPE,2,1,KernelF><<<gridSize,blockSize,blockSize.x*(dimVect+dimPoint)*sizeof(TYPE)>>>
        (ooSigma2, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==2 && dimVect==2)
        KernelGpuConvOnDevice<TYPE,2,2,KernelF><<<gridSize,blockSize,blockSize.x*(dimVect+dimPoint)*sizeof(TYPE)>>>
        (ooSigma2, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==3 && dimVect==3)
        KernelGpuConvOnDevice<TYPE,3,3,KernelF><<<gridSize,blockSize,blockSize.x*(dimVect+dimPoint)*sizeof(TYPE)>>>
        (ooSigma2, x_d, y_d, beta_d, gamma_d, nx, ny);
    else {
        printf("Error: dimensions of Gauss kernel not implemented in cuda. You probably just need a copy-paste in the conda_conv.cu file !");
		cudaFree(x_d);
		cudaFree(y_d);
		cudaFree(beta_d);
		cudaFree(gamma_d);
        return(-1);
    }

    // block until the device has completed
    cudaThreadSynchronize();

    // Send data from device to host.
    cudaMemcpy(gamma_h, gamma_d, sizeof(TYPE)*(nx*dimVect),cudaMemcpyDeviceToHost);

    // Free memory.
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(beta_d);
    cudaFree(gamma_d);

    return 0;
}


// Couldn't find a clean way to give a name to an explicit instantiation :-(

#if !(UseCudaOnDoubles) 

extern "C" int GaussGpuEvalConv(float ooSigma2, float* x_h, float* y_h, float* beta_h, float* gamma_h, int dimPoint, int dimVect, int nx, int ny) {
    return KernelGpuEvalConv<float,GaussF>(ooSigma2, x_h, y_h, beta_h, gamma_h, dimPoint, dimVect, nx, ny);
}
extern "C" int LaplaceGpuEvalConv(float ooSigma2, float* x_h, float* y_h, float* beta_h, float* gamma_h, int dimPoint, int dimVect, int nx, int ny) {
    return KernelGpuEvalConv<float,LaplaceF>(ooSigma2, x_h, y_h, beta_h, gamma_h, dimPoint, dimVect, nx, ny);
}
extern "C" int EnergyGpuEvalConv(float ooSigma2, float* x_h, float* y_h, float* beta_h, float* gamma_h, int dimPoint, int dimVect, int nx, int ny) {
    return KernelGpuEvalConv<float,EnergyF>(ooSigma2, x_h, y_h, beta_h, gamma_h, dimPoint, dimVect, nx, ny);
}
#else

extern "C" int GaussGpuEvalConv(double ooSigma2, double* x_h, double* y_h, double* beta_h, double* gamma_h, int dimPoint, int dimVect, int nx, int ny) {
    return KernelGpuEvalConv<double,GaussF>(ooSigma2, x_h, y_h, beta_h, gamma_h, dimPoint, dimVect, nx, ny);
}
extern "C" int LaplaceGpuEvalConv(double ooSigma2, double* x_h, double* y_h, double* beta_h, double* gamma_h, int dimPoint, int dimVect, int nx, int ny) {
    return KernelGpuEvalConv<double,LaplaceF>(ooSigma2, x_h, y_h, beta_h, gamma_h, dimPoint, dimVect, nx, ny);
}
extern "C" int EnergyGpuEvalConv(double ooSigma2, double* x_h, double* y_h, double* beta_h, double* gamma_h, int dimPoint, int dimVect, int nx, int ny) {
    return KernelGpuEvalConv<double,EnergyF>(ooSigma2, x_h, y_h, beta_h, gamma_h, dimPoint, dimVect, nx, ny);
}

#endif



void ExitFcn(void) {
  cudaDeviceReset();
}

