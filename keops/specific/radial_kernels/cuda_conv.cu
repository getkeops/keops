#include <stdio.h>
#include <assert.h>
#include <cuda.h>

#include "specific/radial_kernels/radial_kernels.h"
#include "specific/radial_kernels/cuda_conv.cx"


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
    cudaDeviceSynchronize();

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

// This instantiation bypass the function KernelGpuEvalConv as the pointers contain a address directly on the device
extern "C" int GaussGpuEval_onDevice(__TYPE__ ooSigma2, __TYPE__* x_d, __TYPE__* y_d, __TYPE__* beta_d, __TYPE__* gamma_d, int dimPoint, int dimVect, int nx, int ny) {
    dim3 blockSize (CUDA_BLOCK_SIZE,1,1); // number of threads in each block
    dim3 gridSize (nx / blockSize.x + (nx%blockSize.x==0 ? 0 : 1));

    KernelGpuConvOnDevice<__TYPE__,3,3,GaussF><<<gridSize,blockSize,blockSize.x*(3+3)*sizeof(__TYPE__)>>>
    (ooSigma2, x_d, y_d, beta_d, gamma_d, nx, ny);
    return 0;
}

extern "C" int GaussGpuEval(__TYPE__ ooSigma2, __TYPE__* x_h, __TYPE__* y_h, __TYPE__* beta_h, __TYPE__* gamma_h, int dimPoint, int dimVect, int nx, int ny) {
    return KernelGpuEvalConv<__TYPE__,GaussF>(ooSigma2, x_h, y_h, beta_h, gamma_h, dimPoint, dimVect, nx, ny);
}
extern "C" int LaplaceGpuEval(__TYPE__ ooSigma2, __TYPE__* x_h, __TYPE__* y_h, __TYPE__* beta_h, __TYPE__* gamma_h, int dimPoint, int dimVect, int nx, int ny) {
    return KernelGpuEvalConv<__TYPE__,LaplaceF>(ooSigma2, x_h, y_h, beta_h, gamma_h, dimPoint, dimVect, nx, ny);
}
extern "C" int InverseMultiquadricGpuEval(__TYPE__ ooSigma2, __TYPE__* x_h, __TYPE__* y_h, __TYPE__* beta_h, __TYPE__* gamma_h, int dimPoint, int dimVect, int nx, int ny) {
    return KernelGpuEvalConv<__TYPE__,InverseMultiquadricF>(ooSigma2, x_h, y_h, beta_h, gamma_h, dimPoint, dimVect, nx, ny);
}
extern "C" int CauchyGpuEval(__TYPE__ ooSigma2, __TYPE__* x_h, __TYPE__* y_h, __TYPE__* beta_h, __TYPE__* gamma_h, int dimPoint, int dimVect, int nx, int ny) {
    return KernelGpuEvalConv<__TYPE__,CauchyF>(ooSigma2, x_h, y_h, beta_h, gamma_h, dimPoint, dimVect, nx, ny);
}



void ExitFcn(void) {
    cudaDeviceReset();
}

