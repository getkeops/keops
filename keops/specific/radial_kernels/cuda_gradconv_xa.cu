/*
*	This cuda routine allows one to compute the derivative wrt the dual vector 'a' of the derivative
*	wrt 'x' of the expression
*		K(x_i,y_j) @ b_j =  sum_j f( |x_i-y_j|^2 ) b_j
*
*
*	We're looking for the gradient with respect to a of
*
*	< e, K(s,a,x,y,b) >  =  \sum_{i,j} f_s'( |x_i-y_j|^2 ) * < a_i, b_j > * 2 < e_i, x_i-y_j>,
*
*	which is an N-by-E array g_i (i from 1 to N), where each line is equal to
*
*	g_i  =  \sum_j 2* f_s'( |x_i-y_j|^2 ) * < e_i, x_i-y_j> * b_j
*
*	We will compute this sum over the index 'j' on the GPU, with 'one thread' = 'one index i'.
*	Data will be stored as follow:
*	  - e_i in the thread memory
*	  - x_i in the thread memory
*	  - y_j in the SharedData
*	  - b_j in the SharedData (beta_j, really)
*
*
* Author : Jean Feydy, heavily based on the work of Joan Glaunès and Benjamin Charlier.
*
*/

#include <stdio.h>
#include <assert.h>
#include <cuda.h>

#include "specific/radial_kernels/radial_kernels.h"
#include "specific/radial_kernels/cuda_gradconv_xa.cx"


//////////////////////////////////////////////////////
/////////// CPU -> GPU -> CPU routines ///////////////
//////////////////////////////////////////////////////


template <typename TYPE, KernelFun KernelFp >
int KernelGpuGradConvXA(TYPE ooSigma2,               // 1 / sigma^2
                        TYPE* e_h,                     // N-by-D array (same as x)
                        TYPE* alpha_h, TYPE* x_h,     // N-by-E, N-by-D arrays
                        TYPE* y_h,     TYPE* beta_h,  // M-by-D, M-by-E arrays
                        TYPE* gamma_h,                 // Output: N-by-E (same as alpha)
                        int dimPoint, int dimVect, int nx, int ny) { // D, E, N, M

    // Data on the device.
    TYPE* e_d;
    TYPE* alpha_d;
    TYPE* x_d;
    TYPE* y_d;
    TYPE* beta_d;
    TYPE* gamma_d;

    // Allocate arrays on device.
    cudaMalloc((void**)&e_d,     sizeof(TYPE)*(nx*dimPoint));
    cudaMalloc((void**)&alpha_d, sizeof(TYPE)*(nx*dimVect ));
    cudaMalloc((void**)&x_d,     sizeof(TYPE)*(nx*dimPoint));
    cudaMalloc((void**)&y_d,     sizeof(TYPE)*(ny*dimPoint));
    cudaMalloc((void**)&beta_d,  sizeof(TYPE)*(ny*dimVect ));
    cudaMalloc((void**)&gamma_d, sizeof(TYPE)*(nx*dimVect )); // Output: N-by-E (same as alpha)

    // Send data from host to device.
    cudaMemcpy(e_d,     e_h,     sizeof(TYPE)*(nx*dimPoint), cudaMemcpyHostToDevice);
    cudaMemcpy(alpha_d, alpha_h, sizeof(TYPE)*(nx*dimVect ), cudaMemcpyHostToDevice);
    cudaMemcpy(x_d,     x_h,     sizeof(TYPE)*(nx*dimPoint), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d,     y_h,     sizeof(TYPE)*(ny*dimPoint), cudaMemcpyHostToDevice);
    cudaMemcpy(beta_d,  beta_h,  sizeof(TYPE)*(ny*dimVect ), cudaMemcpyHostToDevice);

    // compute on device.
    dim3 blockSize;
    blockSize.x = CUDA_BLOCK_SIZE; // number of threads in each block
    dim3 gridSize;
    gridSize.x =  nx / blockSize.x + (nx%blockSize.x==0 ? 0 : 1);

    // Copy-paste templating, allowing us to pass the DIMPOINT and DIMVECT at compilation time :
    if(     dimPoint==1 && dimVect==1)
        KernelGpuGradConvXAOnDevice<TYPE,1,1,KernelFp><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect)*sizeof(TYPE)>>>
        (ooSigma2, e_d, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==2 && dimVect==1)
        KernelGpuGradConvXAOnDevice<TYPE,2,1,KernelFp><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect)*sizeof(TYPE)>>>
        (ooSigma2, e_d, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==3 && dimVect==1)
        KernelGpuGradConvXAOnDevice<TYPE,3,1,KernelFp><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect)*sizeof(TYPE)>>>
        (ooSigma2, e_d, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==4 && dimVect==1)
        KernelGpuGradConvXAOnDevice<TYPE,4,1,KernelFp><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect)*sizeof(TYPE)>>>
        (ooSigma2, e_d, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==2 && dimVect==2)
        KernelGpuGradConvXAOnDevice<TYPE,2,2,KernelFp><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect)*sizeof(TYPE)>>>
        (ooSigma2, e_d, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==3 && dimVect==3)
        KernelGpuGradConvXAOnDevice<TYPE,3,3,KernelFp><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect)*sizeof(TYPE)>>>
        (ooSigma2, e_d, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==4 && dimVect==4)
        KernelGpuGradConvXAOnDevice<TYPE,4,4,KernelFp><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect)*sizeof(TYPE)>>>
        (ooSigma2, e_d, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else {
        printf("GaussGpuGradConvXA error: dimensions of Gauss kernel not implemented in cuda\nYou probably just need a copy-paste in the conda_gradconv_xa.cu file !");
        cudaFree(e_d);
        cudaFree(alpha_d);
        cudaFree(x_d);
        cudaFree(y_d);
        cudaFree(beta_d);
        cudaFree(gamma_d);
        return(-1);
    }

    // block until the device has completed
    cudaDeviceSynchronize();

    // Send data from device to host.
    cudaMemcpy(gamma_h, gamma_d, sizeof(TYPE)*(nx*dimVect),cudaMemcpyDeviceToHost); // Output: N-by-E (same as alpha)

    // Free memory.
    cudaFree(e_d);
    cudaFree(alpha_d);
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(beta_d);
    cudaFree(gamma_d);

    return 0;
}

// Couldn't find a clean way to give a name to an explicit instantiation :-(

extern "C" int GaussGpuGradConvXA(__TYPE__ ooSigma2, __TYPE__* e_h, __TYPE__* alpha_h, __TYPE__* x_h, __TYPE__* y_h, __TYPE__* beta_h, __TYPE__* gamma_h, int dimPoint, int dimVect, int nx, int ny) {
    return KernelGpuGradConvXA<__TYPE__,GaussFp>(ooSigma2, e_h, alpha_h, x_h, y_h, beta_h, gamma_h, dimPoint, dimVect, nx, ny);
}
extern "C" int CauchyGpuGradConvXA(__TYPE__ ooSigma2, __TYPE__* e_h, __TYPE__* alpha_h, __TYPE__* x_h, __TYPE__* y_h, __TYPE__* beta_h, __TYPE__* gamma_h, int dimPoint, int dimVect, int nx, int ny) {
    return KernelGpuGradConvXA<__TYPE__,CauchyFp>(ooSigma2, e_h, alpha_h, x_h, y_h, beta_h, gamma_h, dimPoint, dimVect, nx, ny);
}
extern "C" int LaplaceGpuGradConvXA(__TYPE__ ooSigma2, __TYPE__* e_h, __TYPE__* alpha_h, __TYPE__* x_h, __TYPE__* y_h, __TYPE__* beta_h, __TYPE__* gamma_h, int dimPoint, int dimVect, int nx, int ny) {
    return KernelGpuGradConvXA<__TYPE__,LaplaceFp>(ooSigma2, e_h, alpha_h, x_h, y_h, beta_h, gamma_h, dimPoint, dimVect, nx, ny);
}
extern "C" int InverseMultiquadricGpuGradConvXA(__TYPE__ ooSigma2, __TYPE__* e_h, __TYPE__* alpha_h, __TYPE__* x_h, __TYPE__* y_h, __TYPE__* beta_h, __TYPE__* gamma_h, int dimPoint, int dimVect, int nx, int ny) {
    return KernelGpuGradConvXA<__TYPE__,InverseMultiquadricFp>(ooSigma2, e_h, alpha_h, x_h, y_h, beta_h, gamma_h, dimPoint, dimVect, nx, ny);
}

void ExitFcn(void) {
    cudaDeviceReset();
}
