/*
*	This cuda routine allows one to compute the derivative wrt the point cloud 'y' of the derivative
*	wrt 'x' of the expression
*		K(x_i,y_j) @ b_j =  sum_j f( |x_i-y_j|^2 ) b_j
*
*
*	We're looking for the gradient with respect to y of
*
*	< e, K(s,a,x,y,b) >  =  \sum_{i,j} f_s'( |x_i-y_j|^2 ) * < a_i, b_j > * 2 < e_i, x_i-y_j>,
*
*	which is an M-by-D array g_j (j from 1 to M), where each line is equal to
*
*	g_j = -2* \sum_i < a_i, b_j > * [                       f_s'(  |x_i-y_j|^2 ) * e_i
*	                                + 2* < x_i-y_j, e_i > * f_s''( |x_i-y_j|^2 ) * (x_i-y_j) ]
*
*	We will compute this sum over the index 'i' on the GPU, with 'one thread' = 'one index j'.
*	Data will be stored as follow:
*	  - e_i in the SharedData
* 	  - a_i in the SharedData
*	  - x_i in the SharedData
*	  - y_j in the thread memory
*	  - b_j in the thread memory
*
*
* Author : Jean Feydy, heavily based on the work of Joan Glaunès and Benjamin Charlier.
*
*/

#include <stdio.h>
#include <assert.h>
#include <cuda.h>

#include "specific/radial_kernels/radial_kernels.h"
#include "specific/radial_kernels/cuda_gradconv_xy.cx"


//////////////////////////////////////////////////////
/////////// CPU -> GPU -> CPU routines ///////////////
//////////////////////////////////////////////////////


template < typename TYPE, KernelFun KernelFp , KernelFun KernelFpp >
int KernelGpuGradConvXY(TYPE ooSigma2,               // 1 / sigma^2
                        TYPE* e_h,                     // N-by-D array (same as x)
                        TYPE* alpha_h, TYPE* x_h,     // N-by-E, N-by-D arrays
                        TYPE* y_h,     TYPE* beta_h,  // M-by-D, M-by-E arrays
                        TYPE* gamma_h,                 // Output: M-by-D (same as y)
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
    cudaMalloc((void**)&gamma_d, sizeof(TYPE)*(ny*dimPoint)); // Output: M-by-D (same as y)

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
    gridSize.x =  ny / blockSize.x + (ny%blockSize.x==0 ? 0 : 1); // NB: here, we're working columnwise !

    // Copy-paste templating, allowing us to pass the DIMPOINT and DIMVECT at compilation time :
    // NB: Here, we use more SharedData than in the rowwise code !
    //     One block of SharedData = [ei,ai,xi], of size (dimPoint+dimVect+dimPoint)*sizeof(TYPE)
    if(     dimPoint==1 && dimVect==1)
        KernelGpuGradConvXYOnDevice<TYPE,1,1,KernelFp,KernelFpp><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect+dimPoint)*sizeof(TYPE)>>>
        (ooSigma2, e_d, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==2 && dimVect==1)
        KernelGpuGradConvXYOnDevice<TYPE,2,1,KernelFp,KernelFpp><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect+dimPoint)*sizeof(TYPE)>>>
        (ooSigma2, e_d, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==3 && dimVect==1)
        KernelGpuGradConvXYOnDevice<TYPE,3,1,KernelFp,KernelFpp><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect+dimPoint)*sizeof(TYPE)>>>
        (ooSigma2, e_d, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==4 && dimVect==1)
        KernelGpuGradConvXYOnDevice<TYPE,4,1,KernelFp,KernelFpp><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect+dimPoint)*sizeof(TYPE)>>>
        (ooSigma2, e_d, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==2 && dimVect==2)
        KernelGpuGradConvXYOnDevice<TYPE,2,2,KernelFp,KernelFpp><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect+dimPoint)*sizeof(TYPE)>>>
        (ooSigma2, e_d, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==3 && dimVect==3)
        KernelGpuGradConvXYOnDevice<TYPE,3,3,KernelFp,KernelFpp><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect+dimPoint)*sizeof(TYPE)>>>
        (ooSigma2, e_d, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==4 && dimVect==4)
        KernelGpuGradConvXYOnDevice<TYPE,4,4,KernelFp,KernelFpp><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect+dimPoint)*sizeof(TYPE)>>>
        (ooSigma2, e_d, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else {
        printf("GaussGpuGradConvXY error: dimensions of Gauss kernel not implemented in cuda\nYou probably just need a copy-paste in the conda_gradconv_xy.cu file !");
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
    cudaMemcpy(gamma_h, gamma_d, sizeof(TYPE)*(ny*dimPoint),cudaMemcpyDeviceToHost); // Output: M-by-D (same as y)

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
extern "C" int GaussGpuGradConvXY(__TYPE__ ooSigma2, __TYPE__* e_h, __TYPE__* alpha_h, __TYPE__* x_h, __TYPE__* y_h, __TYPE__* beta_h, __TYPE__* gamma_h, int dimPoint, int dimVect, int nx, int ny) {
    return KernelGpuGradConvXY<__TYPE__,GaussFp,GaussFpp>(ooSigma2, e_h, alpha_h, x_h, y_h, beta_h, gamma_h, dimPoint, dimVect, nx, ny);
}
extern "C" int CauchyGpuGradConvXY(__TYPE__ ooSigma2, __TYPE__* e_h, __TYPE__* alpha_h, __TYPE__* x_h, __TYPE__* y_h, __TYPE__* beta_h, __TYPE__* gamma_h, int dimPoint, int dimVect, int nx, int ny) {
    return KernelGpuGradConvXY<__TYPE__,CauchyFp,CauchyFpp>(ooSigma2, e_h, alpha_h, x_h, y_h, beta_h, gamma_h, dimPoint, dimVect, nx, ny);
}
extern "C" int LaplaceGpuGradConvXY(__TYPE__ ooSigma2, __TYPE__* e_h, __TYPE__* alpha_h, __TYPE__* x_h, __TYPE__* y_h, __TYPE__* beta_h, __TYPE__* gamma_h, int dimPoint, int dimVect, int nx, int ny) {
    return KernelGpuGradConvXY<__TYPE__,LaplaceFp,LaplaceFpp>(ooSigma2, e_h, alpha_h, x_h, y_h, beta_h, gamma_h, dimPoint, dimVect, nx, ny);
}
extern "C" int InverseMultiquadricGpuGradConvXY(__TYPE__ ooSigma2, __TYPE__* e_h, __TYPE__* alpha_h, __TYPE__* x_h, __TYPE__* y_h, __TYPE__* beta_h, __TYPE__* gamma_h, int dimPoint, int dimVect, int nx, int ny) {
    return KernelGpuGradConvXY<__TYPE__,InverseMultiquadricFp,InverseMultiquadricFpp>(ooSigma2, e_h, alpha_h, x_h, y_h, beta_h, gamma_h, dimPoint, dimVect, nx, ny);
}

void ExitFcn(void) {
    cudaDeviceReset();
}
