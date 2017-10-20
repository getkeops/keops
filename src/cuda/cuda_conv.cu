/*
   This file is part of the libds by b. Charlier and J. Glaunes
*/

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include "radial_kernels.cx"


#define UseCudaOnDoubles USE_DOUBLE_PRECISION

///////////////////////////////////////
/////////// CUDA KERNEL ///////////////
///////////////////////////////////////


// thread kernel: computation of gammai = sum_j k(xi,yj)betaj for index i given by thread id.
template < typename TYPE, int DIMPOINT, int DIMVECT >
__global__ void GaussGpuConvOnDevice(TYPE ooSigma2,
                                      TYPE *x, TYPE *y, TYPE *beta, TYPE *gamma,
                                      int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ TYPE SharedData[];  // shared data will contain x and alpha data for the block

    TYPE xi[DIMPOINT], gammai[DIMVECT];
    if(i<nx){  // we will compute gammai only if i is in the range
        // load xi from device global memory
        for(int k=0; k<DIMPOINT; k++)
            xi[k] = x[i*DIMPOINT+k];
        for(int k=0; k<DIMVECT; k++)
            gammai[k] = 0.0f;
    }

    for(int jstart = 0, tile = 0; jstart < ny; jstart += blockDim.x, tile++) {
        int j = tile * blockDim.x + threadIdx.x;
        if(j<ny){// we load yj and betaj from device global memory only if j<ny
            int inc = DIMPOINT + DIMVECT;
            for(int k=0; k<DIMPOINT; k++)
                SharedData[threadIdx.x*inc+k] = y[j*DIMPOINT+k];
            for(int k=0; k<DIMVECT; k++)
                SharedData[threadIdx.x*inc+DIMPOINT+k] = beta[j*DIMVECT+k];
        }
        __syncthreads();
        
        if(i<nx){ // we compute gammai only if needed
            TYPE *yj, *betaj;
            yj = SharedData;
            betaj = SharedData + DIMPOINT;
            int inc = DIMPOINT + DIMVECT;
            for(int jrel = 0; jrel < blockDim.x && jrel<ny-jstart; jrel++, yj+=inc, betaj+=inc) {
                TYPE s = KernelGauss(xi,yj,ooSigma2,DIMPOINT);
                for(int k=0; k<DIMVECT; k++)
                    gammai[k] += s * betaj[k];
            }
        }
        __syncthreads();
    }

    // Save the result in global memory.
    if(i<nx)
        for(int k=0; k<DIMVECT; k++)
            gamma[i*DIMVECT+k] = gammai[k];
}

///////////////////////////////////////////////////
#if !(UseCudaOnDoubles) 
extern "C" int GaussGpuEvalConv(float ooSigma2,
                                   float* x_h, float* y_h, float* beta_h, float* gamma_h,
                                   int dimPoint, int dimVect, int nx, int ny) {

    //printf("%f %f %f", x_h[0] , x_h[1] ,x_h[2]);

    // Data on the device.
    float* x_d;
    float* y_d;
    float* beta_d;
    float* gamma_d;

    // Allocate arrays on device.
    cudaMalloc((void**)&x_d, sizeof(float)*(nx*dimPoint));
    cudaMalloc((void**)&y_d, sizeof(float)*(ny*dimPoint));
    cudaMalloc((void**)&beta_d, sizeof(float)*(ny*dimVect));
    cudaMalloc((void**)&gamma_d, sizeof(float)*(nx*dimVect));

    // Set values to zeros
    cudaMemset(x_d,0, sizeof(float)*(nx*dimPoint));
    cudaMemset(y_d,0, sizeof(float)*(ny*dimPoint));
    cudaMemset(beta_d,0, sizeof(float)*(ny*dimVect));
    cudaMemset(gamma_d,0, sizeof(float)*(nx*dimVect));

    // Send data from host to device.
    cudaMemcpy(x_d, x_h, sizeof(float)*(nx*dimPoint), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y_h, sizeof(float)*(ny*dimPoint), cudaMemcpyHostToDevice);
    cudaMemcpy(beta_d, beta_h, sizeof(float)*(ny*dimVect), cudaMemcpyHostToDevice);

    // Compute on device.
    dim3 blockSize;
    blockSize.x = CUDA_BLOCK_SIZE; // number of threads in each block
    dim3 gridSize;
    gridSize.x =  nx / blockSize.x + (nx%blockSize.x==0 ? 0 : 1);

    if(dimPoint==1 && dimVect==1)
        GaussGpuConvOnDevice<float,1,1><<<gridSize,blockSize,blockSize.x*(dimVect+dimPoint)*sizeof(float)>>>
        (ooSigma2, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==3 && dimVect==1)
        GaussGpuConvOnDevice<float,3,1><<<gridSize,blockSize,blockSize.x*(dimVect+dimPoint)*sizeof(float)>>>
        (ooSigma2, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==2 && dimVect==1)
        GaussGpuConvOnDevice<float,2,1><<<gridSize,blockSize,blockSize.x*(dimVect+dimPoint)*sizeof(float)>>>
        (ooSigma2, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==2 && dimVect==2)
        GaussGpuConvOnDevice<float,2,2><<<gridSize,blockSize,blockSize.x*(dimVect+dimPoint)*sizeof(float)>>>
        (ooSigma2, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==3 && dimVect==3)
        GaussGpuConvOnDevice<float,3,3><<<gridSize,blockSize,blockSize.x*(dimVect+dimPoint)*sizeof(float)>>>
        (ooSigma2, x_d, y_d, beta_d, gamma_d, nx, ny);
    else {
        printf("error: dimensions of Gauss kernel not implemented in cuda");
		cudaFree(x_d);
		cudaFree(y_d);
		cudaFree(beta_d);
		cudaFree(gamma_d);
        return(-1);
    }

    // block until the device has completed
    cudaThreadSynchronize();

    // Send data from device to host.
    cudaMemcpy(gamma_h, gamma_d, sizeof(float)*(nx*dimVect),cudaMemcpyDeviceToHost);

    // Free memory.
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(beta_d);
    cudaFree(gamma_d);

    return 0;
}

///////////////////////////////////////////////////

#else
extern "C" int GaussGpuEvalConv(double ooSigma2,
                                   double* x_h, double* y_h, double* beta_h, double* gamma_h,
                                   int dimPoint, int dimVect, int nx, int ny) {

    // Data on the device.
    double* x_d;
    double* y_d;
    double* beta_d;
    double* gamma_d;

    // Allocate arrays on device.
    cudaMalloc((void**)&x_d, sizeof(double)*(nx*dimPoint));
    cudaMalloc((void**)&y_d, sizeof(double)*(ny*dimPoint));
    cudaMalloc((void**)&beta_d, sizeof(double)*(ny*dimVect));
    cudaMalloc((void**)&gamma_d, sizeof(double)*(nx*dimVect));

    // Send data from host to device.
    cudaMemcpy(x_d, x_h, sizeof(double)*(nx*dimPoint), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y_h, sizeof(double)*(ny*dimPoint), cudaMemcpyHostToDevice);
    cudaMemcpy(beta_d, beta_h, sizeof(double)*(ny*dimVect), cudaMemcpyHostToDevice);

    // Compute on device.
    dim3 blockSize;
    blockSize.x = CUDA_BLOCK_SIZE; // number of threads in each block
    dim3 gridSize;
    gridSize.x =  nx / blockSize.x + (nx%blockSize.x==0 ? 0 : 1);
//test if ggridSIze \leq  65535
    if(dimPoint==1 && dimVect==1)
        GaussGpuConvOnDevice<double,1,1><<<gridSize,blockSize,blockSize.x*(dimVect+dimPoint)*sizeof(double)>>>
        (ooSigma2, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==3 && dimVect==1)
        GaussGpuConvOnDevice<double,3,1><<<gridSize,blockSize,blockSize.x*(dimVect+dimPoint)*sizeof(double)>>>
        (ooSigma2, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==2 && dimVect==1)
        GaussGpuConvOnDevice<double,2,1><<<gridSize,blockSize,blockSize.x*(dimVect+dimPoint)*sizeof(double)>>>
        (ooSigma2, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==2 && dimVect==2)
        GaussGpuConvOnDevice<double,2,2><<<gridSize,blockSize,blockSize.x*(dimVect+dimPoint)*sizeof(double)>>>
        (ooSigma2, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==3 && dimVect==3)
        GaussGpuConvOnDevice<double,3,3><<<gridSize,blockSize,blockSize.x*(dimVect+dimPoint)*sizeof(double)>>>
        (ooSigma2, x_d, y_d, beta_d, gamma_d, nx, ny);
    else {
        printf("error: dimensions of Gauss kernel not implemented in cuda");
		cudaFree(x_d);
		cudaFree(y_d);
		cudaFree(beta_d);
		cudaFree(gamma_d);
        return(-1);
    }

    // block until the device has completed
    cudaThreadSynchronize();

    // Send data from device to host.
    cudaMemcpy(gamma_h, gamma_d, sizeof(double)*(nx*dimVect),cudaMemcpyDeviceToHost);


    // Free memory.
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(beta_d);
    cudaFree(gamma_d);

    return 0;
}
#endif

void ExitFcn(void) {
  cudaDeviceReset();
}

