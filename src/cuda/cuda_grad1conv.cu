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


template < typename TYPE, int DIMPOINT, int DIMVECT >
__global__ void GaussGpuGrad1ConvOnDevice(TYPE ooSigma2,
        TYPE *alpha, TYPE *x, TYPE *y, TYPE *beta, TYPE *gamma,
        int nx, int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

     extern __shared__ TYPE SharedData[];  // shared data will contain x and alpha data for the block

    TYPE xi[DIMPOINT], alphai[DIMVECT], xmy[DIMPOINT], gammai[DIMPOINT];
    if(i<nx)  // we will compute gammai only if i is in the range
    {
        // load xi and alphai from device global memory
        for(int k=0; k<DIMPOINT; k++)
            xi[k] = x[i*DIMPOINT+k];
        for(int k=0; k<DIMVECT; k++)
            alphai[k] = alpha[i*DIMVECT+k];
        for(int k=0; k<DIMPOINT; k++)
            gammai[k] = 0.0f;
    }

    for(int jstart = 0, tile = 0; jstart < ny; jstart += blockDim.x, tile++)
    {
        int j = tile * blockDim.x + threadIdx.x;
        if(j<ny) // we load yj and betaj from device global memory only if j<ny
        {
            int inc = DIMPOINT + DIMVECT;
            for(int k=0; k<DIMPOINT; k++)
                SharedData[threadIdx.x*inc+k] = y[j*DIMPOINT+k];
            for(int k=0; k<DIMVECT; k++)
                SharedData[threadIdx.x*inc+DIMPOINT+k] = beta[j*DIMVECT+k];
        }
        __syncthreads();
        if(i<nx) // we compute gammai only if i is in the range
        {
            TYPE *yj, *betaj;
            yj = SharedData;
            betaj = SharedData + DIMPOINT;
            int inc = DIMPOINT + DIMVECT;
            for(int jrel = 0; jrel < blockDim.x && jrel<ny-jstart; jrel++, yj+=inc, betaj+=inc)
            {
                TYPE r2 = 0.0f, sga = 0.0f;
                for(int k=0; k<DIMPOINT; k++)
                {
                    xmy[k] =  xi[k]-yj[k];
                    r2 += xmy[k]*xmy[k];
                }
                for(int k=0; k<DIMVECT; k++)
                    sga += betaj[k]*alphai[k];
                TYPE s =  (-ooSigma2*2.0f*sga) * exp(-r2*ooSigma2);
                for(int k=0; k<DIMPOINT; k++)
                    gammai[k] += s * xmy[k];
            }
        }
        __syncthreads();
    }

    // Save the result in global memory.
    if(i<nx)
        for(int k=0; k<DIMPOINT; k++)
            gamma[i*DIMPOINT+k] = gammai[k];
}

//////////////////////////////////////////////////////////////

#if !(UseCudaOnDoubles) 
extern "C" int GaussGpuGrad1Conv(float ooSigma2,
                                float* alpha_h, float* x_h, float* y_h, float* beta_h, float* gamma_h,
                                int dimPoint, int dimVect, int nx, int ny){

    // Data on the device.
    float* x_d;
    float* y_d;
    float* alpha_d;
    float* gamma_d;
    float* beta_d;

    // Allocate arrays on device.
    cudaMalloc((void**)&x_d, sizeof(float)*(nx*dimPoint));
    cudaMalloc((void**)&y_d, sizeof(float)*(ny*dimPoint));
    cudaMalloc((void**)&alpha_d, sizeof(float)*(nx*dimVect));
    cudaMalloc((void**)&beta_d, sizeof(float)*(ny*dimVect));
    cudaMalloc((void**)&gamma_d, sizeof(float)*(nx*dimPoint));

    // Send data from host to device.
    cudaMemcpy(x_d, x_h, sizeof(float)*(nx*dimPoint), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y_h, sizeof(float)*(ny*dimPoint), cudaMemcpyHostToDevice);
    cudaMemcpy(alpha_d, alpha_h, sizeof(float)*(nx*dimVect), cudaMemcpyHostToDevice);
    cudaMemcpy(beta_d, beta_h, sizeof(float)*(ny*dimVect), cudaMemcpyHostToDevice);

    // compute on device.
    dim3 blockSize;
    blockSize.x = CUDA_BLOCK_SIZE; // number of threads in each block
    dim3 gridSize;
    gridSize.x =  nx / blockSize.x + (nx%blockSize.x==0 ? 0 : 1);

    if(dimPoint==1 && dimVect==1)
        GaussGpuGrad1ConvOnDevice<float,1,1><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect)*sizeof(float)>>>
            (ooSigma2, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==2 && dimVect==1)
        GaussGpuGrad1ConvOnDevice<float,2,1><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect)*sizeof(float)>>>
            (ooSigma2, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==3 && dimVect==1)
        GaussGpuGrad1ConvOnDevice<float,3,1><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect)*sizeof(float)>>>
            (ooSigma2, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==4 && dimVect==1)
        GaussGpuGrad1ConvOnDevice<float,4,1><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect)*sizeof(float)>>>
            (ooSigma2, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==2 && dimVect==2)
        GaussGpuGrad1ConvOnDevice<float,2,2><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect)*sizeof(float)>>>
            (ooSigma2, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==3 && dimVect==3)
        GaussGpuGrad1ConvOnDevice<float,3,3><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect)*sizeof(float)>>>
            (ooSigma2, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==4 && dimVect==4)
        GaussGpuGrad1ConvOnDevice<float,4,4><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect)*sizeof(float)>>>
            (ooSigma2, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else
    {
        printf("GaussGpuGrad1Conv error: dimensions of Gauss kernel not implemented in cuda\n");
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
    cudaMemcpy(gamma_h, gamma_d, sizeof(float)*(nx*dimPoint),cudaMemcpyDeviceToHost);

    // Free memory.
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(alpha_d);
    cudaFree(gamma_d);
    cudaFree(beta_d);

    return 0;
}

#else
//////////////////////////////////////////////////////////////
extern "C" int GaussGpuGrad1Conv(double ooSigma2,
                                double* alpha_h, double* x_h, double* y_h, double* beta_h, double* gamma_h,
                                int dimPoint, int dimVect, int nx, int ny){

    // Data on the device.
    double* x_d;
    double* y_d;
    double* alpha_d;
    double* gamma_d;
    double* beta_d;

    // Allocate arrays on device.
    cudaMalloc((void**)&x_d, sizeof(double)*(nx*dimPoint));
    cudaMalloc((void**)&y_d, sizeof(double)*(ny*dimPoint));
    cudaMalloc((void**)&alpha_d, sizeof(double)*(nx*dimVect));
    cudaMalloc((void**)&beta_d, sizeof(double)*(ny*dimVect));
    cudaMalloc((void**)&gamma_d, sizeof(double)*(nx*dimPoint));

    // Send data from host to device.
    cudaMemcpy(x_d, x_h, sizeof(double)*(nx*dimPoint), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y_h, sizeof(double)*(ny*dimPoint), cudaMemcpyHostToDevice);
    cudaMemcpy(alpha_d, alpha_h, sizeof(double)*(nx*dimVect), cudaMemcpyHostToDevice);
    cudaMemcpy(beta_d, beta_h, sizeof(double)*(ny*dimVect), cudaMemcpyHostToDevice);

    // compute on device.
    dim3 blockSize;
    blockSize.x = CUDA_BLOCK_SIZE; // number of threads in each block
    dim3 gridSize;
    gridSize.x =  nx / blockSize.x + (nx%blockSize.x==0 ? 0 : 1);

    if(dimPoint==1 && dimVect==1)
        GaussGpuGrad1ConvOnDevice<double,1,1><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect)*sizeof(double)>>>
            (ooSigma2, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==2 && dimVect==1)
        GaussGpuGrad1ConvOnDevice<double,2,1><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect)*sizeof(double)>>>
            (ooSigma2, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==3 && dimVect==1)
        GaussGpuGrad1ConvOnDevice<double,3,1><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect)*sizeof(double)>>>
            (ooSigma2, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==4 && dimVect==1)
        GaussGpuGrad1ConvOnDevice<double,4,1><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect)*sizeof(double)>>>
            (ooSigma2, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==2 && dimVect==2)
        GaussGpuGrad1ConvOnDevice<double,2,2><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect)*sizeof(double)>>>
            (ooSigma2, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==3 && dimVect==3)
        GaussGpuGrad1ConvOnDevice<double,3,3><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect)*sizeof(double)>>>
            (ooSigma2, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==4 && dimVect==4)
        GaussGpuGrad1ConvOnDevice<double,4,4><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect)*sizeof(double)>>>
            (ooSigma2, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else
    {
        printf("GaussGpuGrad1Conv error: dimensions of Gauss kernel not implemented in cuda\n");
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
    cudaMemcpy(gamma_h, gamma_d, sizeof(double)*(nx*dimPoint),cudaMemcpyDeviceToHost);

    // Free memory.
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(alpha_d);
    cudaFree(gamma_d);
    cudaFree(beta_d);

    return 0;
}

#endif

void ExitFcn(void)
{
    cudaDeviceReset();
}
