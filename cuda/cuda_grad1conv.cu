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
    // Thread kernel:
    // Computation of gamma_i = \partial_{x_i} < alpha_i, sum_j k(x_i,y_j)*beta_j >
    // for index i given by thread id.
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // the following line does not work with nvcc 3.0 (it is a bug; it works with anterior and posterior versions)
    // extern __shared__ TYPE SharedData[];  // shared data will contain x and alpha data for the block
    // here is the bug fix (see http://forums.nvidia.com/index.php?showtopic=166905)
    extern __shared__ char SharedData_char[];
    TYPE* const SharedData = reinterpret_cast<TYPE*>(SharedData_char);
    // end of bug fix
    
    // One thread = One line = One x_i + One a_i + One gamma_i + a whole bunch of "y_j", "b_j".
    TYPE xi[DIMPOINT], alphai[DIMVECT], xmy[DIMPOINT], gammai[DIMPOINT];
    if(i<nx) { // we will compute gammai only if i is in the range
        for(int k=0; k<DIMPOINT; k++)
            xi[k]     =     x[i*DIMPOINT+k]; // load   x_i   from device global memory
        for(int k=0; k<DIMVECT; k++)
            alphai[k] = alpha[i*DIMVECT +k]; // load alpha_i from device global memory
        for(int k=0; k<DIMPOINT; k++)
            gammai[k] = 0.0f;                // Make sure to put to zero the output array 
    }

    // Here, we use a tiled matrix decomposition. See cuda_conv.cu for graphs and explanations.
    
    for(int jstart = 0, tile = 0; jstart < ny; jstart += blockDim.x, tile++) {

        // Load data in Shared memory -----------------------------------------------------------
        int j = tile * blockDim.x + threadIdx.x; // Current column
        // We load yj and betaj from device global memory...
        if(j<ny) { // ...only if j<ny (we may be in the last columns of the last tile...)
            // Pretty uneasy to read : we store yj and betaj interleaved, for better performance
            // SharedData = "[ y0, b0, y1, b1, y2, b2, ... ]"
            int inc = DIMPOINT + DIMVECT; // Size of a  [yj, bj] block
            for(int k=0; k<DIMPOINT; k++)
                SharedData[threadIdx.x*inc+k]          =    y[j*DIMPOINT+k];
            for(int k=0; k<DIMVECT; k++)
                SharedData[threadIdx.x*inc+DIMPOINT+k] = beta[j*DIMVECT+k];
        }
        __syncthreads();
        // At this point :
        // - x_i, alpha_i sit in the thread memory
        // - [y_N, ..., y_{N+blockDim.x}] and [b_N, ..., b_{N+blockDim.x}] sit
        //   in the SharedData, where [N : N+blockDim.x] is the tile span.
        // - the output line gamma_i is in the thread memory, and contains the result
        //   of the summation over the previous tiles.
      
        
        // Map-Reduction loop -------------------------------------------------------------------
        // We can now proceed to the "tiled" matrix product, where one line = one thread.
        if(i<nx) // we compute gammai only if i is in the range
        {
            TYPE *yj, *betaj;                  // As y_j and beta_j are interleaved...
            yj      = SharedData;              // We'll on some cute pointer arithmetics!
            betaj   = SharedData + DIMPOINT;
            int inc = DIMPOINT   + DIMVECT;    // The increment, size of a [y_j,b_j] block.
            
            for(int jrel = 0; jrel < blockDim.x && jrel<ny-jstart; jrel++, yj+=inc, betaj+=inc) {
                // Reduction loop over j : we're getting to the maths ***************************
                TYPE r2 = 0.0f, sga = 0.0f;
                // Compute x_i-y_j and its squared norm:
                for(int k=0; k<DIMPOINT; k++) {
                    xmy[k]  =  xi[k]-yj[k];
                    r2     += xmy[k]*xmy[k];
                }
                // Compute < alpha_i, beta_j > :
                for(int k=0; k<DIMVECT; k++)
                    sga += betaj[k]*alphai[k];
                // Now, we use the formula for
                // d/dx f(|x-y|^2) = 2 * (x-y) * f'(|x-y|^2)
                TYPE s =  2.0f * sga * GaussFp( r2 , ooSigma2 );
                for(int k=0; k<DIMPOINT; k++)
                    gammai[k] += s * xmy[k];
                // ******************************************************************************
            }
        }
        // Once the loop is over, the current tiled matrix product has been reduced to gamma_i
        __syncthreads();  // So make sure that no one's left behind...
        // And move on to the next tile.
    }

    // Save the result in global memory.
    if(i<nx)
        for(int k=0; k<DIMPOINT; k++)
            gamma[i*DIMPOINT+k] = gammai[k];
}

//////////////////////////////////////////////////////
/////////// CPU -> GPU -> CPU routines ///////////////
//////////////////////////////////////////////////////
//////////////////////////////////////////////////////

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
    cudaMalloc((void**)&x_d,     sizeof(float)*(nx*dimPoint));
    cudaMalloc((void**)&y_d,     sizeof(float)*(ny*dimPoint));
    cudaMalloc((void**)&alpha_d, sizeof(float)*(nx*dimVect ));
    cudaMalloc((void**)&beta_d,  sizeof(float)*(ny*dimVect ));
    cudaMalloc((void**)&gamma_d, sizeof(float)*(nx*dimPoint));

    // Send data from host to device.
    cudaMemcpy(x_d,     x_h,     sizeof(float)*(nx*dimPoint), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d,     y_h,     sizeof(float)*(ny*dimPoint), cudaMemcpyHostToDevice);
    cudaMemcpy(alpha_d, alpha_h, sizeof(float)*(nx*dimVect ), cudaMemcpyHostToDevice);
    cudaMemcpy(beta_d,  beta_h,  sizeof(float)*(ny*dimVect ), cudaMemcpyHostToDevice);

    // compute on device.
    dim3 blockSize;
    blockSize.x = CUDA_BLOCK_SIZE; // number of threads in each block
    dim3 gridSize;
    gridSize.x =  nx / blockSize.x + (nx%blockSize.x==0 ? 0 : 1);
    
    // Copy-paste templating, allowing us to pass the DIMPOINT and DIMVECT at compilation time : 
    if(     dimPoint==1 && dimVect==1)
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
