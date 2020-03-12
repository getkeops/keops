#include <iostream>
#include <math.h>

#include <stdio.h>
#include <assert.h>
#include <cuda.h>

#include <chrono>
typedef std::chrono::high_resolution_clock Clock;


#define __TYPE__ float


///////////////////////
//  Gaussian Kernel  //
///////////////////////

template < typename TYPE, int DIMPOINT, int DIMVECT >
__global__ void KernelGpuConvOnDevice(TYPE *x, TYPE *y, TYPE *beta, TYPE *gamma,
                                      int nx, int ny) {
    // Thread kernel:
    // Computation of gamma_i = sum_j k(x_i,y_j)*beta_j
    // for index i given by thread id.

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ TYPE SharedData[];  // shared data will contain x and alpha data for the block

    // One thread = One line = One x_i + One gamma_i + a whole bunch of "y_j".
    TYPE xi[DIMPOINT], gammai[DIMVECT];
    if(i<nx) { // we will compute gammai only if i is in the range
#pragma unroll
        for(int k=0; k<DIMPOINT; k++)
            xi[k] = x[i*DIMPOINT+k];  // Load xi from device global memory
#pragma unroll
        for(int k=0; k<DIMVECT; k++)
            gammai[k] = 0.0f;         // Make sure to put to zero the output array
    }

    for(int jstart = 0, tile = 0; jstart < ny; jstart += blockDim.x, tile++) {

        // Load data in Shared memory -----------------------------------------------------------
        int j = tile * blockDim.x + threadIdx.x; // Current column
        // We load yj and betaj from device global memory...
        if(j<ny) { // ...only if j<ny (we may be in the last columns of the last tile...)
            // Pretty uneasy to read : we store yj and betaj interleaved, for better performance
            // SharedData = "[ y0, b0, y1, b1, y2, b2, ... ]"
            int inc = DIMPOINT + DIMVECT; // Size of a  [yj, bj] block
#pragma unroll
            for(int k=0; k<DIMPOINT; k++)
                SharedData[threadIdx.x*inc+k]          =    y[j*DIMPOINT+k];
#pragma unroll
            for(int k=0; k<DIMVECT; k++)
                SharedData[threadIdx.x*inc+DIMPOINT+k] = beta[j*DIMVECT +k];
        }
        __syncthreads();

        // Map-Reduction loop -------------------------------------------------------------------
        // We can now proceed to the "tiled" matrix product, where one line = one thread.
        if(i<nx) { // we compute gammai only if needed
            TYPE *yj, *betaj;              // As y_j and beta_j are interleaved...
            yj    = SharedData;            // We'll on some cute pointer arithmetics!
            betaj = SharedData + DIMPOINT;
            int inc = DIMPOINT + DIMVECT;  // The increment, size of a [y_j,b_j] block.
            for(int jrel = 0; jrel < blockDim.x && jrel<ny-jstart; jrel++, yj+=inc, betaj+=inc) {
                TYPE r2 = 0.0f;
                // Compute x_i-y_j and its squared norm:
#pragma unroll
                for(int k=0; k<DIMPOINT; k++) {
                    TYPE temp = xi[k] - yj[k];
                    r2 += temp * temp;
                }
                // Straighforward inplace reduction loop over j : at last, we're getting to the maths... **********
                TYPE s = expf(-r2);  // The kernel function is stored in "radial_kernels.cx"
#pragma unroll
                for(int k=0; k<DIMVECT; k++)   // Add the vector s*beta_j to gamma_i
                    gammai[k] += s * betaj[k]; // (no need to be extra-clever here)
                // ************************************************************************************************
            }
        }

        // Once the loop is over, the current tiled matrix product has been reduced to gamma_i
        __syncthreads(); // So make sure that no one's left behind...
        // And move on to the next tile.
    }

    // Save the result in global memory.
    if(i<nx)
#pragma unroll
        for(int k=0; k<DIMVECT; k++)
            gamma[i*DIMVECT+k] = gammai[k];
}



template < typename TYPE, int DIMPOINT, int DIMVECT >
int KernelGpuEvalConv(TYPE* x_h, TYPE* y_h, TYPE* beta_h, TYPE* gamma_h,
                      int nx, int ny, int nits) {

    // Data on the device.
    TYPE* x_d;
    TYPE* y_d;
    TYPE* beta_d;
    TYPE* gamma_d;

    // Allocate arrays on device.
    cudaMalloc((void**)&x_d,     sizeof(TYPE)*(nx*DIMPOINT + ny*DIMPOINT + ny*DIMVECT  + nx*DIMVECT ));
    y_d = x_d + nx * DIMPOINT;
    beta_d = y_d + ny * DIMPOINT;
    gamma_d = beta_d + ny * DIMVECT;

    // Send data from host to device.
    cudaMemcpy(x_d,    x_h,    sizeof(TYPE)*(nx*DIMPOINT + ny*DIMPOINT + ny*DIMVECT ), cudaMemcpyHostToDevice);

    int n_blocksizes = 1;
    int BlockSizes[n_blocksizes] = {192}; //{64, 128, 192, 256, 512, 1024};
    int CUDA_BLOCK_SIZE = 0;

    for(int b = 0; b < n_blocksizes; b++) {
        CUDA_BLOCK_SIZE = BlockSizes[b];
        std::cout << "BlockSize = " << CUDA_BLOCK_SIZE << ", " ;

        // Compute on device.
        dim3 blockSize;
        blockSize.x = CUDA_BLOCK_SIZE; // number of threads in each block
        dim3 gridSize;
        gridSize.x =  nx / blockSize.x + (nx%blockSize.x==0 ? 0 : 1);

        
        auto start = Clock::now();

        for (int it=0; it < nits; it++) {
            KernelGpuConvOnDevice<TYPE,3 , 1><<<gridSize,blockSize,blockSize.x*(DIMVECT+DIMPOINT)*sizeof(TYPE)>>>
            (x_d, y_d, beta_d, gamma_d, nx, ny);

        }

        // block until the device has completed
        cudaDeviceSynchronize();

        auto end= Clock::now();
        std::cout << "time = " 
                  << nits << "x "
                  << (float) std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / (float) (1000000 * nits)
                  << " milliseconds" << std::endl;

    }
    std::cout << std::endl;

    // Send data from device to host.
    cudaMemcpy(gamma_h, gamma_d, sizeof(TYPE)*(nx*DIMVECT),cudaMemcpyDeviceToHost);

    // Free memory.
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(beta_d);
    cudaFree(gamma_d);

    return 0;
}




int main(void)
{
  //int Ns[3] = {10000, 100000, 1000000};
  //int nits[3] = {100, 10, 1};

  int Ns[1] = {1000000};
  int nits[1] = {1};

  int N = 0;
  int nit = 0;

  for (int n = 0; n < 1; n++){

      N = Ns[n];
      nit = nits[n];

      float *x, *y, *p, *g;

      // Allocate Unified Memory – accessible from CPU or GPU
      cudaMallocManaged(&x, (3*N + 3*N + N + N) * sizeof(float));
      y = x + 3*N;
      p = y + 3*N;
      g = p + N;

      // initialize x and y arrays on the host
      for (int i = 0; i < 3*N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
      }
      for (int i = 0; i < N; i++) {
        p[i] = 3.0f;
      }

      std::cout << "N = " << N << " : " << std::endl;

      KernelGpuEvalConv< __TYPE__, 3, 1 >(
            x, y, p, g,
            N, N,
            nit
      );

      // Wait for GPU to finish before accessing on host
      cudaDeviceSynchronize();


      // Free memory
      cudaFree(x);
    }

      std::cout << "Done" << std::endl;
  
  return 0;
}

