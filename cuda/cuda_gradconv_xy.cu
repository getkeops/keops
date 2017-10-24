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
#include "kernels_old.cx"


#define UseCudaOnDoubles USE_DOUBLE_PRECISION

///////////////////////////////////////
/////////// CUDA KERNEL ///////////////
///////////////////////////////////////


template < typename TYPE, int DIMPOINT, int DIMVECT > // Typically, float32, D, E
__global__ void GaussGpuGradConvXYOnDevice(TYPE ooSigma2, // 1/sigma^2
		TYPE *e,                                   // N-by-D array
		TYPE *alpha, TYPE *x, TYPE *y, TYPE *beta, // N-by-E, N-by-D, M-by-D, M-by-E arrays
		TYPE *gamma,                               // Output variable, M-by-D (same as y)
		int nx, int ny)
{
    // Thread kernel:
    // Computation of gamma_j = \partial_{y_j} < e_i, \partial_{x_i} < alpha_i, sum_j k(x_i,y_j)*beta_j > >
    //
    //                        = -2* \sum_i < a_i, b_j > * [                       f_s'(  |x_i-y_j|^2 ) * e_i
    //                                                    + 2* < x_i-y_j, e_i > * f_s''( |x_i-y_j|^2 ) * (x_i-y_j) ]
    // for index j given by thread id.
    
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // the following line does not work with nvcc 3.0 (it is a bug; it works with anterior and posterior versions)
    // extern __shared__ TYPE SharedData[];  // shared data will contain x and alpha data for the block
    // here is the bug fix (see http://forums.nvidia.com/index.php?showtopic=166905)
    extern __shared__ char SharedData_char[];
    TYPE* const SharedData = reinterpret_cast<TYPE*>(SharedData_char);
    // end of bug fix
    
    // One thread = One line = One y_j + One b_j + One gamma_j + a whole bunch of "e_i", "a_i", "x_i".
    TYPE yj[DIMPOINT], betaj[DIMVECT], xmy[DIMPOINT], gammaj[DIMPOINT];
    if(j<ny) { // we will compute gammaj only if j is in the range
        for(int k=0; k<DIMPOINT; k++)
            yj[k]     =     y[j*DIMPOINT+k]; // load y_j from device global memory
        for(int k=0; k<DIMVECT; k++)
            betaj[k]  =  beta[j*DIMVECT+k];  // load b_j from device global memory
        for(int k=0; k<DIMPOINT; k++)    // output : M-by-D : DIMPOINT
            gammaj[k] = 0.0f;            // Make sure to put to zero the output array 
    }

    // Here, we use a tiled matrix decomposition. See cuda_conv.cu for graphs and explanations.
    // Note that here, each thread reads from top to bottom (i++), instead of left to right (j++):
    for(int istart = 0, tile = 0; istart < nx; istart += blockDim.x, tile++) {

        // Load data in Shared memory -----------------------------------------------------------
        int i = tile * blockDim.x + threadIdx.x; // Current line
        // We load ei, alphai and xi from device global memory...
        if(i<nx) { // ...only if i<nx (we may be in the last columns of the last tile...)
            // Pretty uneasy to read : we store ei, ai and xi interleaved, for better performance
            // SharedData = "[ e0, a0, x0, e1, a1, x1, e2, a2, x2 ... ]"
            int inc = DIMPOINT + DIMVECT + DIMPOINT; // Size of a  [ei, ai, xi] block
            for(int k=0; k<DIMPOINT; k++)
                SharedData[threadIdx.x*inc+k]                  =     e[i*DIMPOINT+k];
            for(int k=0; k<DIMVECT; k++)
                SharedData[threadIdx.x*inc+DIMPOINT+k]         = alpha[i*DIMVECT +k];
            for(int k=0; k<DIMPOINT; k++)
                SharedData[threadIdx.x*inc+DIMPOINT+DIMVECT+k] =     x[i*DIMPOINT+k];
        }
        __syncthreads();
        // At this point :
        // - y_j, b_j sit in the thread memory
        // - [e_I, ..., e_{I+blockDim.x}], [a_I, ..., a_{I+blockDim.x}] and [x_I, ..., x_{I+blockDim.x}] sit
        //   in the SharedData, where [I : I+blockDim.x] is the tile span.
        // - the output line gamma_j is in the thread memory, and contains the result
        //   of the summation over the previous tiles.
      
        
        // Map-Reduction loop -------------------------------------------------------------------
        // We can now proceed to the "tiled" matrix product, where one line = one thread.
        if(j<ny) // we compute gammaj only if j is in the range
        {
            TYPE *ei, *alphai, *xi;           // As ei, ai and xi are interleaved...
            ei     = SharedData;              // We'll on some cute pointer arithmetics!
            alphai = SharedData + DIMPOINT;
            xi     = SharedData + DIMPOINT + DIMVECT;
            int inc = DIMPOINT  + DIMVECT + DIMPOINT; // The increment, size of a [ei, ai, xi] block.
            
            for(int irel = 0; irel < blockDim.x && irel<nx-istart; irel++, ei+=inc, alphai+=inc, xi+=inc) {
                // Reduction loop over i : we're getting to the maths ***************************
                // Remember: we're computing 
                //    g_j  = -2* \sum_i < a_i, b_j > * [                       f_s'(  |x_i-y_j|^2 ) * e_i
                //                                     + 2* < x_i-y_j, e_i > * f_s''( |x_i-y_j|^2 ) * (x_i-y_j) ]

                TYPE r2 = 0.0f, ei_s_xmy = 0.0f, ai_s_bj;
                // Compute x_i-y_j and its squared norm:
                for(int k=0; k<DIMPOINT; k++) {
                    xmy[k]  =  xi[k]-yj[k];
                    r2     += xmy[k]*xmy[k];
                }
                // Compute < e_i, x_i-y_j> :
                for(int k=0; k<DIMPOINT; k++) // Scalar product between POINTS.
                    ei_s_xmy += ei[k]*xmy[k];
                // Compute < a_i, b_j> :
                for(int k=0; k<DIMVECT; k++)  // Scalar product between VECTORS.
                    ai_s_bj  += alphai[k]* betaj[k];
                // Scalar factor for the first line,   "-2* <a_i,b_j> * f_s'( |x_i-y_j|^2 )"
                TYPE s1 =  -2.0f * ai_s_bj *            GaussFp(  r2 , ooSigma2 );
                // Scalar factor for the second line,  "-4* <a_i,b_j> * < e_i, x_i-y_j > * f_s''( |x_i-y_j|^2 )"
                TYPE s2 =  -4.0f * ai_s_bj * ei_s_xmy * GaussFpp( r2 , ooSigma2 );
                
                for(int k=0; k<DIMPOINT; k++)    // Output: M-by-D
                    gammaj[k] += s1 * ei[k] + s2 * xmy[k];  // Final increment
                // ******************************************************************************
            }
        }
        // Once the loop is over, the current tiled matrix product has been reduced to gamma_j
        __syncthreads();  // So make sure that no one's left behind...
        // And move on to the next tile.
    }

    // Save the result in global memory.
    if(j<ny)
        for(int k=0; k<DIMPOINT; k++)        // Remember: the output, here, is M-by-D (-> DIMPOINT)
            gamma[j*DIMPOINT+k] = gammaj[k];
}

//////////////////////////////////////////////////////
/////////// CPU -> GPU -> CPU routines ///////////////
//////////////////////////////////////////////////////


#if !(UseCudaOnDoubles) 
extern "C" int GaussGpuGradConvXY(float ooSigma2,               // 1 / sigma^2
								float* e_h,                     // N-by-D array (same as x)
								float* alpha_h, float* x_h,     // N-by-E, N-by-D arrays
								float* y_h,     float* beta_h,  // M-by-D, M-by-E arrays
								float* gamma_h,                 // Output: M-by-D (same as y)
								int dimPoint, int dimVect, int nx, int ny){ // D, E, N, M

	// Data on the device.
	float* e_d;
	float* alpha_d;
	float* x_d;
	float* y_d;
	float* beta_d;
	float* gamma_d;

	// Allocate arrays on device.
	cudaMalloc((void**)&e_d,     sizeof(float)*(nx*dimPoint));
	cudaMalloc((void**)&alpha_d, sizeof(float)*(nx*dimVect ));
	cudaMalloc((void**)&x_d,     sizeof(float)*(nx*dimPoint));
	cudaMalloc((void**)&y_d,     sizeof(float)*(ny*dimPoint));
	cudaMalloc((void**)&beta_d,  sizeof(float)*(ny*dimVect ));
	cudaMalloc((void**)&gamma_d, sizeof(float)*(ny*dimPoint)); // Output: M-by-D (same as y)

	// Send data from host to device.
	cudaMemcpy(e_d,     e_h,     sizeof(float)*(nx*dimPoint), cudaMemcpyHostToDevice);
	cudaMemcpy(alpha_d, alpha_h, sizeof(float)*(nx*dimVect ), cudaMemcpyHostToDevice);
	cudaMemcpy(x_d,     x_h,     sizeof(float)*(nx*dimPoint), cudaMemcpyHostToDevice);
	cudaMemcpy(y_d,     y_h,     sizeof(float)*(ny*dimPoint), cudaMemcpyHostToDevice);
	cudaMemcpy(beta_d,  beta_h,  sizeof(float)*(ny*dimVect ), cudaMemcpyHostToDevice);

	// compute on device.
	dim3 blockSize;
	blockSize.x = CUDA_BLOCK_SIZE; // number of threads in each block
	dim3 gridSize;
	gridSize.x =  ny / blockSize.x + (ny%blockSize.x==0 ? 0 : 1); // NB: here, we're working columnwise !

	// Copy-paste templating, allowing us to pass the DIMPOINT and DIMVECT at compilation time : 
	// NB: Here, we use more SharedData than in the rowwise code !
	//     One block of SharedData = [ei,ai,xi], of size (dimPoint+dimVect+dimPoint)*sizeof(float)
	if(     dimPoint==1 && dimVect==1)
		GaussGpuGradConvXYOnDevice<float,1,1><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect+dimPoint)*sizeof(float)>>>
			(ooSigma2, e_d, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
	else if(dimPoint==2 && dimVect==1)
		GaussGpuGradConvXYOnDevice<float,2,1><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect+dimPoint)*sizeof(float)>>>
			(ooSigma2, e_d, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
	else if(dimPoint==3 && dimVect==1)
		GaussGpuGradConvXYOnDevice<float,3,1><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect+dimPoint)*sizeof(float)>>>
			(ooSigma2, e_d, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
	else if(dimPoint==4 && dimVect==1)
		GaussGpuGradConvXYOnDevice<float,4,1><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect+dimPoint)*sizeof(float)>>>
			(ooSigma2, e_d, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
	else if(dimPoint==2 && dimVect==2)
		GaussGpuGradConvXYOnDevice<float,2,2><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect+dimPoint)*sizeof(float)>>>
			(ooSigma2, e_d, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
	else if(dimPoint==3 && dimVect==3)
		GaussGpuGradConvXYOnDevice<float,3,3><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect+dimPoint)*sizeof(float)>>>
			(ooSigma2, e_d, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
	else if(dimPoint==4 && dimVect==4)
		GaussGpuGradConvXYOnDevice<float,4,4><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect+dimPoint)*sizeof(float)>>>
			(ooSigma2, e_d, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
	else
	{
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
	cudaThreadSynchronize();

	// Send data from device to host.
	cudaMemcpy(gamma_h, gamma_d, sizeof(float)*(ny*dimPoint),cudaMemcpyDeviceToHost); // Output: M-by-D (same as y)

	// Free memory.
	cudaFree(e_d);
	cudaFree(alpha_d);
	cudaFree(x_d);
	cudaFree(y_d);
	cudaFree(beta_d);
	cudaFree(gamma_d);

	return 0;
}

#else
//////////////////////////////////////////////////////////////
extern "C" int GaussGpuGradConvXY(double ooSigma2,               // 1 / sigma^2
								double* e_h,                     // N-by-D array (same as x)
								double* alpha_h, double* x_h,     // N-by-E, N-by-D arrays
								double* y_h,     double* beta_h,  // M-by-D, M-by-E arrays
								double* gamma_h,                 // Output: M-by-D (same as y)
								int dimPoint, int dimVect, int nx, int ny){ // D, E, N, M

	// Data on the device.
	double* e_d;
	double* alpha_d;
	double* x_d;
	double* y_d;
	double* beta_d;
	double* gamma_d;

	// Allocate arrays on device.
	cudaMalloc((void**)&e_d,     sizeof(double)*(nx*dimPoint));
	cudaMalloc((void**)&alpha_d, sizeof(double)*(nx*dimVect ));
	cudaMalloc((void**)&x_d,     sizeof(double)*(nx*dimPoint));
	cudaMalloc((void**)&y_d,     sizeof(double)*(ny*dimPoint));
	cudaMalloc((void**)&beta_d,  sizeof(double)*(ny*dimVect ));
	cudaMalloc((void**)&gamma_d, sizeof(double)*(ny*dimPoint)); // Output: M-by-D (same as y)

	// Send data from host to device.
	cudaMemcpy(e_d,     e_h,     sizeof(double)*(nx*dimPoint), cudaMemcpyHostToDevice);
	cudaMemcpy(alpha_d, alpha_h, sizeof(double)*(nx*dimVect ), cudaMemcpyHostToDevice);
	cudaMemcpy(x_d,     x_h,     sizeof(double)*(nx*dimPoint), cudaMemcpyHostToDevice);
	cudaMemcpy(y_d,     y_h,     sizeof(double)*(ny*dimPoint), cudaMemcpyHostToDevice);
	cudaMemcpy(beta_d,  beta_h,  sizeof(double)*(ny*dimVect ), cudaMemcpyHostToDevice);

	// compute on device.
	dim3 blockSize;
	blockSize.x = CUDA_BLOCK_SIZE; // number of threads in each block
	dim3 gridSize;
	gridSize.x =  ny / blockSize.x + (ny%blockSize.x==0 ? 0 : 1); // NB: here, we're working columnwise !

	// Copy-paste templating, allowing us to pass the DIMPOINT and DIMVECT at compilation time : 
	// NB: Here, we use more SharedData than in the rowwise code !
	//     One block of SharedData = [ei,ai,xi], of size (dimPoint+dimVect+dimPoint)*sizeof(double)
	if(     dimPoint==1 && dimVect==1)
		GaussGpuGradConvXYOnDevice<double,1,1><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect+dimPoint)*sizeof(double)>>>
			(ooSigma2, e_d, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
	else if(dimPoint==2 && dimVect==1)
		GaussGpuGradConvXYOnDevice<double,2,1><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect+dimPoint)*sizeof(double)>>>
			(ooSigma2, e_d, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
	else if(dimPoint==3 && dimVect==1)
		GaussGpuGradConvXYOnDevice<double,3,1><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect+dimPoint)*sizeof(double)>>>
			(ooSigma2, e_d, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
	else if(dimPoint==4 && dimVect==1)
		GaussGpuGradConvXYOnDevice<double,4,1><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect+dimPoint)*sizeof(double)>>>
			(ooSigma2, e_d, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
	else if(dimPoint==2 && dimVect==2)
		GaussGpuGradConvXYOnDevice<double,2,2><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect+dimPoint)*sizeof(double)>>>
			(ooSigma2, e_d, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
	else if(dimPoint==3 && dimVect==3)
		GaussGpuGradConvXYOnDevice<double,3,3><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect+dimPoint)*sizeof(double)>>>
			(ooSigma2, e_d, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
	else if(dimPoint==4 && dimVect==4)
		GaussGpuGradConvXYOnDevice<double,4,4><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect+dimPoint)*sizeof(double)>>>
			(ooSigma2, e_d, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
	else
	{
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
	cudaThreadSynchronize();

	// Send data from device to host.
	cudaMemcpy(gamma_h, gamma_d, sizeof(double)*(ny*dimPoint),cudaMemcpyDeviceToHost); // Output: M-by-D (same as y)

	// Free memory.
	cudaFree(e_d);
	cudaFree(alpha_d);
	cudaFree(x_d);
	cudaFree(y_d);
	cudaFree(beta_d);
	cudaFree(gamma_d);

	return 0;
}

#endif

void ExitFcn(void)
{
    cudaDeviceReset();
}
