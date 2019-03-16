#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <cuda.h>

#include "../core/CudaErrorCheck.cu"
#include "../core/Pack.h" // __INDEX__ = int32_t is defined here!

#define MY_CUDA_BLOCK_SIZE 256

namespace keops {

template< typename TYPE, int DIM >
__device__ __INDEX__ hash( TYPE* voxelsize, TYPE *point ) {
    return (__INDEX__) point[0] / voxelsize[0] ;
}


template < typename TYPE, int DIM >
__global__ void GpuGridLabelOnDevice(int npoints,
    __INDEX__* result, TYPE* voxelsize, TYPE* points) {

    // get the index of the current thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i<npoints) {
        result[i] = hash<TYPE,DIM>( voxelsize + DIMPOINTS*i, 
                                    points    + DIMPOINTS*i );
    }

}

template < typename TYPE >
static int Eval_GpuGridLabel_FromDevice(int npoints, int dimpoints, 
    __INDEX__* result, TYPE* voxelsize, TYPE* points,
    int device_id ) {
    
    if(device_id==-1) {
        // We set the GPU device on which computations will be performed
        // to be the GPU on which data is located.
        // NB. we only check location of result which is the output vector
        // so we assume that input data is on the same GPU
        // note : cudaPointerGetAttributes has a strange behaviour:
        // it looks like it makes a copy of the vector on the default GPU device (0) !!! 
	    // So we prefer to avoid this and provide directly the device_id as input (else statement below)
        cudaPointerAttributes attributes;
        CudaSafeCall(cudaPointerGetAttributes(&attributes,result));
        CudaSafeCall(cudaSetDevice(attributes.device));
    } else // device_id is provided, so we use it. Warning : is has to be consistent with location of data
        CudaSafeCall(cudaSetDevice(device_id));

    assert(("Wrong dimension!", dimpoints == DIMPOINTS)) ;

    dim3 blockSize;
    blockSize.x = MY_CUDA_BLOCK_SIZE; // number of threads in each block

    dim3 gridSize;
    gridSize.x =  npoints / blockSize.x + (npoints%blockSize.x==0 ? 0 : 1);

    GpuGridLabelOnDevice<TYPE,DIMPOINTS><<<gridSize,blockSize>>>(npoints, result, voxelsize, points);

    // block until the device has completed
    CudaSafeCall(cudaDeviceSynchronize());
    CudaCheckError();

    return 0;
}


}

extern "C" int GpuGridLabel_FromDevice(int npoints, int dimpoints, 
    __INDEX__* result, __TYPE__* voxelsize, __TYPE__* points,
    int device_id ) {
    return keops::Eval_GpuGridLabel_FromDevice(npoints, dimpoints, result, voxelsize, points, device_id );
}
