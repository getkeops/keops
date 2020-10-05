#pragma once

#include <stdio.h>
#include <sstream>
#include <assert.h>
#include <cuda.h>

#include "core/pack/Pack.h"
#include "core/pack/Load.h"
#include "core/pack/Call.h"
#include "core/pack/GetInds.h"
#include "core/pack/GetDims.h"
#include "core/utils/CudaErrorCheck.cu"
#include "core/utils/CudaSizes.h"
#include "core/utils/TypesUtils.h"

namespace keops {

template <typename T>
__device__ static constexpr T static_max_device(T a, T b) {
    return a < b ? b : a;
}

template <typename TYPE, int DIMIN, int DIMOUT, class FUN>
__global__ void reduce2D(TYPE *in, TYPE *out, int sizeY,int nx) {
    /* Function used as a final reduction pass in the 2D scheme,
     * once the block reductions have been made.
     * Takes as input:
     * - in,  a  sizeY * (nx * DIMIN ) array
     * - out, an          nx * DIMOUT   array
     *
     * Computes, in parallel, the "columnwise"-reduction (which correspond to lines of blocks)
     * of *in and stores the result in out.
     */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    /* As shown below, the code that is used to store the block-wise sum
      "tmp" in parallel is:
        if(i<nx)
            for(int k=0; k<DIMX1; k++)
                (*px)[blockIdx.y*DIMX1*nx+i*DIMX1+k] = tmp[k];
    */

    /* // This code should be a bit more efficient (more parallel) in the case
       // of a simple "fully parallel" reduction op such as "sum", "max" or "min"
    TYPE res = 0;
    if(tid < nx*DIMVECT) {
        for (int i = 0; i < sizeY; i++)
            res += in[tid + i*nx*DIMVECT]; // We use "+=" as a reduction op. But it could be anything, really!
        // res = in[tid+ nx* DIMVECT];
        out[tid] = res;
    }
    */

    // However, for now, we use a "vectorized" reduction op.,
    // which can also handle non-trivial reductions such as "LogSumExp"
    __TYPEACC__ acc[DIMIN];
    typename FUN::template InitializeReduction<__TYPEACC__,TYPE>()(acc); // acc = 0
    if(tid < nx) {
        for (int y = 0; y < sizeY; y++)
            typename FUN::template ReducePair<__TYPEACC__,TYPE>()(acc, in + (tid+y*nx)*DIMIN);     // acc += in[(tid+y*nx) *DIMVECT : +DIMVECT];
        typename FUN::template FinalizeOutput<__TYPEACC__,TYPE>()(acc, out+tid*DIMOUT, tid);
    }

}

// thread kernel: computation of x1i = sum_j k(x2i,x3i,...,y1j,y2j,...) for index i given by thread id.
// N.B.: This routine by itself is generic, and does not specifically refer to the "sum" operation.
//       It can be used for any Map-Reduce operation, provided that "fun" is well-understood.
template < typename TYPE, class FUN >
__global__ void GpuConv2DOnDevice(FUN fun, int nx, int ny, TYPE *out, TYPE **args) {
    /*
     * px, py and pp are pointers to the device global memory.
     * They are arrays of arrays with the relevant size: for instance,
     * px[1] is a TYPE array of size ( nx * DIMSX::VAL(1) ).
     *
     * out is the output array, of size (nx * DIMRED).
     *
     */
    // gets dimensions and number of variables of inputs of function FUN
    using DIMSX = typename FUN::DIMSX;  // DIMSX is a "vector" of templates giving dimensions of xi variables
    using DIMSY = typename FUN::DIMSY;  // DIMSY is a "vector" of templates giving dimensions of yj variables
    using DIMSP = typename FUN::DIMSP;  // DIMSP is a "vector" of templates giving dimensions of parameters variables
    typedef typename FUN::INDSI INDSI;
    typedef typename FUN::INDSJ INDSJ;
    typedef typename FUN::INDSP INDSP;
    const int DIMX = DIMSX::SUM;        // DIMX  is sum of dimensions for xi variables
    const int DIMY = DIMSY::SUM;        // DIMY  is sum of dimensions for yj variables
    const int DIMP = DIMSP::SUM;        // DIMP  is sum of dimensions for parameters variables
    const int DIMRED = FUN::DIMRED; // dimension of reduction operation
    const int DIMFOUT = FUN::F::DIM;     // DIMFOUT is dimension of output variable of inner function


    TYPE fout[DIMFOUT];

    // Load the parameter vector in the Thread Memory, for improved efficiency
    //TYPE param_loc[static_max_device(DIMP,1)];
    // (Jean :) Direct inlining to compile on Ubuntu 16.04 with nvcc7.5,
    //          which is a standard config in research. For whatever reason, I can't make
    //          it work an other way... Is it bad practice/performance?
    TYPE param_loc[DIMP < 1 ? 1 : DIMP];
    load<DIMSP, INDSP>(0,param_loc,args); // load parameters variables from global memory to local thread memory
    
    // Weird syntax to create a pointer in shared memory.
    extern __shared__ char yj_char[];
    TYPE* const yj = reinterpret_cast<TYPE*>(yj_char);

    // Step 1 : Load in Thread Memory the information needed in the current line ---------------------------
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    TYPE xi[DIMX < 1 ? 1 : DIMX];
#if SUM_SCHEME == BLOCK_SUM 
// N.B. To be consistent with the convention used in GpuConv1D, when SUM_SCHEME == BLOCK_SUM=1 we accumulate results in TYPE 
// instead of __TYPEACC__ in each block, __TYPEACC__ will be used only to sum up results from each block
    TYPE acc[DIMRED];
#else
    __TYPEACC__ acc[DIMRED];
#endif
#if SUM_SCHEME == KAHAN_SCHEME
    const int DIM_KAHAN = FUN::template KahanScheme<__TYPEACC__,TYPE>::DIMACC;
    TYPE tmp[DIM_KAHAN];
#endif
    if(i<nx) { // we will compute outi only if i is in the range
#if SUM_SCHEME == BLOCK_SUM         
        typename FUN::template InitializeReduction<TYPE,TYPE>()(acc); // acc = 0
#else
        typename FUN::template InitializeReduction<__TYPEACC__,TYPE>()(acc); // acc = 0
#endif
#if SUM_SCHEME == KAHAN_SCHEME
        VectAssign<DIM_KAHAN>(tmp,0.0f);
#endif
        // Load xi from device global memory.
        // Remember that we use an interleaved memory scheme where
        // xi = [ x1i, x2i, x3i, ... ].
	load< DIMSX, INDSI>(i,xi,args); // load xi variables from global memory to local thread memory
    }

    // Step 2 : Load in Shared Memory the information needed in the current block of the product -----------
    // In the 1D scheme, we use a loop to run through the line.
    // In the 2D scheme presented here, the computation is done in parallel wrt both lines and columns.
    // Hence, we use "blockId.y" to get our current column number.
    int j = blockIdx.y * blockDim.x + threadIdx.x; // Same blockDim in x and y : squared tiles.
    if(j<ny) // we load yj from device global memory only if j<ny
        load<DIMSY,INDSJ>(j,yj+threadIdx.x*DIMY,args); // load yj variables from global memory to shared memory
    // More precisely : the j-th line of py is loaded to yj, at a location which depends on the
    // current threadId.

    __syncthreads(); // Make sure nobody lags behind

    // Step 3 : Once the data is loaded, execute fun --------------------------------------------------------
    // N.B.: There's no explicit summation here. Just calls to fun, which *accumulates* the results
    //       along the line, but does not *have* to use a "+=" as reduction operator.
    //       In the future, we could provide other reductions: max, min, ... whatever's needed.

    if(i<nx) { // we compute x1i only if needed
        TYPE* yjrel = yj; // Loop on the columns of the current block.
        for(int jrel = 0; (jrel<blockDim.x) && ((blockDim.x*blockIdx.y+jrel)< ny); jrel++, yjrel+=DIMY) {
            call<DIMSX,DIMSY,DIMSP>(fun,fout,xi,yjrel,param_loc); // Call the function, which outputs results in fout
#if SUM_SCHEME == BLOCK_SUM
#if USE_HALF
        int ind = blockDim.x*blockIdx.y+jrel;
        typename FUN::template ReducePairShort<TYPE,TYPE>()(acc, fout, __floats2half2_rn(2*ind,2*ind+1));     // acc += fout
#else
        typename FUN::template ReducePairShort<TYPE,TYPE>()(acc, fout, blockDim.x*blockIdx.y+jrel);     // acc += fout
#endif
#elif SUM_SCHEME == KAHAN_SCHEME
            typename FUN::template KahanScheme<__TYPEACC__,TYPE>()(acc, fout, tmp);   
#else
#if USE_HALF
        int ind = blockDim.x*blockIdx.y+jrel;
        typename FUN::template ReducePairShort<TYPE,TYPE>()(acc, fout, __floats2half2_rn(2*ind,2*ind+1));     // acc += fout
#else
        typename FUN::template ReducePairShort<TYPE,TYPE>()(acc, fout, blockDim.x*blockIdx.y+jrel);     // acc += fout
#endif
#endif
        }
    }
    __syncthreads();

    // Step 4 : Save the result in global memory -----------------------------------------------------------
    // The current thread has computed the "linewise-sum" of a small block of the full Kernel Product
    // matrix, which corresponds to KP[ blockIdx.x * blockDim.x : (blockIdx.x+1) * blockDim.x ,
    //                                  blockIdx.y * blockDim.x : (blockIdx.y+1) * blockDim.x ]
    // We accumulate it in the output array out, which has in fact gridSize.y * nx
    // lines of size DIMRED. The final reduction, which "sums over the block lines",
    // shall be done in a later step.
    if(i<nx)
        for(int k=0; k<DIMRED; k++)
            out[blockIdx.y*DIMRED*nx+i*DIMRED+k] = acc[k];
}
///////////////////////////////////////////////////


struct GpuConv2D_FromHost {
template < typename TYPE, class FUN >
static int Eval_(FUN fun, int nx, int ny, TYPE *out, TYPE **args_h) {

    using DIMSX = typename FUN::DIMSX;
    using DIMSY = typename FUN::DIMSY;
    using DIMSP = typename FUN::DIMSP;
    typedef typename FUN::INDSI INDSI;
    typedef typename FUN::INDSJ INDSJ;
    typedef typename FUN::INDSP INDSP;
    const int DIMX = DIMSX::SUM;
    const int DIMY = DIMSY::SUM;
    const int DIMP = DIMSP::SUM;
    const int DIMOUT = FUN::DIM; // dimension of output variable
    const int DIMRED = FUN::DIMRED; // dimension of reduction operation
    const int SIZEI = DIMSX::SIZE;
    const int SIZEJ = DIMSY::SIZE;
    const int SIZEP = DIMSP::SIZE;
    static const int NMINARGS = FUN::NMINARGS;

    // Compute on device : grid is 2d and block is 1d
    int dev = -1;
    CudaSafeCall(cudaGetDevice(&dev));

    SetGpuProps(dev);

    dim3 blockSize;
    // warning : blockSize.x was previously set to CUDA_BLOCK_SIZE; currently CUDA_BLOCK_SIZE value is used as a bound.
    blockSize.x = ::std::min(CUDA_BLOCK_SIZE,::std::min(maxThreadsPerBlock, (int) (sharedMemPerBlock / ::std::max(1, (int)(DIMY*sizeof(TYPE))) ))); // number of threads in each block

    dim3 gridSize;
    gridSize.x =  nx / blockSize.x + (nx%blockSize.x==0 ? 0 : 1);
    gridSize.y =  ny / blockSize.x + (ny%blockSize.x==0 ? 0 : 1);

    // Reduce  : grid and block are both 1d
    dim3 blockSize2;
    blockSize2.x = CUDA_BLOCK_SIZE; // number of threads in each block
    dim3 gridSize2;
    gridSize2.x =  (nx*DIMRED) / blockSize2.x + ((nx*DIMRED)%blockSize2.x==0 ? 0 : 1);

    // Data on the device. We need an "inflated" outB, which contains gridSize.y "copies" of out
    // that will be reduced in the final pass.
    TYPE *outB, *out_d;

    // device array of pointers to device data
    TYPE **args_d;

    // single cudaMalloc
    void *p_data;
    CudaSafeCall(cudaMalloc(&p_data, sizeof(TYPE*)*NMINARGS+sizeof(TYPE)*(DIMP+nx*(DIMX+DIMOUT)+ny*DIMY+nx*DIMRED*gridSize.y)));

    args_d = (TYPE **) p_data;
    TYPE *dataloc = (TYPE *) (args_d + NMINARGS);
    out_d = dataloc;
    dataloc += nx*DIMOUT;

    // host array of pointers to device data
    TYPE *ph[NMINARGS];

      for (int k = 0; k < SIZEP; k++) {
        int indk = INDSP::VAL(k);
        int nvals = DIMSP::VAL(k);        
        CudaSafeCall(cudaMemcpy(dataloc, args_h[indk], sizeof(TYPE) * nvals, cudaMemcpyHostToDevice));
        ph[indk] = dataloc;
        dataloc += nvals;
      }

    for (int k = 0; k < SIZEI; k++) {
      int indk = INDSI::VAL(k);
      int nvals = nx * DIMSX::VAL(k);
      CudaSafeCall(cudaMemcpy(dataloc, args_h[indk], sizeof(TYPE) * nvals, cudaMemcpyHostToDevice));
      ph[indk] = dataloc;
      dataloc += nvals;
    }

      for (int k = 0; k < SIZEJ; k++) {
        int indk = INDSJ::VAL(k);
        int nvals = ny * DIMSY::VAL(k);
        CudaSafeCall(cudaMemcpy(dataloc, args_h[indk], sizeof(TYPE) * nvals, cudaMemcpyHostToDevice));
        ph[indk] = dataloc;
        dataloc += nvals;
      }

    outB = dataloc; // we write the result before reduction in the "inflated" vector

    // copy arrays of pointers
    CudaSafeCall(cudaMemcpy(args_d, ph, NMINARGS * sizeof(TYPE *), cudaMemcpyHostToDevice));

    // Size of the SharedData : blockSize.x*(DIMY)*sizeof(TYPE)
    GpuConv2DOnDevice<TYPE><<<gridSize,blockSize,blockSize.x*(DIMY)*sizeof(TYPE)>>>(fun,nx,ny,outB,args_d);

    // block until the device has completed
    CudaSafeCall(cudaDeviceSynchronize());
    CudaCheckError();

    // Since we've used a 2D scheme, there's still a "blockwise" line reduction to make on
    // the output array outB. We go from shape ( gridSize.y * nx, DIMRED ) to (nx, DIMOUT)
    reduce2D<TYPE,DIMRED,DIMOUT,FUN><<<gridSize2, blockSize2>>>(outB, out_d, gridSize.y,nx);

    // block until the device has completed
    CudaSafeCall(cudaDeviceSynchronize());
    CudaCheckError();

    // Send data from device to host.
    CudaSafeCall(cudaMemcpy(out, out_d, sizeof(TYPE)*(nx*DIMOUT),cudaMemcpyDeviceToHost));

    // Free memory.
    CudaSafeCall(cudaFree(p_data));

    return 0;
}


// Wrapper around GpuConv2D, which takes lists of arrays *x1, *x2, ..., *y1, *y2, ...
// and use getlist to enroll them into "pointers arrays" px and py.
template < typename TYPE, class FUN, typename... Args >
static int Eval(FUN fun, int nx, int ny, int device_id, TYPE *out, Args... args) {

    // We set the GPU device on which computations will be performed
    if(device_id!=-1)
        CudaSafeCall(cudaSetDevice(device_id));

    static const int Nargs = sizeof...(Args);
    TYPE *pargs[Nargs];
    unpack(pargs, args...);

    return Eval_(fun,nx,ny,out,pargs);

}

// same without the device_id argument
template < typename TYPE, class FUN, typename... Args >
static int Eval(FUN fun, int nx, int ny, TYPE *out, Args... args) {
    return Eval(fun, nx, ny, -1, out, args...);
}

// Idem, but with args given as an array of arrays, instead of an explicit list of arrays
template < typename TYPE, class FUN >
static int Eval(FUN fun, int nx, int ny, TYPE *out, TYPE **pargs, int device_id=-1) {

    // We set the GPU device on which computations will be performed
    if(device_id!=-1)
        CudaSafeCall(cudaSetDevice(device_id));

    return Eval_(fun,nx,ny,out,pargs);

}



};


struct GpuConv2D_FromDevice {
template < typename TYPE, class FUN >
static int Eval_(FUN fun, int nx, int ny, TYPE *out, TYPE **args) {

    static const int DIMRED = FUN::DIMRED;
    static const int DIMOUT = FUN::DIM;
    static const int NMINARGS = FUN::NMINARGS;

    // Data on the device. We need an "inflated" outB, which contains gridSize.y "copies" of out
    // that will be reduced in the final pass.
    TYPE *outB;

    // device array of pointers to device data
    TYPE **args_d;

    // Compute on device : grid is 2d and block is 1d
    int dev = -1;
    CudaSafeCall(cudaGetDevice(&dev));

    SetGpuProps(dev);

    dim3 blockSize;
      typedef typename FUN::DIMSY DIMSY;
      const int DIMY = DIMSY::SUM;
    // warning : blockSize.x was previously set to CUDA_BLOCK_SIZE; currently CUDA_BLOCK_SIZE value is used as a bound.
    blockSize.x = ::std::min(CUDA_BLOCK_SIZE,::std::min(maxThreadsPerBlock, (int) (sharedMemPerBlock / ::std::max(1, (int)(DIMY*sizeof(TYPE))) ))); // number of threads in each block

    dim3 gridSize;
    gridSize.x =  nx / blockSize.x + (nx%blockSize.x==0 ? 0 : 1);
    gridSize.y =  ny / blockSize.x + (ny%blockSize.x==0 ? 0 : 1);

    // Reduce : grid and block are both 1d
    dim3 blockSize2;
    blockSize2.x = blockSize.x; // number of threads in each block
    dim3 gridSize2;
    gridSize2.x =  (nx*DIMRED) / blockSize2.x + ((nx*DIMRED)%blockSize2.x==0 ? 0 : 1);

    // single cudaMalloc
    void *p_data;
    CudaSafeCall(cudaMalloc(&p_data, sizeof(TYPE*)*NMINARGS + sizeof(TYPE)*(nx*DIMRED*gridSize.y)));

    args_d = (TYPE **) p_data;
    CudaSafeCall(cudaMemcpy(args_d, args, NMINARGS * sizeof(TYPE *), cudaMemcpyHostToDevice));

    outB = (TYPE *) (args_d + NMINARGS);

    // Size of the SharedData : blockSize.x*(DIMY)*sizeof(TYPE)
    GpuConv2DOnDevice<TYPE><<<gridSize,blockSize,blockSize.x*(DIMY)*sizeof(TYPE)>>>(fun,nx,ny,outB,args_d);

    // block until the device has completed
    CudaSafeCall(cudaDeviceSynchronize());
    CudaCheckError();

    // Since we've used a 2D scheme, there's still a "blockwise" line reduction to make on
    // the output array px_d[0] = x1B. We go from shape ( gridSize.y * nx, DIMRED ) to (nx, DIMOUT)
    reduce2D<TYPE,DIMRED,DIMOUT,FUN><<<gridSize2, blockSize2>>>(outB, out, gridSize.y,nx);

    // block until the device has completed
    CudaSafeCall(cudaDeviceSynchronize());
    CudaCheckError();

    CudaSafeCall(cudaFree(p_data));

    return 0;
}


// Same wrappers, but for data located on the device
template < typename TYPE, class FUN, typename... Args >
static int Eval(FUN fun, int nx, int ny, int device_id, TYPE *out, Args... args) {

    // device_id is provided, so we set the GPU device accordingly
    // Warning : is has to be consistent with location of data
    CudaSafeCall(cudaSetDevice(device_id));

    static const int Nargs = sizeof...(Args);
    TYPE *pargs[Nargs];
    unpack(pargs, args...);

    return Eval_(fun,nx,ny,out,pargs);
}

// same without the device_id argument
template < typename TYPE, class FUN, typename... Args >
static int Eval(FUN fun, int nx, int ny, TYPE *out, Args... args) {
    // We set the GPU device on which computations will be performed
    // to be the GPU on which data is located.
    // NB. we only check location of x1_d which is the output vector
    // so we assume that input data is on the same GPU
    // note : cudaPointerGetAttributes has a strange behaviour:
    // it looks like it makes a copy of the vector on the default GPU device (0) !!! 
    // So we prefer to avoid this and provide directly the device_id as input (first function above)
    cudaPointerAttributes attributes;
    CudaSafeCall(cudaPointerGetAttributes(&attributes,out));
    return Eval(fun, nx, ny, attributes.device, out, args...);
}

template < typename TYPE, class FUN >
static int Eval(FUN fun, int nx, int ny, TYPE *out, TYPE **pargs, int device_id=-1) {

    if(device_id==-1) {
        // We set the GPU device on which computations will be performed
        // to be the GPU on which data is located.
        // NB. we only check location of x1_d which is the output vector
        // so we assume that input data is on the same GPU
        // note : cudaPointerGetAttributes has a strange behaviour:
        // it looks like it makes a copy of the vector on the default GPU device (0) !!! 
	// So we prefer to avoid this and provide directly the device_id as input (else statement below)
        cudaPointerAttributes attributes;
        CudaSafeCall(cudaPointerGetAttributes(&attributes,out));
        CudaSafeCall(cudaSetDevice(attributes.device));
    } else // device_id is provided, so we use it. Warning : is has to be consistent with location of data
        CudaSafeCall(cudaSetDevice(device_id));

    return Eval_(fun,nx,ny,out,pargs);

}

};

}
