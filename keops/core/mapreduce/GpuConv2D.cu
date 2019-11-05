#pragma once

#include <stdio.h>
#include <sstream>
#include <assert.h>
#include <cuda.h>

#include "core/pack/Pack.h"
#include "core/pack/GetInds.h"
#include "core/pack/GetDims.h"
#include "core/utils/CudaErrorCheck.cu"
#include "core/utils/CudaSizes.h"

namespace keops {

template <typename T>
__device__ static constexpr T static_max_device(T a, T b) {
    return a < b ? b : a;
}

// thread kernel: computation of x1i = sum_j k(x2i,x3i,...,y1j,y2j,...) for index i given by thread id.
// N.B.: This routine by itself is generic, and does not specifically refer to the "sum" operation.
//       It can be used for any Map-Reduce operation, provided that "fun" is well-understood.
template < typename TYPE, class FUN >
__global__ void GpuConv2DOnDevice(FUN fun, int nx, int ny, TYPE** px, TYPE** py, TYPE** pp) {
    /*
     * px, py and pp are pointers to the device global memory.
     * They are arrays of arrays with the relevant size: for instance,
     * px[1] is a TYPE array of size ( nx * DIMSX::VAL(1) ).
     *
     * (*px) = px[0] is the output array, of size (nx * DIMRED).
     *
     */
    // gets dimensions and number of variables of inputs of function FUN
    using DIMSX = typename FUN::DIMSX;  // DIMSX is a "vector" of templates giving dimensions of xi variables
    using DIMSY = typename FUN::DIMSY;  // DIMSY is a "vector" of templates giving dimensions of yj variables
    using DIMSP = typename FUN::DIMSP;  // DIMSP is a "vector" of templates giving dimensions of parameters variables
    const int DIMX = DIMSX::SUM;        // DIMX  is sum of dimensions for xi variables
    const int DIMY = DIMSY::SUM;        // DIMY  is sum of dimensions for yj variables
    const int DIMP = DIMSP::SUM;        // DIMP  is sum of dimensions for parameters variables
    const int DIMRED = FUN::DIMRED; // dimension of reduction operation
    const int DIMFOUT = DIMSX::FIRST;     // DIMFOUT is dimension of output variable of inner function

    // Load the parameter vector in the Thread Memory, for improved efficiency
    //TYPE param_loc[static_max_device(DIMP,1)];
    // (Jean :) Direct inlining to compile on Ubuntu 16.04 with nvcc7.5,
    //          which is a standard config in research. For whatever reason, I can't make
    //          it work an other way... Is it bad practice/performance?
    TYPE param_loc[DIMP < 1 ? 1 : DIMP];
	load<DIMSP>(0,param_loc,pp); // load parameters variables from global memory to local thread memory
    
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
    if(i<nx) { // we will compute x1i only if i is in the range
        if (blockIdx.y==0)
            typename FUN::template InitializeReduction<TYPE>()(px[0] + i * DIMRED); 
#if SUM_SCHEME == BLOCK_SUM         
        typename FUN::template InitializeReduction<TYPE>()(acc); // acc = 0
#else
        typename FUN::template InitializeReduction<__TYPEACC__>()(acc); // acc = 0
#endif
#if SUM_SCHEME == KAHAN_SCHEME
#pragma unroll
        for (int k = 0; k < DIM_KAHAN; k++)
          tmp[k] = 0.0f;
#endif
        // Load xi from device global memory.
        // Remember that we use an interleaved memory scheme where
        // xi = [ x1i, x2i, x3i, ... ].
        // Since we do not want to erase x1i, and only load x2i, x3i, etc.,
        // we add a small offset to the pointer given as an argument to the loading routine,
        // and ask it to only load "DIMSX::NEXT" bits of memory.
	load<typename DIMSX::NEXT>(i,xi+DIMFOUT,px+1); // load xi variables from global memory to local thread memory
    }

    // Step 2 : Load in Shared Memory the information needed in the current block of the product -----------
    // In the 1D scheme, we use a loop to run through the line.
    // In the 2D scheme presented here, the computation is done in parallel wrt both lines and columns.
    // Hence, we use "blockId.y" to get our current column number.
    int j = blockIdx.y * blockDim.x + threadIdx.x; // Same blockDim in x and y : squared tiles.
    if(j<ny) // we load yj from device global memory only if j<ny
        load<DIMSY>(j,yj+threadIdx.x*DIMY,py); // load yj variables from global memory to shared memory
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
            call<DIMSX,DIMSY,DIMSP>(fun,xi,yjrel,param_loc); // Call the function, which accumulates results in xi[0:DIMX1]
#if SUM_SCHEME == BLOCK_SUM
            typename FUN::template ReducePairShort<TYPE,TYPE>()(acc, xi, blockDim.x*blockIdx.y+jrel);     // acc += xi
#elif SUM_SCHEME == KAHAN_SCHEME
            typename FUN::template KahanScheme<__TYPEACC__,TYPE>()(acc, xi, tmp);   
#else
            typename FUN::template ReducePairShort<__TYPEACC__,TYPE>()(acc, xi, blockDim.x*blockIdx.y+jrel);     // acc += xi
#endif
        }
    }
    __syncthreads();

    // Step 4 : Save the result in global memory -----------------------------------------------------------
    // The current thread has computed the "linewise-sum" of a small block of the full Kernel Product
    // matrix, which corresponds to KP[ blockIdx.x * blockDim.x : (blockIdx.x+1) * blockDim.x ,
    //                                  blockIdx.y * blockDim.x : (blockIdx.y+1) * blockDim.x ]
    // We accumulate it in the output array (*px) = px[0], which has in fact gridSize.y * nx
    // lines of size DIMRED. The final reduction, which "sums over the block lines",
    // shall be done in a later step.
    if(i<nx) {

extern __shared__ int test;
if (i==0)
    test = 1;
__syncthreads();
while(test)
{
if (i==0)
{
  TYPE* out = px[0];
  int cnt = 0;
  while(1)
  {
    cnt++;
    TYPE tmp1 = *out;
    if (tmp1 == 0)
      *out = blockIdx.y+1;
    TYPE tmp2 = *out;
    printf("here in block %d, cnt=%d, *out before = %f, *out after = %f\n", blockIdx.y, cnt, tmp1, tmp2);
    if (tmp2 == blockIdx.y+1)
      break;
  }
}
__syncthreads();
TYPE outi[DIMRED];
if (i)
{
    for (int k=0; k<DIMRED; k++)
        outi[k] = (px[0] + i * DIMRED)[k];
    typename FUN::template ReducePairShort<TYPE,__TYPEACC__>()(outi, acc, 0);
}
__syncthreads();
if (i==0)
if (*px[0]==blockIdx.y+1 )
    test = 0;
else
    printf("block %d : restart\n",blockIdx.y);
__syncthreads();
if (i && test==0)
    for (int k=0; k<DIMRED; k++)
        (px[0] + i * DIMRED)[k] = outi[k];
__syncthreads();
if(i==0 && test==0)
if(*px[0]==blockIdx.y+1)
    *px[0] = 0;
else
    printf("block %d : error!!!!\n",blockIdx.y);
__syncthreads();
}

}

}
///////////////////////////////////////////////////


struct GpuConv2D_FromHost {
template < typename TYPE, class FUN >
static int Eval_(FUN fun, int nx, int ny, TYPE** px_h, TYPE** py_h, TYPE** pp_h) {

    using DIMSX = typename FUN::DIMSX;
    using DIMSY = typename FUN::DIMSY;
    using DIMSP = typename FUN::DIMSP;
    const int DIMX = DIMSX::SUM;
    const int DIMY = DIMSY::SUM;
    const int DIMP = DIMSP::SUM;
    const int DIMOUT = FUN::DIM; // dimension of output variable
    const int DIMFOUT = DIMSX::FIRST;     // DIMFOUT is dimension of output variable of inner function
    const int SIZEI = DIMSX::SIZE;
    const int SIZEJ = DIMSY::SIZE;
    const int SIZEP = DIMSP::SIZE;

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

    // Data on the device. 
    TYPE *x_d, *y_d, *param_d;

    // device arrays of pointers to device data
    TYPE **px_d, **py_d, **pp_d;

    // single cudaMalloc
    void **p_data;
    CudaSafeCall(cudaMalloc((void**)&p_data, sizeof(TYPE*)*(SIZEI+SIZEJ+SIZEP)+sizeof(TYPE)*(DIMP+nx*(DIMX-DIMFOUT+DIMOUT)+ny*DIMY)));

    TYPE **p_data_a = (TYPE**)p_data;
    px_d = p_data_a;
    p_data_a += SIZEI;
    py_d = p_data_a;
    p_data_a += SIZEJ;
    pp_d = p_data_a;
    p_data_a += SIZEP;
    TYPE *p_data_b = (TYPE*)p_data_a;
    param_d = p_data_b;
    p_data_b += DIMP;
    x_d = p_data_b;
    p_data_b += nx*(DIMX-DIMFOUT+DIMOUT);
    y_d = p_data_b;

    // host arrays of pointers to device data
    TYPE *phx_d[SIZEI];
    TYPE *phy_d[SIZEJ];
    TYPE *php_d[SIZEP];

    // Send data from host to device.
    int nvals;

    // if DIMSP is empty (i.e. no parameter), nvals = -1 which could result in a segfault
    if(SIZEP > 0){ 
      php_d[0] = param_d;
      nvals = DIMSP::VAL(0);
      CudaSafeCall(cudaMemcpy(php_d[0], pp_h[0], sizeof(TYPE)*nvals, cudaMemcpyHostToDevice));

      for(int k=1; k<SIZEP; k++) {
        php_d[k] = php_d[k-1] + nvals;
        nvals = DIMSP::VAL(k);
        CudaSafeCall(cudaMemcpy(php_d[k], pp_h[k], sizeof(TYPE)*nvals, cudaMemcpyHostToDevice));
      }
    }

    phx_d[0] = x_d;
    nvals = nx*DIMOUT;
    for(int k=1; k<SIZEI; k++) {
        phx_d[k] = phx_d[k-1] + nvals;
        nvals = nx*DIMSX::VAL(k);
        CudaSafeCall(cudaMemcpy(phx_d[k], px_h[k], sizeof(TYPE)*nvals, cudaMemcpyHostToDevice));
    }

    // if DIMSY is empty (i.e. no Vj variable), nvals = -1 which could result in a segfault
    if (SIZEJ > 0) {
      phy_d[0] = y_d;
      nvals = ny * DIMSY::VAL(0);
      CudaSafeCall(cudaMemcpy(phy_d[0], py_h[0], sizeof(TYPE)*nvals, cudaMemcpyHostToDevice));
      for(int k=1; k<SIZEJ; k++) {
        phy_d[k] = phy_d[k-1] + nvals;
        nvals = ny*max(0, (int) DIMSY::VAL(k));
        CudaSafeCall(cudaMemcpy(phy_d[k], py_h[k], sizeof(TYPE)*nvals, cudaMemcpyHostToDevice));
      }
    }

    // copy arrays of pointers
    CudaSafeCall(cudaMemcpy(px_d, phx_d, SIZEI*sizeof(TYPE*), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(py_d, phy_d, SIZEJ*sizeof(TYPE*), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(pp_d, php_d, SIZEP*sizeof(TYPE*), cudaMemcpyHostToDevice));

    // Size of the SharedData : blockSize.x*(DIMY)*sizeof(TYPE)
    GpuConv2DOnDevice<TYPE><<<gridSize,blockSize,blockSize.x*(DIMY)*sizeof(TYPE)>>>(fun,nx,ny,px_d,py_d,pp_d);

    // block until the device has completed
    CudaSafeCall(cudaDeviceSynchronize());
    CudaCheckError();

    // Send data from device to host.
    CudaSafeCall(cudaMemcpy(*px_h, x_d, sizeof(TYPE)*(nx*DIMOUT),cudaMemcpyDeviceToHost));

    // Free memory.
    CudaSafeCall(cudaFree(p_data));

    return 0;
}


// Wrapper around GpuConv2D, which takes lists of arrays *x1, *x2, ..., *y1, *y2, ...
// and use getlist to enroll them into "pointers arrays" px and py.
template < typename TYPE, class FUN, typename... Args >
static int Eval(FUN fun, int nx, int ny, int device_id, TYPE* x1_h, Args... args) {

    // We set the GPU device on which computations will be performed
    if(device_id!=-1)
        CudaSafeCall(cudaSetDevice(device_id));

    typedef typename FUN::VARSI VARSI;
    typedef typename FUN::VARSJ VARSJ;
    typedef typename FUN::VARSP VARSP;

    const int SIZEI = VARSI::SIZE+1;
    const int SIZEJ = VARSJ::SIZE;
    const int SIZEP = VARSP::SIZE;

    using DIMSX = GetDims<VARSI>;
    using DIMSY = GetDims<VARSJ>;
    using DIMSP = GetDims<VARSP>;

    using INDSI = GetInds<VARSI>;
    using INDSJ = GetInds<VARSJ>;
    using INDSP = GetInds<VARSP>;

    TYPE *px_h[SIZEI];
    TYPE *py_h[SIZEJ];
    TYPE *pp_h[SIZEP];

    px_h[0] = x1_h;
    getlist<INDSI>(px_h+1,args...);
    getlist<INDSJ>(py_h,args...);
    getlist<INDSP>(pp_h,args...);

    return Eval_(fun,nx,ny,px_h,py_h,pp_h);

}

// same without the device_id argument
template < typename TYPE, class FUN, typename... Args >
static int Eval(FUN fun, int nx, int ny, TYPE* x1_h, Args... args) {
    return Eval(fun, nx, ny, -1, x1_h, args...);
}

// Idem, but with args given as an array of arrays, instead of an explicit list of arrays
template < typename TYPE, class FUN >
static int Eval(FUN fun, int nx, int ny, TYPE* x1_h, TYPE** args, int device_id=-1) {

    // We set the GPU device on which computations will be performed
    if(device_id!=-1)
        CudaSafeCall(cudaSetDevice(device_id));

    typedef typename FUN::VARSI VARSI;
    typedef typename FUN::VARSJ VARSJ;
    typedef typename FUN::VARSP VARSP;

    const int SIZEI = VARSI::SIZE+1;
    const int SIZEJ = VARSJ::SIZE;
    const int SIZEP = VARSP::SIZE;

    using DIMSX = GetDims<VARSI>;
    using DIMSY = GetDims<VARSJ>;
    using DIMSP = GetDims<VARSP>;

    using INDSI = GetInds<VARSI>;
    using INDSJ = GetInds<VARSJ>;
    using INDSP = GetInds<VARSP>;

    TYPE *px_h[SIZEI];
    TYPE *py_h[SIZEJ];
    TYPE *pp_h[SIZEP];

    px_h[0] = x1_h;
    for(int i=1; i<SIZEI; i++)
        px_h[i] = args[INDSI::VAL(i-1)];
    for(int i=0; i<SIZEJ; i++)
        py_h[i] = args[INDSJ::VAL(i)];
    for(int i=0; i<SIZEP; i++)
        pp_h[i] = args[INDSP::VAL(i)];

    return Eval_(fun,nx,ny,px_h,py_h,pp_h);

}



};


struct GpuConv2D_FromDevice {
template < typename TYPE, class FUN >
static int Eval_(FUN fun, int nx, int ny, TYPE** phx_d, TYPE** phy_d, TYPE** php_d) {

    typedef typename FUN::DIMSX DIMSX;
    typedef typename FUN::DIMSY DIMSY;
    typedef typename FUN::DIMSP DIMSP;
    const int DIMY = DIMSY::SUM;
    const int SIZEI = DIMSX::SIZE;
    const int SIZEJ = DIMSY::SIZE;
    const int SIZEP = DIMSP::SIZE;

    // device arrays of pointers to device data
    TYPE **px_d, **py_d, **pp_d;

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

    // single cudaMalloc
    void **p_data;

    CudaSafeCall(cudaMalloc((void**)&p_data, sizeof(TYPE*)*(SIZEI+SIZEJ+SIZEP)));

    TYPE **p_data_a = (TYPE**)p_data;
    px_d = p_data_a;
    p_data_a += SIZEI;
    py_d = p_data_a;
    p_data_a += SIZEJ;
    pp_d = p_data_a;

    CudaSafeCall(cudaMemcpy(px_d, phx_d, SIZEI*sizeof(TYPE*), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(py_d, phy_d, SIZEJ*sizeof(TYPE*), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(pp_d, php_d, SIZEP*sizeof(TYPE*), cudaMemcpyHostToDevice));

    // Size of the SharedData : blockSize.x*(DIMY)*sizeof(TYPE)
    GpuConv2DOnDevice<TYPE><<<gridSize,blockSize,blockSize.x*(DIMY)*sizeof(TYPE)>>>(fun,nx,ny,px_d,py_d,pp_d);

    // block until the device has completed
    CudaSafeCall(cudaDeviceSynchronize());
    CudaCheckError();

    CudaSafeCall(cudaFree(p_data));

    return 0;
}


// Same wrappers, but for data located on the device
template < typename TYPE, class FUN, typename... Args >
static int Eval(FUN fun, int nx, int ny, int device_id, TYPE* x1_d, Args... args) {

    // device_id is provided, so we set the GPU device accordingly
    // Warning : is has to be consistent with location of data
    CudaSafeCall(cudaSetDevice(device_id));

    typedef typename FUN::VARSI VARSI;
    typedef typename FUN::VARSJ VARSJ;
    typedef typename FUN::VARSP VARSP;

    const int SIZEI = VARSI::SIZE+1;
    const int SIZEJ = VARSJ::SIZE;
    const int SIZEP = VARSP::SIZE;

    using DIMSX = GetDims<VARSI>;
    using DIMSY = GetDims<VARSJ>;
    using DIMSP = GetDims<VARSP>;

    using INDSI = GetInds<VARSI>;
    using INDSJ = GetInds<VARSJ>;
    using INDSP = GetInds<VARSP>;

    TYPE *px_d[SIZEI];
    TYPE *py_d[SIZEJ];
    TYPE *pp_d[SIZEP];

    px_d[0] = x1_d;
    getlist<INDSI>(px_d+1,args...);
    getlist<INDSJ>(py_d,args...);
    getlist<INDSP>(pp_d,args...);

    return Eval_(fun,nx,ny,px_d,py_d,pp_d);
}

// same without the device_id argument
template < typename TYPE, class FUN, typename... Args >
static int Eval(FUN fun, int nx, int ny, TYPE* x1_d, Args... args) {
    // We set the GPU device on which computations will be performed
    // to be the GPU on which data is located.
    // NB. we only check location of x1_d which is the output vector
    // so we assume that input data is on the same GPU
    // note : cudaPointerGetAttributes has a strange behaviour:
    // it looks like it makes a copy of the vector on the default GPU device (0) !!! 
    // So we prefer to avoid this and provide directly the device_id as input (first function above)
    cudaPointerAttributes attributes;
    CudaSafeCall(cudaPointerGetAttributes(&attributes,x1_d));
    return Eval(fun, nx, ny, attributes.device, x1_d, args...);
}

template < typename TYPE, class FUN >
static int Eval(FUN fun, int nx, int ny, TYPE* x1_d, TYPE** args, int device_id=-1) {

    if(device_id==-1) {
        // We set the GPU device on which computations will be performed
        // to be the GPU on which data is located.
        // NB. we only check location of x1_d which is the output vector
        // so we assume that input data is on the same GPU
        // note : cudaPointerGetAttributes has a strange behaviour:
        // it looks like it makes a copy of the vector on the default GPU device (0) !!! 
	// So we prefer to avoid this and provide directly the device_id as input (else statement below)
        cudaPointerAttributes attributes;
        CudaSafeCall(cudaPointerGetAttributes(&attributes,x1_d));
        CudaSafeCall(cudaSetDevice(attributes.device));
    } else // device_id is provided, so we use it. Warning : is has to be consistent with location of data
        CudaSafeCall(cudaSetDevice(device_id));

    typedef typename FUN::VARSI VARSI;
    typedef typename FUN::VARSJ VARSJ;
    typedef typename FUN::VARSP VARSP;

    const int SIZEI = VARSI::SIZE+1;
    const int SIZEJ = VARSJ::SIZE;
    const int SIZEP = VARSP::SIZE;

    using DIMSX = GetDims<VARSI>;
    using DIMSY = GetDims<VARSJ>;
    using DIMSP = GetDims<VARSP>;

    using INDSI = GetInds<VARSI>;
    using INDSJ = GetInds<VARSJ>;
    using INDSP = GetInds<VARSP>;

    TYPE *phx_d[SIZEI];
    TYPE *phy_d[SIZEJ];
    TYPE *php_d[SIZEP];

    phx_d[0] = x1_d;
    for(int i=1; i<SIZEI; i++)
        phx_d[i] = args[INDSI::VAL(i-1)];
    for(int i=0; i<SIZEJ; i++)
        phy_d[i] = args[INDSJ::VAL(i)];
    for(int i=0; i<SIZEP; i++)
        php_d[i] = args[INDSP::VAL(i)];

    return Eval_(fun,nx,ny,phx_d,phy_d,php_d);

}

};

}
