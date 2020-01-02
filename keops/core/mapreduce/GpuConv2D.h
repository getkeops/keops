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

template <typename TYPE, int DIMIN, int DIMOUT, class FUN>
__global__ void reduce2D(TYPE *in, TYPE *out, TYPE ** px, int sizeY,int nx) {
    /* Function used as a final reduction pass in the 2D scheme,
     * once the block reductions have been made.
     * Takes as input:
     * - in,  a  sizeY * (nx * DIMIN ) array
     * - out, an          nx * DIMOUT   array
     * also px array of pointers to input device data is needed, used for some reductions
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
    typename FUN::template InitializeReduction<__TYPEACC__>()(acc); // acc = 0
    if(tid < nx) {
        for (int y = 0; y < sizeY; y++)
            typename FUN::template ReducePair<__TYPEACC__,TYPE>()(acc, in + (tid+y*nx)*DIMIN);     // acc += in[(tid+y*nx) *DIMVECT : +DIMVECT];
        typename FUN::template FinalizeOutput<__TYPEACC__,TYPE>()(acc, out+tid*DIMOUT, px, tid);
    }

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
#if USE_BLOCKRED 
// N.B. To be consistent with the convention used in GpuConv1D, when USE_BLOCKRED=1 we accumulate results in TYPE 
// instead of __TYPEACC__ in each block, __TYPEACC__ will be used only to sum up results from each block
    TYPE acc[DIMRED];
#else
    __TYPEACC__ acc[DIMRED];
#endif
#if USE_KAHAN
    const int DIM_KAHAN = FUN::template KahanScheme<__TYPEACC__,TYPE>::DIMACC;
    TYPE tmp[DIM_KAHAN];
#endif
    if(i<nx) { // we will compute x1i only if i is in the range
#if USE_BLOCKRED         
        typename FUN::template InitializeReduction<TYPE>()(acc); // acc = 0
#else
        typename FUN::template InitializeReduction<__TYPEACC__>()(acc); // acc = 0
#endif
#if USE_KAHAN
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
#if USE_BLOCKRED
            typename FUN::template ReducePairShort<TYPE,TYPE>()(acc, xi, blockDim.x*blockIdx.y+jrel);     // acc += xi
#elif USE_KAHAN
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
    if(i<nx)
        for(int k=0; k<DIMRED; k++)
            (*px)[blockIdx.y*DIMRED*nx+i*DIMRED+k] = acc[k];
}
///////////////////////////////////////////////////

}
