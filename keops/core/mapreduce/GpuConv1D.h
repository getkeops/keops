#pragma once

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <cuda.h>

#include "core/pack/Pack.h"
#include "core/pack/GetInds.h"
#include "core/pack/GetDims.h"


namespace keops {

template< typename TYPE, class FUN >
__global__ void GpuConv1DOnDevice(FUN fun, int nx, int ny, TYPE** px, TYPE** py, TYPE** pp) {
  
  // get the index of the current thread
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  // declare shared mem
  extern __shared__ TYPE yj[];
  
  // get templated dimensions :
  typedef typename FUN::DIMSX DIMSX;  // DIMSX is a "vector" of templates giving dimensions of xi variables
  typedef typename FUN::DIMSY DIMSY;  // DIMSY is a "vector" of templates giving dimensions of yj variables
  typedef typename FUN::DIMSP DIMSP;  // DIMSP is a "vector" of templates giving dimensions of parameters variables
  const int DIMX = DIMSX::SUM;        // DIMX  is sum of dimensions for xi variables
  const int DIMY = DIMSY::SUM;        // DIMY  is sum of dimensions for yj variables
  const int DIMP = DIMSP::SUM;        // DIMP  is sum of dimensions for parameters variables
  const int DIMOUT = FUN::DIM; // dimension of output variable
  const int DIMRED = FUN::DIMRED; // dimension of reduction operation
  const int DIMFOUT = DIMSX::FIRST;     // DIMFOUT is dimension of output variable of inner function
  
  // load parameter(s)
  TYPE param_loc[DIMP < 1 ? 1 : DIMP];
  load< DIMSP >(0, param_loc, pp); // load parameters variables from global memory to local thread memory
  
  // get the value of variable (index with i)
  TYPE xi[DIMX < 1 ? 1 : DIMX];
  __TYPEACC__ acc[DIMRED];
#if USE_BLOCKRED
  // additional tmp vector to store intermediate results from each block
  TYPE tmp[DIMRED];
#elif USE_KAHAN
  // additional tmp vector to accumulate errors
  const int DIM_KAHAN = FUN::template KahanScheme<__TYPEACC__,TYPE>::DIMACC;
  TYPE tmp[DIM_KAHAN];
#endif
  if (i < nx) {
    typename FUN::template InitializeReduction< __TYPEACC__ >()(acc); // acc = 0
#if USE_KAHAN
#pragma unroll
    for (int k = 0; k < DIM_KAHAN; k++)
      tmp[k] = 0.0f;
#endif
    load< typename DIMSX::NEXT >(i, xi + DIMFOUT,
                                 px + 1); // load xi variables from global memory to local thread memory
  }
  
  for (int jstart = 0, tile = 0; jstart < ny; jstart += blockDim.x, tile++) {
    
    // get the current column
    int j = tile * blockDim.x + threadIdx.x;
    
    if (j < ny) { // we load yj from device global memory only if j<ny
      load< DIMSY >(j, yj + threadIdx.x * DIMY, py); // load yj variables from global memory to shared memory
    }
    __syncthreads();
    
    if (i < nx) { // we compute x1i only if needed
      TYPE * yjrel = yj; // Loop on the columns of the current block.
#if USE_BLOCKRED
      typename FUN::template InitializeReduction<TYPE>()(tmp); // tmp = 0
#endif
      for (int jrel = 0; (jrel < blockDim.x) && (jrel < ny - jstart); jrel++, yjrel += DIMY) {
        call< DIMSX, DIMSY, DIMSP >(fun,
                                    xi,
                                    yjrel,
                                    param_loc); // Call the function, which outputs results in xi[0:DIMX1]
#if USE_BLOCKRED
        typename FUN::template ReducePairShort<TYPE,TYPE>()(tmp, xi, jrel + tile * blockDim.x);     // tmp += xi
#elif USE_KAHAN
        typename FUN::template KahanScheme<__TYPEACC__,TYPE>()(acc, xi, tmp);
#else
        typename FUN::template ReducePairShort< __TYPEACC__, TYPE >()(acc, xi, jrel + tile * blockDim.x);     // acc += xi
#endif
      }
#if USE_BLOCKRED
      typename FUN::template ReducePair<__TYPEACC__,TYPE>()(acc, tmp);     // acc += tmp
#endif
    }
    __syncthreads();
  }
  if (i < nx) {
    typename FUN::template FinalizeOutput< __TYPEACC__, TYPE >()(acc, px[0] + i * DIMOUT, px, i);
  }
  
}

}
