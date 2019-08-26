


#ifdef __CUDACC__
  // fix some Gpu properties
  // CUDA_BLOCK_SIZE gives an upper bound on size of the size of Cuda blocks
  // The actual block size may be lower due to memory limitations, depending on the formula used
  #ifndef CUDA_BLOCK_SIZE
    #define CUDA_BLOCK_SIZE 192 
  #endif
  // Here we define the maximum number of threads per block and the shared memory per block
  // These values can depend on the Gpu, although in fact values 1024 and 49152 respectively
  // are the good values for almost all cards. 
  // So these values should be fine, but you can check them with GetGpuProps.cu program
  // Here we assume that: either the user has defined MAXIDGPU (=number of Gpu devices minus one)
  // and corresponding specific values MAXTHREADSPERBLOCK0, SHAREDMEMPERBLOCK0, MAXTHREADSPERBLOCK1, SHAREDMEMPERBLOCK1, ...
  // for each device, or MAXIDGPU is not defined, and we will use global MAXTHREADSPERBLOCK and SHAREDMEMPERBLOCK
  #ifndef MAXIDGPU
    // we give default values
    #ifndef MAXTHREADSPERBLOCK
      #define MAXTHREADSPERBLOCK 1024 
    #endif
    #ifndef SHAREDMEMPERBLOCK
      #define SHAREDMEMPERBLOCK 49152 
    #endif
  #endif 
#endif

#ifndef __TYPE__
  #define __TYPE__ float
#endif

#ifndef C_CONTIGUOUS
  #define C_CONTIGUOUS 0
#endif

#ifndef MAX_UNROLL_COUNT
  #define MAX_UNROLL_COUNT 1048576
#endif

#include "core/formulas/constants.h"
#include "core/formulas/maths.h"
#include "core/formulas/kernels.h"
#include "core/formulas/norms.h"
#include "core/formulas/factorize.h"
#include "core/formulas/newsyntax.h"

#include "core/CpuConv.cpp"
#ifdef __CUDACC__
	#include "core/GpuConv1D.cu"
	#include "core/GpuConv2D.cu"
#endif
#include "core/reductions/sum.h"
#include "core/reductions/min.h"
#include "core/reductions/kmin.h"
#include "core/reductions/max_sumshiftexp.h"
#include "core/reductions/max.h"
#include "core/reductions/zero.h"
