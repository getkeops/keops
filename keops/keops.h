


#ifdef __CUDACC__
	// fix some Gpu properties
	// These values should be fine, but you can check them with GetGpuProps.cu program
	#ifndef MAXIDGPU
	  #define MAXIDGPU 0 // (= number of Gpu devices - 1)
	  #define CUDA_BLOCK_SIZE 192
	  #define MAXTHREADSPERBLOCK0 1024 
	  #define SHAREDMEMPERBLOCK0 49152
	#endif 
#endif

#ifndef __TYPE__
  #define __TYPE__ float
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