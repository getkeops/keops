


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
  #define USE_DOUBLE 0
#endif

#ifndef C_CONTIGUOUS
  #define C_CONTIGUOUS 0
#endif

#include "core/formulas/constants.h"


// import all math implementations
#include "core/formulas/maths/Minus.h"
#include "core/formulas/maths/Sum.h"
#include "core/formulas/maths/Add.h"
#include "core/formulas/maths/Concat.h"
#include "core/formulas/maths/Scal.h"
#include "core/formulas/maths/Mult.h"
#include "core/formulas/maths/ScalOrMult.h"
#include "core/formulas/maths/Subtract.h"
#include "core/formulas/maths/Exp.h"
#include "core/formulas/maths/Sin.h"
#include "core/formulas/maths/Cos.h"
#include "core/formulas/maths/Pow.h"
#include "core/formulas/maths/Square.h"
#include "core/formulas/maths/Inv.h"
#include "core/formulas/maths/IntInv.h"
#include "core/formulas/maths/Divide.h"
#include "core/formulas/maths/Log.h"
#include "core/formulas/maths/Sign.h"
#include "core/formulas/maths/Abs.h"
#include "core/formulas/maths/Step.h"
#include "core/formulas/maths/ReLu.h"
#include "core/formulas/maths/Powf.h"
#include "core/formulas/maths/Sqrt.h"
#include "core/formulas/maths/Rsqrt.h"
#include "core/formulas/maths/MatVecMult.h"
#include "core/formulas/maths/GradMatrix.h"
#include "core/formulas/maths/TensorDot.h"
#include "core/formulas/maths/TensorProd.h"
#include "core/formulas/maths/VecMatMult.h"


// import all operations on vector implementations
#include "core/formulas/norms/Norm2.h"
#include "core/formulas/norms/Normalize.h"
#include "core/formulas/norms/Scalprod.h"
#include "core/formulas/norms/SqDist.h"
#include "core/formulas/norms/SqNorm2.h"
#include "core/formulas/norms/SqNormDiag.h"
#include "core/formulas/norms/SqNormIso.h"
#include "core/formulas/norms/WeightedSqDist.h"
#include "core/formulas/norms/WeightedSqNorm.h"

// import all reductions
#include "core/formulas/reductions/Sum_Reduction.h"
#include "core/formulas/reductions/max_sumshiftexp.h"
#include "core/formulas/reductions/min.h"
#include "core/formulas/reductions/max.h"
#include "core/formulas/reductions/kmin.h"

// import all Kernels
#include "core/formulas/kernels/CauchyKernel.h"
#include "core/formulas/kernels/CurlFreeGaussKernel.h"
#include "core/formulas/kernels/DivFreeGaussKernel.h"
#include "core/formulas/kernels/GaussKernel.h"
#include "core/formulas/kernels/InverseMultiquadricKernel.h"
#include "core/formulas/kernels/LaplaceKernel.h"
#include "core/formulas/kernels/ScalarRadialKernels.h"
#include "core/formulas/kernels/SumGaussKernel.h"
#include "core/formulas/kernels/TRI_Kernel.h"
#include "core/formulas/kernels/TRIGaussKernel.h"


#include "core/formulas/factorize.h"
#include "core/formulas/newsyntax.h"

#include "core/CpuConv.cpp"
#ifdef __CUDACC__
	#include "core/GpuConv1D.cu"
	#include "core/GpuConv2D.cu"
#endif


#include "core/formulas/utils.h"
