
// special computation scheme for dim>100
#ifndef ENABLECHUNK
  #define ENABLECHUNK 1
#endif
#ifndef DIMCHUNK
  #define DIMCHUNK 64
#endif
#ifndef DIM_TRESHOLD_CHUNK
  #define DIM_TRESHOLD_CHUNK 143
#endif
#ifndef SPECDIM_USE_CHUNK1
  #define SPECDIM_USE_CHUNK1 -1 // originally 80 but deactivated for release 1.4.2
#endif
#ifndef SPECDIM_USE_CHUNK2
  #define SPECDIM_USE_CHUNK2 109
#endif
#ifndef SPECDIM_USE_CHUNK3
  #define SPECDIM_USE_CHUNK3 112
#endif
#ifndef SPECDIM_USE_CHUNK4
  #define SPECDIM_USE_CHUNK4 114
#endif

// special mode for formula of the type sum_j k(x_i,y_j)*b_j with high dimensional b_j
#ifndef ENABLE_FINAL_CHUNKS
  #define ENABLE_FINAL_CHUNKS 1
#endif
#ifndef DIMFINALCHUNK
  #define DIMFINALCHUNK 64
#endif
#ifndef MULT_VAR_HIGHDIM
  #define MULT_VAR_HIGHDIM 0
#endif
#if ENABLE_FINAL_CHUNKS==1 && MULT_VAR_HIGHDIM==1
	#define USE_FINAL_CHUNKS 1
#else
	#define USE_FINAL_CHUNKS 0
#endif

#if USE_HALF
  #include <cuda_fp16.h>
#endif

// import constant
#include "core/formulas/constants/IntConst.h"
#include "core/formulas/constants/Zero.h"


// import all math implementations
#include "core/formulas/maths/Minus.h"
#include "core/formulas/maths/Sum.h"
#include "core/formulas/maths/Min.h"
#include "core/formulas/maths/Max.h"
#include "core/formulas/maths/ArgMin.h"
#include "core/formulas/maths/ArgMax.h"
#include "core/formulas/maths/Add.h"
#include "core/formulas/maths/Concat.h"
#include "core/formulas/maths/Scal.h"
#include "core/formulas/maths/Mult.h"
#include "core/formulas/maths/ScalOrMult.h"
#include "core/formulas/maths/Subtract.h"
#include "core/formulas/maths/Exp.h"
#include "core/formulas/maths/Sin.h"
#include "core/formulas/maths/Asin.h"
#include "core/formulas/maths/Cos.h"
#include "core/formulas/maths/Acos.h"
#include "core/formulas/maths/Pow.h"
#include "core/formulas/maths/Square.h"
#include "core/formulas/maths/Inv.h"
#include "core/formulas/maths/IntInv.h"
#include "core/formulas/maths/Divide.h"
#include "core/formulas/maths/Log.h"
#include "core/formulas/maths/XLogX.h"
#include "core/formulas/maths/Sign.h"
#include "core/formulas/maths/Abs.h"
#include "core/formulas/maths/Step.h"
#include "core/formulas/maths/ReLu.h"
#include "core/formulas/maths/Clamp.h"
#include "core/formulas/maths/ClampInt.h"
#include "core/formulas/maths/Powf.h"
#include "core/formulas/maths/Sqrt.h"
#include "core/formulas/maths/Rsqrt.h"
#include "core/formulas/maths/Atan.h"
#include "core/formulas/maths/MatVecMult.h"
#include "core/formulas/maths/GradMatrix.h"
#if ((__CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 100 + __CUDACC_VER_BUILD__) >= 11100)
    #include "core/formulas/maths/TensorDotNoTao.h"
#else
    #include "core/formulas/maths/TensorDot.h"
#endif
#include "core/formulas/maths/TensorProd.h"
#include "core/formulas/maths/VecMatMult.h"
#include "core/formulas/maths/OneHot.h"


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
#include "core/reductions/Sum_Reduction.h"
#include "core/reductions/Max_SumShiftExp_Reduction.h"
#include "core/reductions/Min_Reduction.h"
#include "core/reductions/Max_Reduction.h"
#include "core/reductions/KMin_Reduction.h"


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

#include "core/formulas/Factorize.h"
#include "core/formulas/PrintFormula.h"

// special options for accuracy
#ifndef __TYPEACC__
  #define __TYPEACC__ __TYPE__
#endif
#define DIRECT_SUM 0
#define BLOCK_SUM 1
#define KAHAN_SCHEME 2
#ifndef SUM_SCHEME
  #define SUM_SCHEME DIRECT_SUM
#endif

// float16 support
#if !USE_HALF
#include "core/mapreduce/CpuConv.cpp"
#endif

#ifdef __CUDACC__
  #include <cuda_fp16.h>
  #include "core/mapreduce/GpuConv1D.cu"
  #include "core/mapreduce/GpuConv2D.cu"
#endif

