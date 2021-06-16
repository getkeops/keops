
#include <vector>
#include <string>

#include "includes.h"
#include "Ranges.h"
#include "Sizes.h"



void launch_keops(int deviceId,
						 int nx,
						 int ny,
						 __TYPE__ *result_ptr,
                         int nargs,
                         __TYPE__ **args_ptr,
						 int **argshapes_ptr,
                         int nranges = 0,
                         index_t* ranges = NULL) {
							 
  check_nargs(nargs, keops_nminargs);

  Sizes SS(nargs, args_ptr, argshapes_ptr, nx, ny);

#if USE_HALF
  SS.switch_to_half2_indexing();
#endif

  Ranges RR(SS, nranges, ranges);


  // Create a decimal word to avoid nested conditional below
  int decision = 1000 * RR.tagRanges + 100 * tagHostDevice + 10 * tagCpuGpu + tag1D2D;

  switch (decision) {

#if !USE_HALF
    case 0: {
      //CpuReduc(SS.nx, SS.ny, result_ptr, args_ptr);
    }
#endif
    
    case 10: {
#if USE_CUDA
      //GpuReduc1D_FromHost(SS.nx, SS.ny, result_ptr, args_ptr, deviceId);
#else
      keops_error(Error_msg_no_cuda);
#endif
    }
    
    case 11: {
#if USE_CUDA
      //GpuReduc2D_FromHost(SS.nx, SS.ny, result_ptr, args_ptr, deviceId);
#else
      keops_error(Error_msg_no_cuda);
#endif
    }
    
    case 110: {
#if USE_CUDA
      //GpuReduc1D_FromDevice(SS.nx, SS.ny, result_ptr, args_ptr, deviceId);
#else
      keops_error(Error_msg_no_cuda);
#endif
    }
    
    case 111: {
#if USE_CUDA
      //GpuReduc2D_FromDevice(SS.nx, SS.ny, result_ptr, args_ptr, deviceId);
#else
      keops_error(Error_msg_no_cuda);
#endif
    }

#if !USE_HALF    
    case 1000: {
      //CpuReduc_ranges(SS.nx, SS.ny, SS.nbatchdims, SS.shapes,
      //                RR.nranges_x, RR.nranges_y, RR.castedranges,
      //                result_ptr, args_ptr.data());
    }
#endif
    
    case 1010: {
#if USE_CUDA
      //GpuReduc1D_ranges_FromHost(SS.nx, SS.ny, SS.nbatchdims, SS.shapes,
      //                           RR.nranges_x, RR.nranges_y, RR.nredranges_x, RR.nredranges_y, RR.castedranges,
      //                           result_ptr, args_ptr.data(), deviceId);
#else
      keops_error(Error_msg_no_cuda);
#endif
    }
    
    case 1110: {
#if USE_CUDA
      //GpuReduc1D_ranges_FromDevice(SS.nx, SS.ny, SS.nbatchdims, SS.shapes,
      //                             RR.nranges_x, RR.nranges_y, RR.castedranges,
      //                             result_ptr, args_ptr.data(), deviceId);
#else
      keops_error(Error_msg_no_cuda);
#endif
    }
    
    default: {
      keops_error("[KeOps] Inconsistent values for tagHostDevice, tagCpuGpu, tagRanges, tag1D2D...");
    }
  }
  
}

