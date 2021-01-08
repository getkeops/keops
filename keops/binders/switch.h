#pragma once

#include "binders/keops_cst.h"
#include "binders/utils.h"
#include "binders/checks.h"

extern "C" {
int GetFormulaConstants(int*);
int GetIndsI(int*);
int GetIndsJ(int*);
int GetIndsP(int*);
int GetDimsX(int*);
int GetDimsY(int*);
int GetDimsP(int*);
};

#if !USE_HALF
extern "C" {
int CpuReduc(int, int, __TYPE__*, __TYPE__**);
int CpuReduc_ranges(int, int, int, int*, int, int, __INDEX__**, __TYPE__*, __TYPE__**);
};
#endif

#if USE_CUDA
extern "C" {
    int GpuReduc1D_FromHost(int, int, __TYPE__*, __TYPE__**, int);
    int GpuReduc1D_FromDevice(int, int, __TYPE__*, __TYPE__**, int);
    int GpuReduc2D_FromHost(int, int, __TYPE__*, __TYPE__**, int);
    int GpuReduc2D_FromDevice(int, int, __TYPE__*, __TYPE__**, int);
    int GpuReduc1D_ranges_FromHost(int, int, int, int*, int, int, int, int, __INDEX__**, __TYPE__*, __TYPE__**, int);
    int GpuReduc1D_ranges_FromDevice(int, int, int, int*, int, int, __INDEX__**, __TYPE__*, __TYPE__**, int);
};
#endif

namespace keops_binders {


template< typename array_t, typename index_t >
class Ranges {
public:
  int tagRanges, nranges_x, nranges_y, nredranges_x, nredranges_y;
  
  std::vector< __INDEX__* > _castedranges;
  std::vector< __INDEX__ > ranges_i, slices_i, redranges_j;
  __INDEX__** castedranges;
  
  Ranges(Sizes< array_t > sizes, int nranges, index_t* ranges) {
    
    if ((nranges != 0) && (nranges != 6))
      throw std::runtime_error(
              "[KeOps] the 'ranges' argument should be a tuple of size 0 or 6, "
              "but is of size " + std::to_string(nranges) + "."
      );
    
    // Sparsity: should we handle ranges? ======================================
    if (sizes.nbatchdims == 0) {  // Standard M-by-N computation
      if (nranges == 0) {
        
        tagRanges = 0;
        
        nranges_x = 0;
        nranges_y = 0;
        
        nredranges_x = 0;
        nredranges_y = 0;
        
      } else if (nranges == 6) {
  
        tagRanges = 1;
        nranges_x = get_size(ranges[0], 0);
        nranges_y = get_size(ranges[3], 0);
        
        nredranges_x = get_size(ranges[5], 0);
        nredranges_y = get_size(ranges[2], 0);
        
        // get the pointers to data to avoid a copy
        _castedranges.resize(nranges);
        for (int i = 0; i < nranges; i++)
          _castedranges[i] = get_rangedata< index_t >(ranges[i]);
        
        castedranges = &_castedranges[0];
      }
      
    } else if (nranges == 0) {
      // Batch processing: we'll have to generate a custom, block-diagonal sparsity pattern
      tagRanges = 1;  // Batch processing is emulated through the block-sparse mode
      
      // Create new "castedranges" from scratch ------------------------------
      // With pythonic notations, we'll have:
      //   castedranges = (ranges_i, slices_i, redranges_j,   ranges_j, slices_j, redranges_i)
      // with:
      // - ranges_i    = redranges_i = [ [0,M], [M,2M], ..., [(nbatches-1)M, nbatches*M] ]
      // - slices_i    = slices_j    = [    1,     2,   ...,   nbatches-1,   nbatches    ]
      // - redranges_j = ranges_j    = [ [0,N], [N,2N], ..., [(nbatches-1)N, nbatches*N] ]
      
      //__INDEX__* castedranges[6];
      _castedranges.resize(6);
      
      //__INDEX__ ranges_i[2 * sizes.nbatches];  // ranges_i
      ranges_i.resize(2 * sizes.nbatches, 0);
      
      //__INDEX__ slices_i[sizes.nbatches];    // slices_i
      slices_i.resize(sizes.nbatches, 0);
      
      //__INDEX__ redranges_j[2 * sizes.nbatches];  // redranges_j
      redranges_j.resize(2 * sizes.nbatches, 0);
      
      for (int b = 0; b < sizes.nbatches; b++) {
        ranges_i[2 * b] = b * sizes.M;
        ranges_i[2 * b + 1] = (b + 1) * sizes.M;
        slices_i[b] = (b + 1);
        redranges_j[2 * b] = b * sizes.N;
        redranges_j[2 * b + 1] = (b + 1) * sizes.N;
      }
  
      _castedranges[0] = &ranges_i[0];
      _castedranges[1] = &slices_i[0];
      _castedranges[2] = &redranges_j[0];
      _castedranges[3] = &redranges_j[0];            // ranges_j
      _castedranges[4] = &slices_i[0];            // slices_j
      _castedranges[5] = &ranges_i[0];            // redranges_i
 
      
      nranges_x = sizes.nbatches;
      nredranges_x = sizes.nbatches;
      nranges_y = sizes.nbatches;
      nredranges_y = sizes.nbatches;
      castedranges = &_castedranges[0];
  
    } else {
      throw std::runtime_error(
              "[KeOps] The 'ranges' argument (block-sparse mode) is not supported with batch processing, "
              "but we detected " + std::to_string(sizes.nbatchdims) + " > 0 batch dimensions."
      );
    }
  

    
  };
  
};


template< typename array_t, typename array_t_out = array_t, typename index_t = __INDEX__ >
array_t_out launch_keops(int tag1D2D,
                         int tagCpuGpu,
                         int tagHostDevice,
                         int deviceId,
						 int nx,
						 int ny,
                         int nargs,
                         array_t* args,
                         int nranges = 0,
                         index_t* ranges = {}) {
							 
  keops_binders::check_tag(tag1D2D, "1D2D");
  keops_binders::check_tag(tagCpuGpu, "CpuGpu");
  keops_binders::check_tag(tagHostDevice, "HostDevice");
  
  keops_binders::check_nargs(nargs, keops_nminargs);
  short int deviceId_casted = cast_Device_Id(deviceId);

  Sizes< array_t > SS(nargs, args, nx, ny);
  
  array_t_out result = (tagHostDevice == 0) ? allocate_result_array< array_t_out, __TYPE__ >(SS.shape_out, SS.nbatchdims)
                                            : allocate_result_array_gpu< array_t_out, __TYPE__ >(SS.shape_out, SS.nbatchdims, deviceId_casted);

  __TYPE__* result_ptr = get_data< array_t_out, __TYPE__ >(result);

#if USE_HALF
  SS.switch_to_half2_indexing();
#endif

  Ranges< array_t, index_t > RR(SS, nranges, ranges);

  // get the pointers to data to avoid a copy
  std::vector<__TYPE__*> args_ptr(nargs);
  for (int i = 0; i < nargs; i++)
    args_ptr[i] = get_data< array_t, __TYPE__ >(args[i]);

  // Create a decimal word to avoid nested conditional below
  int decision = 1000 * RR.tagRanges + 100 * tagHostDevice + 10 * tagCpuGpu + tag1D2D;

  switch (decision) {

#if !USE_HALF
    case 0: {
      CpuReduc(SS.nx, SS.ny, result_ptr, args_ptr.data());
      return result;
    }
#endif
    
    case 10: {
#if USE_CUDA
      GpuReduc1D_FromHost(SS.nx, SS.ny, result_ptr, args_ptr.data(), deviceId_casted);
      return result;
#else
      keops_error(Error_msg_no_cuda);
#endif
    }
    
    case 11: {
#if USE_CUDA
      GpuReduc2D_FromHost(SS.nx, SS.ny, result_ptr, args_ptr.data(), deviceId_casted);
      return result;
#else
      keops_error(Error_msg_no_cuda);
#endif
    }
    
    case 110: {
#if USE_CUDA
      GpuReduc1D_FromDevice(SS.nx, SS.ny, result_ptr, args_ptr.data(), deviceId_casted);
      return result;
#else
      keops_error(Error_msg_no_cuda);
#endif
    }
    
    case 111: {
#if USE_CUDA
      GpuReduc2D_FromDevice(SS.nx, SS.ny, result_ptr, args_ptr.data(), deviceId_casted);
      return result;
#else
      keops_error(Error_msg_no_cuda);
#endif
    }

#if !USE_HALF    
    case 1000: {
      CpuReduc_ranges(SS.nx, SS.ny, SS.nbatchdims, SS.shapes,
                      RR.nranges_x, RR.nranges_y, RR.castedranges,
                      result_ptr, args_ptr.data());
      return result;
    }
#endif
    
    case 1010: {
#if USE_CUDA
      GpuReduc1D_ranges_FromHost(SS.nx, SS.ny, SS.nbatchdims, SS.shapes,
                                 RR.nranges_x, RR.nranges_y, RR.nredranges_x, RR.nredranges_y, RR.castedranges,
                                 result_ptr, args_ptr.data(), deviceId_casted);
      return result;
#else
      keops_error(Error_msg_no_cuda);
#endif
    }
    
    case 1110: {
#if USE_CUDA
      GpuReduc1D_ranges_FromDevice(SS.nx, SS.ny, SS.nbatchdims, SS.shapes,
                                   RR.nranges_x, RR.nranges_y, RR.castedranges,
                                   result_ptr, args_ptr.data(), deviceId_casted);
      return result;
#else
      keops_error(Error_msg_no_cuda);
#endif
    }
    
    default: {
      keops_error("[KeOps] Inconsistent values for tagHostDevice, tagCpuGpu, tagRanges, tag1D2D...");
      throw std::runtime_error("A dummy error to avoid return-type warning");
    }
  }
  
}

}
