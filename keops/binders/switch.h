#pragma once

#include <algorithm>

#include "binders/utils.h"
#include "binders/keops_cst.h"
#include "binders/checks.h"


extern "C" {
int CpuReduc(int, int, __TYPE__*, __TYPE__**);
int CpuReduc_ranges(int, int, int, int*, int, int, __INDEX__**, __TYPE__*, __TYPE__**);
};

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

template< typename array_t >
class Sizes {
public:
  // constructors
  Sizes(int _nargs, array_t* args) {
    
    nargs = _nargs;
    
    // fill shapes wit "batch dimensions" [A, .., B], the table will look like:
    //
    // [ A, .., B, M, N, D_out]  -> output
    // [ A, .., B, M, 1, D_1  ]  -> "i" variable
    // [ A, .., B, 1, N, D_2  ]  -> "j" variable
    // [ A, .., B, 1, 1, D_3  ]  -> "parameter"
    // [ A, .., 1, M, 1, D_4  ]  -> N.B.: we support broadcasting on the batch dimensions!
    // [ 1, .., 1, M, 1, D_5  ]  ->      (we'll just ask users to fill in the shapes with *explicit* ones)
    fill_shape(_nargs, args);
    
    check_ranges(_nargs, args);
    shapes = &_shapes[0];
    
    // fill shape_out
    _shape_out.resize(nbatchdims + 3);
    #if C_CONTIGUOUS
    std::copy(_shapes.begin(), _shapes.begin() + nbatchdims + 3, _shape_out.begin());// Copy the "batch dimensions"
    _shape_out.erase(_shape_out.begin() + nbatchdims + (1 - keops::TAGIJ));
    #else
    std::reverse_copy(_shapes.begin(), _shapes.begin() + nbatchdims + 3,
                      _shape_out.begin());// Copy the "batch dimensions"
    _shape_out.erase(_shape_out.begin() + 1 + keops::TAGIJ);
    #endif
    
    shape_out = &_shape_out[0];
    
    /*
    printf("nbatchdim %d\n", _shape_out.size());
    printf("nbatchdim %d\n", (keops::NARGS + 1) * (nbatchdims + 3));
    for (auto &i : _shapes) { std::cout << i << " ";}
    std::cout << '\n';
    for (auto &i : _shape_out) { std::cout << i << " ";}
    std::cout << '\n';
    printf("%d\n", keops::TAGIJ);
    
    _shape_out[0] = 3;
    _shape_out[1] = 5000;
  
    for (int b = 0; b < nbatchdims; b++)
      _shape_out[b] = shapes[b];
    
    _shape_out[nbatchdims] = shapes[MN_pos + keops::TAGIJ];  // M or N
    _shape_out[nbatchdims + 1] = shapes[nbatchdims + 2];         // D
    */
    
    
    // fill nx and ny
    M = _shapes[nbatchdims];      // = M
    N = _shapes[nbatchdims + 1];  // = N
    
    // Compute the product of all "batch dimensions"
    nbatches = 1;
    for (int b = 0; b < nbatchdims; b++)
      nbatches *= shapes[b];
    //int nbatches = std::accumulate(shapes, shapes + nbatchdims, 1, std::multiplies< int >());
    
    nx = nbatches * M;  // = A * ... * B * M
    ny = nbatches * N;  // = A * ... * B * N
  }
  
  // attributs
  int nargs;
  int nx, ny;
  int M, N;
  int nbatchdims;
  int nbatches;
  
  std::vector< int > _shapes;
  int* shapes;
  std::vector< int > _shape_out;
  int* shape_out;
  
  // methods
private:
  void fill_shape(int nargs, array_t* args);
  
  void check_ranges(int nargs, array_t* args);
  
  std::function< int(array_t, int, int) > get_size_batch;
  int MN_pos, D_pos;
};


template< typename array_t >
void Sizes< array_t >::fill_shape(int nargs, array_t* args) {
  
  
  if (keops::NARGS > 0) {
    // Are we working in batch mode? Infer the answer from the first arg =============
    nbatchdims = get_ndim(args[0]);  // Number of dims of the first tensor
    
    // Remove the "trailing" dim (.., D) if the first arg is a parameter,
    // or the last two (.., M/N, D) if it is an "i" or "j" variable:
    static const int trailing_dim = (keops::TYPE_FIRST_ARG == 2) ? 1 : 2;
    nbatchdims -= trailing_dim;
    if (nbatchdims < 0) {
      keops_error("[KeOps] Wrong number of dimensions for arg at position 0: is "
                  + std::to_string(get_ndim(args[0])) + " but should be at least "
                  + std::to_string(trailing_dim) + "."
      );
    }
  } else {
    nbatchdims = 0;
  }
  
  #if C_CONTIGUOUS
  get_size_batch = [](auto args, int nbatch, int b) {
    return get_size(args, b);
  };
  MN_pos = nbatchdims;
  D_pos = nbatchdims + 1;
  #else
  D_pos = 0;
  MN_pos = 1;
  get_size_batch = [](auto obj_ptr, int nbatch, int b) {
    return get_size(obj_ptr, nbatch - b);
  };
  #endif
  
  // Now, we'll keep track of the output + all arguments' shapes in a large array:
  _shapes.resize((keops::NARGS + 1) * (nbatchdims + 3), 1);
  
  if (keops::POS_FIRST_ARGI > -1)
    _shapes[nbatchdims] = get_size(args[keops::POS_FIRST_ARGI], MN_pos);
  
  if (keops::POS_FIRST_ARGJ > -1)
    _shapes[nbatchdims + 1] = get_size(args[keops::POS_FIRST_ARGJ], MN_pos);
  
  _shapes[nbatchdims + 2] = keops::DIMOUT;   // Top right corner: dimension of the output
  
}


template< typename array_t, typename index_t >
class Ranges {
public:
  int tagRanges, nranges_x, nranges_y, nredranges_x, nredranges_y;
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
        
        //__INDEX__* tmp[1];
        //castedranges = tmp;
        
      } else if (nranges == 6) {
        
        tagRanges = 1;
        nranges_x = get_size(ranges[0], 0);
        nranges_y = get_size(ranges[3], 0);
        
        nredranges_x = get_size(ranges[5], 0);
        nredranges_y = get_size(ranges[2], 0);
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
      
      __INDEX__* castedranges[6];
      
      __INDEX__ ranges_i[2 * sizes.nbatches];  // ranges_i
      castedranges[0] = ranges_i;
      __INDEX__ slices_i[sizes.nbatches];    // slices_i
      castedranges[1] = slices_i;
      __INDEX__ redranges_j[2 * sizes.nbatches];  // redranges_j
      castedranges[2] = redranges_j;
      
      castedranges[3] = castedranges[2];            // ranges_j
      castedranges[4] = castedranges[1];            // slices_j
      castedranges[5] = castedranges[0];            // redranges_i
      
      for (int b = 0; b < sizes.nbatches; b++) {
        castedranges[0][2 * b] = b * sizes.M;
        castedranges[0][2 * b + 1] = (b + 1) * sizes.M;
        castedranges[1][b] = (b + 1);
        castedranges[2][2 * b] = b * sizes.N;
        castedranges[2][2 * b + 1] = (b + 1) * sizes.N;
      }
      
      nranges_x = sizes.nbatches;
      nredranges_x = sizes.nbatches;
      nranges_y = sizes.nbatches;
      nredranges_y = sizes.nbatches;
      
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
                         int nargs,
                         array_t* args,
                         int nranges = 0,
                         index_t* ranges = {}) {
  
  keops_binders::check_tag(tag1D2D, "1D2D");
  keops_binders::check_tag(tagCpuGpu, "CpuGpu");
  keops_binders::check_tag(tagHostDevice, "HostDevice");
  
  short int Device_Id = cast_Device_Id(deviceId);
  
  Sizes< array_t > SS(nargs, args);
  Ranges< array_t, index_t > RR(SS, nranges, ranges);
  
  array_t_out result = (tagHostDevice == 0) ? allocate_result_array< array_t_out, __TYPE__ >(SS.shape_out)
                                            : allocate_result_array_gpu< array_t_out, __TYPE__ >(SS.shape_out);
  __TYPE__* result_ptr = get_data< array_t_out, __TYPE__ >(result);
  
  // get the pointers to data to avoid a copy
  __TYPE__* args_ptr[keops::NARGS];
  for (int i = 0; i < keops::NARGS; i++)
    args_ptr[i] = get_data< array_t, __TYPE__ >(args[i]);
  
  // get the pointers to data to avoid a copy
  __INDEX__* ranges_ptr[nranges];
  for (int i = 0; i < nranges; i++)
    ranges_ptr[i] = get_rangedata< index_t >(ranges[i]);
  
  // Create a decimal word to avoid nested conditional below
  int decision = 1000 * RR.tagRanges + 100 * tagHostDevice + 10 * tagCpuGpu + tag1D2D;
  
  switch (decision) {
    case 0: {
      CpuReduc(SS.nx, SS.ny, result_ptr, args_ptr);
      return result;
    }
  
    case 10: {
#if USE_CUDA
      GpuReduc1D_FromHost(SS.nx, SS.ny, result_ptr, args_ptr, Device_Id);
      return result;
#else
      keops_error(Error_msg_no_cuda);
#endif
    }
  
    case 11: {
#if USE_CUDA
      GpuReduc2D_FromHost(SS.nx, SS.ny, result_ptr, args_ptr, Device_Id);
      return result;
#else
      keops_error(Error_msg_no_cuda);
#endif
    }
  
    case 110: {
#if USE_CUDA
      GpuReduc1D_FromDevice(SS.nx, SS.ny, result_ptr, args_ptr, Device_Id);
      return result;
#else
      keops_error(Error_msg_no_cuda);
#endif
    }
  
    case 111: {
#if USE_CUDA
      GpuReduc2D_FromDevice(SS.nx, SS.ny, result_ptr, args_ptr, Device_Id);
      return result;
#else
      keops_error(Error_msg_no_cuda);
#endif
    }
  
    case 1000: {
      CpuReduc_ranges(SS.nx, SS.ny, SS.nbatchdims, SS.shapes,
                      RR.nranges_x, RR.nranges_y, RR.castedranges,
                      result_ptr, args_ptr);
      return result;
    }
  
    case 1010: {
#if USE_CUDA
      GpuReduc1D_ranges_FromHost(SS.nx, SS.ny, SS.nbatchdims, SS.shapes,
                                 RR.nranges_x, RR.nranges_y, RR.nredranges_x, RR.nredranges_y, RR.castedranges,
                                 result_ptr, args_ptr, Device_Id);
      return result;
#else
      keops_error(Error_msg_no_cuda);
#endif
    }
  
    case 1110: {
#if USE_CUDA
      GpuReduc1D_ranges_FromDevice(SS.nx, SS.ny, SS.nbatchdims, SS.shapes,
                                   RR.nranges_x, RR.nranges_y, RR.castedranges,
                                   result_ptr, args_ptr, Device_Id);
      return result;
#else
      keops_error(Error_msg_no_cuda);
#endif
    }
  
    default: {
      keops_error("[KeOps] Inconsistant values for tagHostDevice, tagCpuGpu, tagRanges, tag1D2D...");
      throw std::runtime_error("This is the end"); // A dummy error to avoid return-type warning
    }
  }
}
/*
template< typename array_t, typename array_t_out = array_t, typename index_array_t >
array_t launch_keops_ranges(int tag1D2D,
                            int tagCpuGpu,
                            int tagHostDevice,
                            int deviceId,
                            int nargs,
                            index_array_t* ranges,
                            array_t* args) {
  
  keops_binders::check_tag(tag1D2D, "1D2D");
  keops_binders::check_tag(tagCpuGpu, "CpuGpu");
  keops_binders::check_tag(tagHostDevice, "HostDevice");
  
  short int Device_Id_s = keops_binders::cast_Device_Id(deviceId);
  
  std::tuple< int, int, int, int* > sizes = keops_binders::check_ranges(nargs, args);
  int nx = std::get< 0 >(sizes);
  int ny = std::get< 1 >(sizes);
  int nbatchdims = std::get< 2 >(sizes);
  int* shapes = std::get< 3 >(sizes);
  
  get_output_shape(shapes, nbatchdims);
  
  array_t_out result = create_result_array< array_t_out, __TYPE__ >(nx, ny, tagHostDevice);
  __TYPE__ *result_array =  get_data< array_t_out, __TYPE__ >(result);
  
  // Create a decimal word to avoid nested conditional below
  int decision = 100 * tagHostDevice + 10 * tagCpuGpu + tag1D2D;

  switch (decision) {
    case 0: {
      auto result_array = allocate_result_array< array_t >(shape_out, nbatchdims);
      CpuReduc_ranges(nx,
                      ny,
                      nbatchdims,
                      shapes,
                      nranges_x,
                      nranges_y,
                      castedranges,
                      get_data(result_array),
                      args_ptr);
      return result_array;
    }

    case 10: {
#if USE_CUDA
      auto result_array = allocate_result_array< array_t >(shape_out, nbatchdims);
      GpuReduc1D_ranges_FromHost(nx, ny, nbatchdims, shapes, nranges_x, nranges_y, nredranges_x, nredranges_y, castedranges, get_data(result_array), args_ptr, Device_Id);
      return result_array;
#else
      keops_error(Error_msg_no_cuda);
#endif
    }

    case 110: {
#if USE_CUDA
      auto result_array = allocate_result_array_gpu< array_t >(shape_out, nbatchdims);
      GpuReduc1D_ranges_FromDevice(nx, ny, nbatchdims, shapes, nranges_x, nranges_y, castedranges, get_data(result_array), args_ptr, Device_Id);
      return result_array;
#else
      keops_error(Error_msg_no_cuda);
#endif
    }

    default: {
      keops_error("[KeOps] Inconsistant values for tagHostDevice, tagCpuGpu, tagRanges, tag1D2D...");
      throw std::runtime_error("This is the end"); // A dummy error to avoid return-type warning
    }
  }
}
*/
}
