#pragma once

extern "C" {
int CpuReduc(int, int, __TYPE__ *, __TYPE__ **);
int CpuReduc_ranges(int, int, int, int *, int, int, __INDEX__ **, __TYPE__ *, __TYPE__ **);
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

void keops_error(std::basic_string< char >);

template< typename array_t, typename _T >
array_t allocate_result_array(const size_t* a, const size_t b = 0);

template< typename array_t, typename _T >
array_t allocate_result_array_gpu(const size_t* a, const size_t b = 0);

template< typename array_t, typename _T >
array_t create_result_array(const int nx, const int ny, const int tagHostDevice = 0) {

  #if C_CONTIGUOUS
  size_t shape_out[2] = { (keops::TAGIJ == 1) ? static_cast< size_t >(ny) : static_cast< size_t >(nx), static_cast< size_t >(keops::DIMOUT)};
  #else
  size_t shape_out[2] = {static_cast< size_t >(keops::DIMOUT),
                         (keops::TAGIJ == 1) ? static_cast< size_t >(ny) : static_cast< size_t >(nx)};
  #endif
  array_t result = (tagHostDevice == 0) ? allocate_result_array< array_t, _T >(shape_out)
                                        : allocate_result_array_gpu< array_t, _T >(shape_out);
  return result;
}



int* get_output_shape(int* shapes = {}, int nbatchdims = 0) {
// Store, in a raw int array, the shape of the output: =====================
// [A, .., B, M, D]  if TAGIJ==0
//  or
// [A, .., B, N, D]  if TAGIJ==1
  
  int *shape_output = new int[nbatchdims + 2];
  for (int b = 0; b < nbatchdims; b++) {
    shape_output[b] = shapes[b];                               // Copy the "batch dimensions"
  }
  shape_output[nbatchdims] = shapes[nbatchdims + TAGIJ];      // M or N
  shape_output[nbatchdims + 1] = shapes[nbatchdims + 2];      // D
  return shape_output;
}

template < typename array_t, typename array_t_out >
array_t_out launch_keops(int tag1D2D,
                  int tagCpuGpu,
                  int tagHostDevice,
                  int deviceId,
                  int nargs,
                  array_t *args) {
  
  keops_binders::check_tag(tag1D2D, "1D2D");
  keops_binders::check_tag(tagCpuGpu, "CpuGpu");
  keops_binders::check_tag(tagHostDevice, "HostDevice");
  
  short int Device_Id_s = keops_binders::cast_Device_Id(deviceId);
  
  std::tuple< int, int, int, int* > sizes = keops_binders::check_ranges(nargs, args);
  int nx = std::get< 0 >(sizes);
  int ny = std::get< 1 >(sizes);
  
  array_t_out result = create_result_array< array_t_out, __TYPE__ >(nx, ny, tagHostDevice);
  __TYPE__ *result_array =  get_data< array_t_out, __TYPE__ >(result);
  
  // get the pointers to data to avoid a copy
  __TYPE__* args_ptr[NARGS];
  for (int i = 0; i < NARGS; i++)
    args_ptr[i] = keops_binders::get_data< array_t, __TYPE__ >(args[i]);
  
  // Create a decimal word to avoid nested conditional below
  int decision = 100 * tagHostDevice + 10 * tagCpuGpu + tag1D2D;

  switch (decision) {
    case 0: {
      //auto result_array = allocate_result_array< array_t >(shape_out);
      CpuReduc(nx, ny, result_array, args_ptr);
      return result;
    }

    case 10: {
#if USE_CUDA
      //auto result_array = allocate_result_array< array_t >(shape_out);
      GpuReduc1D_FromHost(nx, ny, result_array, args_ptr, Device_Id);
      return result;
#else
      keops_error(Error_msg_no_cuda);
#endif
    }

    case 11: {
#if USE_CUDA
      //auto result_array = allocate_result_array< array_t >(shape_out);
      GpuReduc2D_FromHost(nx, ny, result_array, args_ptr, Device_Id);
      return result;
#else
      keops_error(Error_msg_no_cuda);
#endif
    }

    case 110: {
#if USE_CUDA
      //auto result_array = allocate_result_array_gpu< array_t >(shape_out);
      GpuReduc1D_FromDevice(nx, ny, result_array, args_ptr, Device_Id);
      return result;
#else
      keops_error(Error_msg_no_cuda);
#endif
    }

    case 111: {
#if USE_CUDA
      //auto result_array = allocate_result_array_gpu< array_t >(shape_out);
      GpuReduc2D_FromDevice(nx, ny, result_array, args_ptr, Device_Id);
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

template < typename array_t >
array_t launch_keops_ranges(int tag1D2D, int tagCpuGpu, int tagHostDevice,
                            short int Device_Id,
                            int nx, int ny,
                            int nbatchdims, int *shapes, int *shape_out,
                            int nranges_x, int nranges_y,
                            int nredranges_x, int nredranges_y,
                            __INDEX__ **castedranges,
                            __TYPE__ **args_ptr) {
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

}
