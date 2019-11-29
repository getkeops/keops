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

template < typename array_t >
array_t allocate_result_array(const size_t *a, const size_t b = 0);

template < typename array_t >
array_t allocate_result_array_gpu(const size_t *a, const size_t b = 0);

template < typename array_t >
array_t create_result_array(const int nx, const int ny, const int tagHostDevice = 0) {

  size_t shape_out[2] = {keops::DIMOUT, (keops::TAGIJ == 1) ? ny : nx};

  array_t result =
      (tagHostDevice == 0) ? allocate_result_array< array_t >(shape_out) : allocate_result_array_gpu< array_t >(
          shape_out);
  return result;
}



size_t* get_output_shape(size_t* shapes = {}, size_t nbatchdims = 0) {
// Store, in a raw int array, the shape of the output: =====================
// [A, .., B, M, D]  if TAGIJ==0
//  or
// [A, .., B, N, D]  if TAGIJ==1
  
  size_t *shape_output = new size_t[nbatchdims + 2];
  for (size_t b = 0; b < nbatchdims; b++) {
    shape_output[b] = shapes[b];                               // Copy the "batch dimensions"
  }
  shape_output[nbatchdims] = shapes[nbatchdims + TAGIJ];      // M or N
  shape_output[nbatchdims + 1] = shapes[nbatchdims + 2];      // D
  return shape_output;
}

void launch_keops(int tag1D2D,
                  int tagCpuGpu,
                  int tagHostDevice,
                  short int Device_Id,
                  int nx,
                  int ny,
                  __TYPE__ *result_array,
                  __TYPE__ **castedargs) {
  // Create a decimal word to avoid nested conditional below
  int decision = 100 * tagHostDevice + 10 * tagCpuGpu + tag1D2D;

  switch (decision) {
    case 0: {
      //auto result_array = allocate_result_array< array_t >(shape_out);
      CpuReduc(nx, ny, result_array, castedargs);
      return;
    }

    case 10: {
#if USE_CUDA
      //auto result_array = allocate_result_array< array_t >(shape_out);
      GpuReduc1D_FromHost(nx, ny, result_array, castedargs, Device_Id);
      return;
#else
      keops_error(Error_msg_no_cuda);
#endif
    }

    case 11: {
#if USE_CUDA
      //auto result_array = allocate_result_array< array_t >(shape_out);
      GpuReduc2D_FromHost(nx, ny, result_array, castedargs, Device_Id);
      return;
#else
      keops_error(Error_msg_no_cuda);
#endif
    }

    case 110: {
#if USE_CUDA
      //auto result_array = allocate_result_array_gpu< array_t >(shape_out);
      GpuReduc1D_FromDevice(nx, ny, result_array, castedargs, Device_Id);
      return;
#else
      keops_error(Error_msg_no_cuda);
#endif
    }

    case 111: {
#if USE_CUDA
      //auto result_array = allocate_result_array_gpu< array_t >(shape_out);
      GpuReduc2D_FromDevice(nx, ny, result_array, castedargs, Device_Id);
      return;
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
                            __TYPE__ **castedargs) {
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
                      castedargs);
      return result_array;
    }

    case 10: {
#if USE_CUDA
      auto result_array = allocate_result_array< array_t >(shape_out, nbatchdims);
      GpuReduc1D_ranges_FromHost(nx, ny, nbatchdims, shapes, nranges_x, nranges_y, nredranges_x, nredranges_y, castedranges, get_data(result_array), castedargs, Device_Id);
      return result_array;
#else
      keops_error(Error_msg_no_cuda);
#endif
    }

    case 110: {
#if USE_CUDA
      auto result_array = allocate_result_array_gpu< array_t >(shape_out, nbatchdims);
      GpuReduc1D_ranges_FromDevice(nx, ny, nbatchdims, shapes, nranges_x, nranges_y, castedranges, get_data(result_array), castedargs, Device_Id);
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
