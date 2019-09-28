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
array_t launch_keops(int tag1D2D,
                     int tagCpuGpu,
                     int tagHostDevice,
                     short int Device_Id,
                     int nx,
                     int ny,
                     int nbatchdims,
                     int *shapes,
                     int *shape_out,
                     int tagRanges,
                     int nranges_x,
                     int nranges_y,
                     int nredranges_x,
                     int nredranges_y,
                     __INDEX__ **castedranges,
                     __TYPE__ **castedargs) {
  // Create a decimal word to avoid nested conditional below
  int decision = 1000 * tagHostDevice + 100 * tagCpuGpu + 10 * tagRanges + tag1D2D;

  switch (decision) {
    case 0: {
      auto result_array = allocate_result_array< array_t >(shape_out, nbatchdims);
      CpuReduc(nx, ny, get_data(result_array), castedargs);
      return result_array;
    }

    case 10: {
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

    case 100: {
#if USE_CUDA
      auto result_array = allocate_result_array< array_t >(shape_out, nbatchdims);
      GpuReduc1D_FromHost(nx, ny, get_data(result_array), castedargs, Device_Id);
      return result_array;
#else
      keops_error(Error_msg_no_cuda);
#endif
    }

    case 101: {
#if USE_CUDA
      auto result_array = allocate_result_array< array_t >(shape_out, nbatchdims);
      GpuReduc2D_FromHost(nx, ny, get_data(result_array), castedargs, Device_Id);
      return result_array;
#else
      keops_error(Error_msg_no_cuda);
#endif
    }

    case 110: {
#if USE_CUDA
      auto result_array = allocate_result_array< array_t >(shape_out, nbatchdims);
      GpuReduc1D_ranges_FromHost(nx, ny, nbatchdims, shapes, nranges_x, nranges_y, nredranges_x, nredranges_y, castedranges, get_data(result_array), castedargs, Device_Id);
      return result_array;
#else
      keops_error(Error_msg_no_cuda);
#endif
    }

    case 1100: {
#if USE_CUDA
      auto result_array = allocate_result_array_gpu<at::Tensor >(shape_out, nbatchdims);
      GpuReduc1D_FromDevice(nx, ny, get_data(result_array), castedargs, Device_Id);
      return result_array;
#else
      keops_error(Error_msg_no_cuda);
#endif
    }
    case 1101: {
#if USE_CUDA
      auto result_array = allocate_result_array_gpu< array_t >(shape_out, nbatchdims);
      GpuReduc2D_FromDevice(nx, ny, get_data(result_array), castedargs, Device_Id);
      return result_array;
#else
      keops_error(Error_msg_no_cuda);
#endif
    }

    case 1110: {
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