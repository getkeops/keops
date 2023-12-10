#include <cuda.h>
#include <numeric>

#define NVRTC_SAFE_CALL(x)                                                     \
  do {                                                                         \
    nvrtcResult result = x;                                                    \
    if (result != NVRTC_SUCCESS) {                                             \
      std::cerr << "\nerror: " #x " failed with error "                        \
                << nvrtcGetErrorString(result) << '\n'                         \
                << '\n';                                                       \
      throw std::runtime_error("[KeOps] NVRTC error.");                        \
    }                                                                          \
  } while (0)

#define CUDA_SAFE_CALL_NO_EXCEPTION(x)                                         \
  do {                                                                         \
    CUresult result = x;                                                       \
    if (result != CUDA_SUCCESS) {                                              \
      const char *msg;                                                         \
      cuGetErrorName(result, &msg);                                            \
      std::cerr << "\n[KeOps] error: " #x " failed with error " << msg << '\n' \
                << '\n';                                                       \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CUDA_SAFE_CALL(x)                                                      \
  do {                                                                         \
    CUresult result = x;                                                       \
    if (result != CUDA_SUCCESS) {                                              \
      const char *msg;                                                         \
      cuGetErrorName(result, &msg);                                            \
      std::cerr << "\n[KeOps] error: " #x " failed with error " << msg << '\n' \
                << '\n';                                                       \
      throw std::runtime_error("[KeOps] Cuda error.");                         \
    }                                                                          \
  } while (0)

char *read_text_file(char const *path) {
  char *buffer = 0;
  long length;
  FILE *f = fopen(path, "rb");
  if (f) {
    fseek(f, 0, SEEK_END);
    length = ftell(f);
    fseek(f, 0, SEEK_SET);
    buffer = (char *)malloc((length + 1) * sizeof(char));
    if (buffer) {
      int res = fread(buffer, sizeof(char), length, f);
    }
    fclose(f);
  }
  buffer[length] = '\0';
  return buffer;
}

template <typename TYPE>
void load_args_FromDevice(CUdeviceptr &p_data, TYPE *out, TYPE *&out_d,
                          int nargs, TYPE **arg, TYPE **&arg_d) {
  out_d = out;
  arg_d = (TYPE **)p_data;
  // copy array of pointers
  CUDA_SAFE_CALL(cuMemcpyHtoD((CUdeviceptr)arg_d, arg, nargs * sizeof(TYPE *)));
}

template <typename TYPE>
void load_args_FromHost(CUdeviceptr &p_data, TYPE *out, TYPE *&out_d, int nargs,
                        TYPE **arg, TYPE **&arg_d,
                        const std::vector<std::vector<signed long int>> &argshape,
                        signed long int sizeout) {
  signed long int sizes[nargs];
  signed long int totsize = sizeout;
  for (int k = 0; k < nargs; k++) {
    sizes[k] = std::accumulate(argshape[k].begin(), argshape[k].end(), 1,
                               std::multiplies<signed long int>());
    totsize += sizes[k];
  }

  CUDA_SAFE_CALL(
      cuMemAlloc(&p_data, sizeof(TYPE *) * nargs + sizeof(TYPE) * totsize));

  arg_d = (TYPE **)p_data;
  TYPE *dataloc = (TYPE *)(arg_d + nargs);

  // host array of pointers to device data
  TYPE *ph[nargs];

  out_d = dataloc;
  dataloc += sizeout;
  for (int k = 0; k < nargs; k++) {
    ph[k] = dataloc;
    CUDA_SAFE_CALL(
        cuMemcpyHtoD((CUdeviceptr)dataloc, arg[k], sizeof(TYPE) * sizes[k]));
    dataloc += sizes[k];
  }

  // copy array of pointers
  CUDA_SAFE_CALL(cuMemcpyHtoD((CUdeviceptr)arg_d, ph, nargs * sizeof(TYPE *)));
}
