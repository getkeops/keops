// nvcc -shared -Xcompiler -fPIC -lnvrtc -lcuda keops_nvrtc.cu -o keops_nvrtc.so
// g++ --verbose -L/opt/cuda/lib64 -L/opt/cuda/targets/x86_64-linux/lib/
// -I/opt/cuda/targets/x86_64-linux/include/ -I../../include -shared -fPIC
// -lcuda -lnvrtc -fpermissive -DMAXIDGPU=0 -DMAXTHREADSPERBLOCK0=1024
// -DSHAREDMEMPERBLOCK0=49152 -DnvrtcGetTARGET=nvrtcGetCUBIN
// -DnvrtcGetTARGETSize=nvrtcGetCUBINSize -DARCHTAG=\"sm\" keops_nvrtc.cpp -o
// keops_nvrtc.so g++ -std=c++11  -shared -fPIC -O3 -fpermissive -L /usr/lib -L
// /opt/cuda/lib64 -lcuda -lnvrtc -DnvrtcGetTARGET=nvrtcGetCUBIN
// -DnvrtcGetTARGETSize=nvrtcGetCUBINSize -DARCHTAG=\"sm\"
// -I/home/bcharlier/projets/keops/keops/keops/include -I/opt/cuda/include
// -I/usr/include/python3.10/ -DMAXIDGPU=0 -DMAXTHREADSPERBLOCK0=1024
// -DSHAREDMEMPERBLOCK0=49152
// /home/bcharlier/projets/keops/keops/keops/binders/nvrtc/keops_nvrtc.cpp -o
// keops_nvrtc.cpython-310-x86_64-linux-gnu.so

#include <cuda.h>
#include <fstream>
#include <iostream>
#include <nvrtc.h>
#include <sstream>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <vector>
// #include <ctime>

#define C_CONTIGUOUS 1
#define USE_HALF 0

#include "include/Ranges.h"
#include "include/Sizes.h"
#include "include/ranges_utils.h"
#include "include/utils_pe.h"

#include "include/CudaSizes.h"
#include <cuda_fp16.h>

extern "C" int Compile(const char *target_file_name, const char *cu_code,
                       int use_half, int use_fast_math, int device_id,
                       const char *cuda_include_path) {

  nvrtcProgram prog;

  int numHeaders;
  const char *header_names[1];
  const char *header_sources[1];

  if (use_half) {
    numHeaders = 1;
    std::ostringstream header_path;
    header_path << cuda_include_path << "cuda_fp16.h";
    header_names[0] = "cuda_fp16.h";
    header_sources[0] = read_text_file(header_path.str().c_str());
  } else {
    numHeaders = 0;
  }

  // Get device id from Driver API
  CUdevice cuDevice;
  CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, device_id));

  // Get Compute Capability from Driver API
  int deviceProp_major, deviceProp_minor;
  CUDA_SAFE_CALL(cuDeviceGetAttribute(
      &deviceProp_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
      cuDevice));
  CUDA_SAFE_CALL(cuDeviceGetAttribute(
      &deviceProp_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
      cuDevice));

  std::ostringstream arch_flag;
  arch_flag << "-arch=" << ARCHTAG << "_" << deviceProp_major
            << deviceProp_minor;

  char *arch_flag_char = new char[arch_flag.str().length()];
  arch_flag_char = strdup(arch_flag.str().c_str());

  NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog,          // prog
                                     cu_code,        // buffer
                                     NULL,           // name
                                     numHeaders,     // numHeaders
                                     header_sources, // headers
                                     header_names    // includeNames
                                     ));

  nvrtcResult compileResult;
  if (use_fast_math) {
    const char *opts[] = {arch_flag_char, "-use_fast_math"};
    compileResult = nvrtcCompileProgram(prog,  // prog
                                        2,     // numOptions
                                        opts); // options
  } else {
    const char *opts[] = {arch_flag_char};
    compileResult = nvrtcCompileProgram(prog,  // prog
                                        1,     // numOptions
                                        opts); // options
  }

  // following "if" block is when there is a mismatch between
  // the device compute capability and the cuda libs versions : typically
  // when the device is more recent than the lib, the -arch flag may fail to
  // compile.
  if (compileResult == NVRTC_ERROR_INVALID_OPTION) {
    const char *new_opts[] = {"-use_fast_math"};
    compileResult = nvrtcCompileProgram(prog,      // prog
                                        1,         // numOptions
                                        new_opts); // options
  }

  if (compileResult != NVRTC_SUCCESS) {
    return compileResult;
  }

  delete[] arch_flag_char;

  // Obtain PTX or CUBIN from the program.
  size_t targetSize;
  NVRTC_SAFE_CALL(nvrtcGetTARGETSize(prog, &targetSize));

  char *target = new char[targetSize];
  NVRTC_SAFE_CALL(nvrtcGetTARGET(prog, target));

  // Destroy the program.
  NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

  // write PTX code to file

  std::ofstream wf(target_file_name, std::ofstream::binary);
  wf.write((char *)&targetSize, sizeof(size_t));
  wf.write(target, targetSize);
  wf.close();

  delete[] target;

  return 0;
}
