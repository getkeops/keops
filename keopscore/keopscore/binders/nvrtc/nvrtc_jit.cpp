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

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdarg.h>
#include <stdio.h>
#include <string>
#include <string.h>
#include <vector>

#include <cuda.h>
#include <nvrtc.h>

#define C_CONTIGUOUS 1
#define USE_HALF 0

#include "include/Ranges.h"
#include "include/Sizes.h"
#include "include/ranges_utils.h"
#include "include/utils_pe.h"

#include "include/CudaSizes.h"

extern "C" int Compile(const char *target_file_name, const char *cu_code,
                       int use_half, int use_fast_math, int device_id,
                       const char *cuda_include_paths) {

  nvrtcProgram prog;

  std::vector<std::string> compile_options_storage;
  std::vector<const char *> compile_options;

  if (cuda_include_paths != nullptr && cuda_include_paths[0] != '\0') {
    std::istringstream include_paths_stream(cuda_include_paths);
    std::string include_path;
    while (std::getline(include_paths_stream, include_path)) {
      if (!include_path.empty()) {
        compile_options_storage.emplace_back("--include-path=" + include_path);
      }
    }
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

  compile_options_storage.push_back(arch_flag.str());
  if (use_fast_math) {
    compile_options_storage.emplace_back("-use_fast_math");
  }
  compile_options.reserve(compile_options_storage.size());
  for (const std::string &option : compile_options_storage) {
    compile_options.push_back(option.c_str());
  }

  NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog,          // prog
                                     cu_code,        // buffer
                                     NULL,           // name
                                     0,              // numHeaders
                                     NULL,           // headers
                                     NULL            // includeNames
                                     ));

  nvrtcResult compileResult =
      nvrtcCompileProgram(prog,                      // prog
                          compile_options.size(),    // numOptions
                          compile_options.data());   // options

  // following "if" block is when there is a mismatch between
  // the device compute capability and the cuda libs versions : typically
  // when the device is more recent than the lib, the -arch flag may fail to
  // compile.
  if (compileResult == NVRTC_ERROR_INVALID_OPTION) {
    std::vector<const char *> fallback_options;
    fallback_options.reserve(compile_options_storage.size());
    for (const std::string &option : compile_options_storage) {
      if (option.rfind("-arch=", 0) != 0) {
        fallback_options.push_back(option.c_str());
      }
    }
    compileResult = nvrtcCompileProgram(prog,                     // prog
                                        fallback_options.size(),  // numOptions
                                        fallback_options.data()); // options
  }

  if (compileResult != NVRTC_SUCCESS) {
    size_t logSize = 0;
    nvrtcGetProgramLogSize(prog, &logSize);
    if (logSize > 1) {
      std::vector<char> log(logSize);
      nvrtcGetProgramLog(prog, log.data());
      std::cerr << "[KeOps] NVRTC compile log:\n" << log.data() << std::endl;
    }
    std::cerr << "[KeOps] nvrtcCompileProgram failed: "
              << nvrtcGetErrorString(compileResult) << std::endl;
    return compileResult;
  }

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
