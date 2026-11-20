// Maps the CUDA driver API names used by the KeOps nvrtc binder to HIP.
// This directory is only on the include path of ROCm builds.
#pragma once

#include <hip/hip_runtime_api.h>

#define CUDA_SUCCESS hipSuccess
#define CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK                             \
  hipDeviceAttributeMaxThreadsPerBlock
#define CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK                       \
  hipDeviceAttributeMaxSharedMemoryPerBlock

typedef hipError_t CUresult;
typedef hipDevice_t CUdevice;
// HIP's context API is deprecated. Each device has a single primary context
// in HIP, so the device stands in for it and is made current with
// hipSetDevice.
typedef hipDevice_t CUcontext;
typedef hipModule_t CUmodule;
typedef hipFunction_t CUfunction;
typedef hipDeviceptr_t CUdeviceptr;

#define cuGetErrorName hipDrvGetErrorName
#define cuInit hipInit
#define cuDeviceGet hipDeviceGet
#define cuDeviceGetAttribute hipDeviceGetAttribute
#define cuCtxSynchronize hipDeviceSynchronize
#define cuMemAlloc hipMalloc
#define cuMemFree hipFree
#define cuMemcpyHtoD hipMemcpyHtoD
#define cuMemcpyDtoH hipMemcpyDtoH
#define cuModuleLoadDataEx hipModuleLoadDataEx
#define cuModuleGetFunction hipModuleGetFunction
#define cuModuleUnload hipModuleUnload
#define cuLaunchKernel hipModuleLaunchKernel

inline hipError_t cuDevicePrimaryCtxRetain(CUcontext *ctx, CUdevice device) {
  *ctx = device;
  return hipSuccess;
}

inline hipError_t cuDevicePrimaryCtxRelease(CUdevice) { return hipSuccess; }

inline hipError_t cuCtxGetCurrent(CUcontext *ctx) { return hipGetDevice(ctx); }

inline hipError_t cuCtxPushCurrent(CUcontext ctx) { return hipSetDevice(ctx); }
