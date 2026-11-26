// Maps the NVRTC names used by the KeOps nvrtc binder to hipRTC.
// This directory is only on the include path of ROCm builds.
#pragma once

#include <hip/hiprtc.h>

#define NVRTC_SUCCESS HIPRTC_SUCCESS
#define NVRTC_ERROR_INVALID_OPTION HIPRTC_ERROR_INVALID_OPTION

typedef hiprtcProgram nvrtcProgram;
typedef hiprtcResult nvrtcResult;

#define nvrtcCreateProgram hiprtcCreateProgram
#define nvrtcCompileProgram hiprtcCompileProgram
#define nvrtcDestroyProgram hiprtcDestroyProgram
#define nvrtcGetProgramLog hiprtcGetProgramLog
#define nvrtcGetProgramLogSize hiprtcGetProgramLogSize
#define nvrtcGetErrorString hiprtcGetErrorString
