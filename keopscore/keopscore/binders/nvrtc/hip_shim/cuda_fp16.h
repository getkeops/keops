// Half precision types and intrinsics for ROCm builds. hipRTC declares
// them itself; the host compile takes them from the HIP headers.
// This directory is only on the include path of ROCm builds.
#pragma once

#if !defined(__HIPCC_RTC__)
#include <hip/hip_fp16.h>
#endif
