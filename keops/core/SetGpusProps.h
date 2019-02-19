
#pragma once

namespace keops {
// global variables maxThreadsPerBlock and sharedMemPerBlock may depend on the device, so we will set them at each call using
// predefined GPU0_MAXTHREADSPERBLOCK, GPU0_SHAREDMEMPERBLOCK, GPU1_MAXTHREADSPERBLOCK, GPU1_SHAREDMEMPERBLOCK, etc.
// through the function SetGpuProps
int maxThreadsPerBlock, sharedMemPerBlock;
#define SET_GPU_PROPS_MACRO(n, _) \
    if(device==n) { \
		maxThreadsPerBlock = MAXTHREADSPERBLOCK ## n; \
		sharedMemPerBlock = SHAREDMEMPERBLOCK ## n; \
	}
void SetGpuProps(int device) { REPEATMACRO(SET_GPU_PROPS_MACRO,MAXIDGPU) }

}
