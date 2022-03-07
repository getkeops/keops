#pragma once

/////////////////////////////////////////////
//            GPU     Options             //
/////////////////////////////////////////////



// fix some Gpu properties
// CUDA_BLOCK_SIZE gives an upper bound on size of the size of Cuda blocks
// The actual block size may be lower due to memory limitations, depending on the formula used
#ifndef CUDA_BLOCK_SIZE
#define CUDA_BLOCK_SIZE 192
#endif
// Here we define the maximum number of threads per block and the shared memory per block
// These values can depend on the Gpu, although in fact values 1024 and 49152 respectively
// are the good values for almost all cards.
// So these values should be fine, but you can check them with GetGpuProps.cu program
// Here we assume that: either the user has defined MAXIDGPU (=number of Gpu devices minus one)
// and corresponding specific values MAXTHREADSPERBLOCK0, SHAREDMEMPERBLOCK0, MAXTHREADSPERBLOCK1, SHAREDMEMPERBLOCK1, ...
// for each device, or MAXIDGPU is not defined, and we will use global MAXTHREADSPERBLOCK and SHAREDMEMPERBLOCK
#ifndef MAXIDGPU
// we give default values
#ifndef MAXTHREADSPERBLOCK
#define MAXTHREADSPERBLOCK 1024
#endif
#ifndef SHAREDMEMPERBLOCK
#define SHAREDMEMPERBLOCK 49152
#endif
#endif

// global variables maxThreadsPerBlock and sharedMemPerBlock may depend on the device, so we will set them at each call using
// predefined MAXTHREADSPERBLOCK0, SHAREDMEMPERBLOCK0, MAXTHREADSPERBLOCK1, SHAREDMEMPERBLOCK1, etc.
// through the function SetGpuProps
int maxThreadsPerBlock, sharedMemPerBlock;

#define SET_GPU_PROPS_MACRO(n) \
    if(device == n) { \
      maxThreadsPerBlock = MAXTHREADSPERBLOCK ## n; \
      sharedMemPerBlock = SHAREDMEMPERBLOCK ## n; \
      return; \
    }

// I have not managed to use a "recursive macro" hack, it was not compiling on all systems.
// This assumes the number of Gpus is <= 10 ; feel free to add more lines if needed !
void SetGpuProps(int device) {

#if defined(MAXTHREADSPERBLOCK) && defined(SHAREDMEMPERBLOCK)
    // global values are defined
    maxThreadsPerBlock = MAXTHREADSPERBLOCK;
    sharedMemPerBlock = SHAREDMEMPERBLOCK;
    return;
#else
#if MAXIDGPU >= 0
    SET_GPU_PROPS_MACRO(0)
#endif
#if MAXIDGPU >= 1
    SET_GPU_PROPS_MACRO(1)
#endif
#if MAXIDGPU >= 2
    SET_GPU_PROPS_MACRO(2)
#endif
#if MAXIDGPU >= 3
    SET_GPU_PROPS_MACRO(3)
#endif
#if MAXIDGPU >= 4
    SET_GPU_PROPS_MACRO(4)
#endif
#if MAXIDGPU >= 5
    SET_GPU_PROPS_MACRO(5)
#endif
#if MAXIDGPU >= 6
    SET_GPU_PROPS_MACRO(6)
#endif
#if MAXIDGPU >= 7
    SET_GPU_PROPS_MACRO(7)
#endif
#if MAXIDGPU >= 8
    SET_GPU_PROPS_MACRO(8)
#endif
#if MAXIDGPU >= 9
    SET_GPU_PROPS_MACRO(9)
#endif
#if MAXIDGPU >= 10
    SET_GPU_PROPS_MACRO(10)
#endif
#if MAXIDGPU >= 11
    SET_GPU_PROPS_MACRO(11)
#endif
    fprintf( stderr, "invalid Gpu device number. If the number of available Gpus is > 12, add required lines at the end of function SetGpuProps and recompile.\n");
    exit( -1 );
#endif

}


