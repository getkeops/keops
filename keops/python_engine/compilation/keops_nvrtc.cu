
// nvcc -shared -Xcompiler -fPIC -lnvrtc -lcuda keops_nvrtc.cu -o keops_nvrtc.so

#include <nvrtc.h>
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <stdarg.h>

#define __INDEX__ int //int32_t

#define C_CONTIGUOUS 1
#define USE_HALF 0
#define CUDA_BLOCK_SIZE 192

#include "Sizes.h"
#include "Ranges.h"

#include "CudaSizes.h"

#include <cuda_fp16.h>

#include "utils.cpp"
#include "ranges_utils.cpp"

extern "C" __host__ int Compile(const char* ptx_file_name, const char* cu_code, int use_half, int device_id) {

    char *ptx;

    nvrtcProgram prog;
    
    int numHeaders;
    const char* header_names[2];
    const char* header_sources[2];
    if (use_half) {
        numHeaders = 2;
        header_names[0] = "cuda_fp16.h";
        header_sources[0] = read_text_file("/usr/include/cuda_fp16.h");
        
        header_names[1] = "cuda_fp16.hpp";
        header_sources[1] = read_text_file("/usr/include/cuda_fp16.hpp");

    } else {
        numHeaders = 0;
    }
    
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_id);
    
    NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog,         // prog
                   cu_code,         // buffer
                   NULL,    // name
                   numHeaders,             // numHeaders
                   header_sources,                    // headers
                   header_names               // includeNames 
                   ));

    std::ostringstream arch_flag;
    arch_flag << "-arch=compute_" << deviceProp.major << deviceProp.minor;
    
    const char *opts[] = {strdup(arch_flag.str().c_str())};
    
    nvrtcResult compileResult = nvrtcCompileProgram(prog,  // prog
                                              1,     // numOptions
                                              opts); // options
              
    // Obtain compilation log from the program.
    size_t logSize;
    NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
    char *log = new char[logSize];
    NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));
    std::cout << log << '\n';
    delete[] log;
    if (compileResult != NVRTC_SUCCESS) {
        exit(1);
    }

    // Obtain PTX from the program.
    size_t ptxSize;
    NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
    ptx = new char[ptxSize];
    NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));
    // Destroy the program.
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
    
    // write ptx code to file
    FILE *ptx_file = fopen(ptx_file_name, "w");
    fputs(ptx, ptx_file);
    fclose(ptx_file);

    return 0;
}






template < typename TYPE >
__host__ int launch_keops(const char* ptx_file_name, int tagHostDevice, int dimY, int nx, int ny, 
                                        int device_id, int tagI, int tagZero, int use_half,
                                        int *indsi, int *indsj, int *indsp,
                                        int dimout, 
                                        int *dimsx, int *dimsy, int *dimsp,
                                        int **ranges, int *shapeout, TYPE *out, int nargs, TYPE **arg, int **argshape) {
    
    CUdevice cuDevice;
    CUDA_SAFE_CALL(cuInit(0));
    CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, device_id));
    
    cudaSetDevice(device_id);
    
    SetGpuProps(device_id);
    
    Sizes<TYPE> SS(nargs, arg, argshape, nx, ny, 
                   tagI, use_half,
                   dimout, 
                   indsi, indsj, indsp,
                   dimsx, dimsy, dimsp);
                        
    if (use_half)
        SS.switch_to_half2_indexing();

    Ranges<TYPE> RR(SS, ranges);
    
    nx = SS.nx;
    ny = SS.ny;  
    
    // now we switch (back...) indsi, indsj and dimsx, dimsy in case tagI=1.
    // This is to be consistent with the convention used in the old
    // bindings where i and j variables had different meanings in bindings
    // and in the core code. Clearly we could do better if we
    // carefully rewrite some parts of the code
    if (tagI==1) {
        int *tmp;
        tmp = indsj;
        indsj = indsi;
        indsi = tmp;
        tmp = dimsy;
        dimsy = dimsx;
        dimsx = tmp;
    }
    
    
    dim3 blockSize;
    blockSize.x = ::std::min(CUDA_BLOCK_SIZE,
                             ::std::min(maxThreadsPerBlock,
                                        (int) (sharedMemPerBlock / ::std::max(1,
                                                    (int) (  dimY * sizeof(TYPE))))));
                                                    
    dim3 gridSize;
    
    int nblocks;
    
    if (tagI==1) {
        int tmp = ny;
        ny = nx;
        nx = tmp;
    }

    
    __INDEX__ *lookup_d = NULL, *slices_x_d = NULL, *ranges_y_d = NULL;
    int *offsets_d = NULL;
    
    if (RR.tagRanges==1) {
        range_preprocess(nblocks, tagI, RR.nranges_x, RR.nranges_y, RR.castedranges,
                         SS.nbatchdims, slices_x_d, ranges_y_d, lookup_d,
                         offsets_d,
                         blockSize.x, indsi, indsj, indsp, SS.shapes);
    }
    
    void *p_data;
    TYPE *out_d;
    TYPE **arg_d;
    int sizeout = get_sum(shapeout);
    
    if(tagHostDevice==1)
        load_args_FromDevice(p_data, out, out_d, nargs, arg, arg_d);
    else
        load_args_FromHost(p_data, out, out_d, nargs, arg, arg_d, argshape, sizeout);
    
    
    char *ptx;
    ptx = read_text_file(ptx_file_name);
    
    CUmodule module;
    CUfunction kernel;
    
    CUjit_option jitOptions[1];
    void* jitOptVals[1];
    jitOptions[0] = CU_JIT_TARGET;
    long targ_comp = 61;
    jitOptVals[0] = (void *)targ_comp;
    
    CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx, 1, jitOptions, jitOptVals));
    
    if (RR.tagRanges==1 && tagZero==0) {
        // ranges mode
        
        gridSize.x = nblocks;
        
        CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "GpuConv1DOnDevice_ranges_NoChunks"));

        void *kernel_params[9];
        kernel_params[0] = &nx;
        kernel_params[1] = &ny;
        kernel_params[2] = &SS.nbatchdims;
        kernel_params[3] = &offsets_d;
        kernel_params[4] = &lookup_d;
        kernel_params[5] = &slices_x_d;
        kernel_params[6] = &ranges_y_d;
        kernel_params[7] = &out_d;
        kernel_params[8] = &arg_d;

        CUDA_SAFE_CALL(cuLaunchKernel(kernel,
                   gridSize.x, gridSize.y, gridSize.z,    // grid dim
                   blockSize.x, blockSize.y, blockSize.z,   // block dim
                   blockSize.x * dimY * sizeof(TYPE), NULL,             // shared mem and stream
                   kernel_params, 0));           // arguments
                   
    } else {
        // simple mode
        
        gridSize.x = nx / blockSize.x + (nx % blockSize.x == 0 ? 0 : 1);
        
        CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "GpuConv1DOnDevice"));

        void *kernel_params[4];
        kernel_params[0] = &nx;
        kernel_params[1] = &ny;
        kernel_params[2] = &out_d;
        kernel_params[3] = &arg_d;

        CUDA_SAFE_CALL(cuLaunchKernel(kernel,
                   gridSize.x, gridSize.y, gridSize.z,    // grid dim
                   blockSize.x, blockSize.y, blockSize.z,   // block dim
                   blockSize.x * dimY * sizeof(TYPE), NULL,             // shared mem and stream
                   kernel_params, 0));           // arguments
    }
    
    CUDA_SAFE_CALL(cuCtxSynchronize());

    CUDA_SAFE_CALL(cuModuleUnload(module));
    
    // Send data from device to host.

    if(tagHostDevice==0)
        cudaMemcpy(out, out_d, sizeof(TYPE) * sizeout, cudaMemcpyDeviceToHost);

    cudaFree(p_data);
    if (RR.tagRanges==1) {
        cudaFree(lookup_d);
        cudaFree(slices_x_d);
        cudaFree(ranges_y_d);
        if (SS.nbatchdims > 0)
            cudaFree(offsets_d);
    }

    return 0;
}






extern "C" __host__ int launch_keops_float(const char* ptx_file_name, int tagHostDevice, int dimY, int nx, int ny, 
                                        int device_id, int tagI, int tagZero, int use_half,
                                        int *indsi, int *indsj, int *indsp, 
                                        int dimout, 
                                        int *dimsx, int *dimsy, int *dimsp,
                                        int **ranges, int *shapeout, float *out, int nargs, ...) {
    // reading arguments
    va_list ap;
    va_start(ap, nargs);
    float *arg[nargs];
    for (int i=0; i<nargs; i++)
        arg[i] = va_arg(ap, float*);
    int *argshape[nargs];
    for (int i=0; i<nargs; i++)
        argshape[i] = va_arg(ap, int*);
    va_end(ap);
    
    return launch_keops(ptx_file_name, tagHostDevice, dimY, nx, ny, device_id, tagI, tagZero, use_half,
                                        indsi, indsj, indsp,
                                        dimout,
                                        dimsx, dimsy, dimsp,
                                        ranges, shapeout, out, nargs, arg, argshape);
                                                                        
}





extern "C" __host__ int launch_keops_double(const char* ptx_file_name, int tagHostDevice, int dimY, int nx, int ny, 
                                        int device_id, int tagI, int tagZero, int use_half,
                                        int *indsi, int *indsj, int *indsp, 
                                        int dimout, 
                                        int *dimsx, int *dimsy, int *dimsp,
                                        int **ranges, int *shapeout, double *out, int nargs, ...) {
    // reading arguments
    va_list ap;
    va_start(ap, nargs);
    double *arg[nargs];
    for (int i=0; i<nargs; i++)
        arg[i] = va_arg(ap, double*);
    int *argshape[nargs];
    for (int i=0; i<nargs; i++)
        argshape[i] = va_arg(ap, int*);
    va_end(ap);
    
    return launch_keops(ptx_file_name, tagHostDevice, dimY, nx, ny, device_id, tagI, tagZero, use_half,
                                        indsi, indsj, indsp,
                                        dimout,
                                        dimsx, dimsy, dimsp,
                                        ranges, shapeout, out, nargs, arg, argshape);
                                                                        
}




extern "C" __host__ int launch_keops_half(const char* ptx_file_name, int tagHostDevice, int dimY, int nx, int ny, 
                                        int device_id, int tagI, int tagZero, int use_half,
                                        int *indsi, int *indsj, int *indsp, 
                                        int dimout, 
                                        int *dimsx, int *dimsy, int *dimsp,
                                        int **ranges, int *shapeout, half2 *out, int nargs, ...) {
    // reading arguments
    va_list ap;
    va_start(ap, nargs);
    half2 *arg[nargs];
    for (int i=0; i<nargs; i++)
        arg[i] = va_arg(ap, half2*);
    int *argshape[nargs];
    for (int i=0; i<nargs; i++)
        argshape[i] = va_arg(ap, int*);
    va_end(ap);
    
    return launch_keops(ptx_file_name, tagHostDevice, dimY, nx, ny, device_id, tagI, tagZero, use_half,
                                        indsi, indsj, indsp,
                                        dimout,
                                        dimsx, dimsy, dimsp,
                                        ranges, shapeout, out, nargs, arg, argshape);
                                                                        
}
