
// nvcc -shared -Xcompiler -fPIC -lnvrtc -lcuda keops_nvrtc.cu -o keops_nvrtc.so
// g++ --verbose -L/opt/cuda/lib64 -L/opt/cuda/targets/x86_64-linux/lib/ -I/opt/cuda/targets/x86_64-linux/include/ -shared -fPIC -lcuda -lnvrtc -fpermissive -DMAXIDGPU=0 -DMAXTHREADSPERBLOCK0=1024 -DSHAREDMEMPERBLOCK0=49152  /home/bcharlier/projets/keops/keops/keops/python_engine/compilation/keops_nvrtc.cpp -o /home/bcharlier/projets/keops/keops/keops/python_engine/build/keops_nvrtc.so

#include <nvrtc.h>
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdarg.h>
#include <ctime>

#define __INDEX__ int
#define C_CONTIGUOUS 1
#define USE_HALF 0

#include "Sizes.h"
#include "Ranges.h"
#include "utils_pe.h"
#include "ranges_utils.h"


#include "CudaSizes.h"
#include <cuda_fp16.h>


extern "C" int Compile(const char *target_file_name, const char *cu_code, int use_half, int device_id, 
                        const char *cuda_include_path) {

    nvrtcProgram prog;

    int numHeaders;
    const char *header_names[2];
    const char *header_sources[2];
    
    std::ostringstream cuda_fp16_h_path, cuda_fp16_hpp_path;
    cuda_fp16_h_path << cuda_include_path << "cuda_fp16.h" ;
    cuda_fp16_hpp_path << cuda_include_path << "cuda_fp16.hpp" ;
    
    if (use_half) {
        numHeaders = 2;
        header_names[0] = "cuda_fp16.h";
        header_sources[0] = read_text_file(cuda_fp16_h_path.str().c_str());

        header_names[1] = "cuda_fp16.hpp";
        header_sources[1] = read_text_file(cuda_fp16_hpp_path.str().c_str());

    } else {
        numHeaders = 0;
    }

    // Get device id from Driver API
    CUdevice cuDevice;
    cuDeviceGet(&cuDevice, device_id);

    // Get Compute Capability from Driver API
    int deviceProp_major, deviceProp_minor;
    cuDeviceGetAttribute(&deviceProp_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice);
    cuDeviceGetAttribute(&deviceProp_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice);
    
    std::ostringstream arch_flag;
    arch_flag << "-arch=" << ARCHTAG << "_" << deviceProp_major << deviceProp_minor;

    char *arch_flag_char = new char[arch_flag.str().length()];
    arch_flag_char = strdup(arch_flag.str().c_str());
    const char *opts[] = {arch_flag_char, "-use_fast_math"};

    NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog,         // prog
                                       cu_code,         // buffer
                                       NULL,            // name
                                       numHeaders,      // numHeaders
                                       header_sources,  // headers
                                       header_names     // includeNames
    ));

    nvrtcResult compileResult = nvrtcCompileProgram(prog,     // prog
                                              2,              // numOptions
                                              opts);          // options
                                            
    delete[] arch_flag_char;

    if (compileResult != NVRTC_SUCCESS) {
        exit(1);
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
    wf.write((char*)&targetSize, sizeof(size_t));
    wf.write(target, targetSize);
    wf.close();

    return 0;
}


template<typename TYPE>
int launch_keops(const char *target_file_name, int tagHostDevice, int dimY, int nx, int ny,
                 int device_id, int tagI, int tagZero, int use_half,
                 int tag1D2D, int dimred,
                 int cuda_block_size, int use_chunk_mode,
                 int *indsi, int *indsj, int *indsp,
                 int dimout,
                 int *dimsx, int *dimsy, int *dimsp,
                 int **ranges, int *shapeout, TYPE *out, int nargs, float *buffer, TYPE **arg, int **argshape) {

    
    clock_t A, B, begin, end;
    A = clock();

    CUdevice cuDevice;
    CUcontext ctx;
    
    begin = clock();
    CUDA_SAFE_CALL(cuInit(0));
    CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, device_id));
    CUDA_SAFE_CALL(cuDevicePrimaryCtxRetain(&ctx, cuDevice));
    CUDA_SAFE_CALL(cuCtxPushCurrent(ctx));

    SetGpuProps(device_id);
    end = clock();
    std::cout << "time for cuda init : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
    begin = clock();
    
    Sizes<TYPE> SS(nargs, arg, argshape, nx, ny,
                   tagI, use_half,
                   dimout,
                   indsi, indsj, indsp,
                   dimsx, dimsy, dimsp);
                   
    end = clock();
    std::cout << "time for Sizes : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
    begin = clock();
    
    if (use_half)
        SS.switch_to_half2_indexing();

    Ranges<TYPE> RR(SS, ranges);
    nx = SS.nx;
    ny = SS.ny;
    
    end = clock();
    std::cout << "time for Ranges : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
    begin = clock();

    // now we switch (back...) indsi, indsj and dimsx, dimsy in case tagI=1.
    // This is to be consistent with the convention used in the old
    // bindings where i and j variables had different meanings in bindings
    // and in the core code. Clearly we could do better if we
    // carefully rewrite some parts of the code
    if (tagI == 1) {
        int *tmp;
        tmp = indsj;
        indsj = indsi;
        indsi = tmp;
        tmp = dimsy;
        dimsy = dimsx;
        dimsx = tmp;
    }


    int blockSize_x = 1, blockSize_y = 1, blockSize_z = 1;

    if (use_chunk_mode == 0) {
        // warning : blockSize.x was previously set to CUDA_BLOCK_SIZE; currently CUDA_BLOCK_SIZE value is used as a bound.
        blockSize_x = std::min(cuda_block_size, std::min(maxThreadsPerBlock, (int) (sharedMemPerBlock / std::max(1, (int) (dimY * sizeof(TYPE)))))); // number of threads in each block
    } else {
        // warning : the value here must match the one which is set in file GpuReduc1D_chunks.py, line 59
        // and file GpuReduc1D_finalchunks.py, line 67
        blockSize_x = std::min(cuda_block_size,
                               std::min(1024, (int) (49152 / std::max(1, (int) (dimY * sizeof(TYPE))))));
    }

    int nblocks;

    if (tagI == 1) {
        int tmp = ny;
        ny = nx;
        nx = tmp;
    }

    __INDEX__ *lookup_d = NULL, *slices_x_d = NULL, *ranges_y_d = NULL;
    int *offsets_d = NULL;
    
    end = clock();
    std::cout << "time for interm : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
    begin = clock();

    if (RR.tagRanges==1) {
        if (tagHostDevice==1) {
            range_preprocess_from_device(nblocks, tagI, RR.nranges_x, RR.nranges_y, RR.castedranges,
                         SS.nbatchdims, slices_x_d, ranges_y_d, lookup_d,
                         offsets_d,
                         blockSize_x, indsi, indsj, indsp, SS.shapes);
        }
        else { // tagHostDevice==0
            range_preprocess_from_host(nblocks, tagI, RR.nranges_x, RR.nranges_y, RR.nredranges_x, RR.nredranges_y, RR.castedranges,
                         SS.nbatchdims, slices_x_d, ranges_y_d, lookup_d,
                         offsets_d,
                         blockSize_x, indsi, indsj, indsp, SS.shapes);
        }
    }
    
    end = clock();
    std::cout << "time for range_preprocess : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
    begin = clock();

    CUdeviceptr p_data;
    TYPE *out_d;
    TYPE **arg_d;
    int sizeout = get_sum(shapeout);

    end = clock();
    std::cout << "time for pre-load args : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
    begin = clock();

    if (tagHostDevice == 1)
        load_args_FromDevice(p_data, out, out_d, nargs, arg, arg_d, buffer);
    else
        load_args_FromHost(p_data, out, out_d, nargs, arg, arg_d, argshape, sizeout);

    end = clock();
    std::cout << "time for load args : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
    begin = clock();

    char *target;
    std::ifstream rf(target_file_name, std::ifstream::binary);
    size_t targetSize;
    rf.read((char*)&targetSize, sizeof(size_t));
    target = new char[targetSize];
    rf.read(target, targetSize);
    rf.close();
    
    end = clock();
    std::cout << "time for reading : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
    begin = clock();
    
    CUmodule module;
    CUfunction kernel;

    //CUjit_option jitOptions[1];
    //void* jitOptVals[1];
    //jitOptions[0] = CU_JIT_TARGET;
    //long targ_comp = 75;
    //jitOptVals[0] = (void *)targ_comp;
    
    CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, target, 0, NULL, NULL));
    
    end = clock();
    std::cout << "time for loading kernel : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
    begin = clock();
    
    int gridSize_x = 1, gridSize_y = 1, gridSize_z = 1;

    if (tag1D2D == 1) { // 2D scheme

        gridSize_x = nx / blockSize_x + (nx % blockSize_x == 0 ? 0 : 1);
        gridSize_y = ny / blockSize_x + (ny % blockSize_x == 0 ? 0 : 1);

        // Reduce : grid and block are both 1d
        int blockSize2_x = 1, blockSize2_y = 1, blockSize2_z = 1;
        blockSize2_x = blockSize_x; // number of threads in each block
        int gridSize2_x = 1, gridSize2_y = 1, gridSize2_z = 1;
        gridSize2_x = (nx * dimred) / blockSize2_x + ((nx * dimred) % blockSize2_x == 0 ? 0 : 1);

        // Data on the device. We need an "inflated" outB, which contains gridSize.y "copies" of out
        // that will be reduced in the final pass.
        TYPE *outB;

        // single cudaMalloc
        CUdeviceptr p_data_outB;
        CUDA_SAFE_CALL(cuMemAlloc(&p_data_outB, sizeof(TYPE) * (nx * dimred * gridSize_y)));

        outB = (TYPE *) ((TYPE **) p_data);

        CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "GpuConv2DOnDevice"));

        void *kernel_params[4];
        kernel_params[0] = &nx;
        kernel_params[1] = &ny;
        kernel_params[2] = &outB;
        kernel_params[3] = &arg_d;

        // Size of the SharedData : blockSize.x*(DIMY)*sizeof(TYPE)
        CUDA_SAFE_CALL(cuLaunchKernel(kernel,
                                      gridSize_x, gridSize_y, gridSize_z,      // grid dim
                                      blockSize_x, blockSize_y, blockSize_z,   // block dim
                                      blockSize_x * dimY * sizeof(TYPE), NULL, // shared mem and stream
                                      kernel_params, 0));

        // block until the device has completed
        CUDA_SAFE_CALL(cuCtxSynchronize());

        // Since we've used a 2D scheme, there's still a "blockwise" line reduction to make on
        // the output array px_d[0] = x1B. We go from shape ( gridSize.y * nx, DIMRED ) to (nx, DIMOUT)
        CUfunction kernel_reduce;
        CUDA_SAFE_CALL(cuModuleGetFunction(&kernel_reduce, module, "reduce2D"));
        void *kernel_reduce_params[4];
        kernel_reduce_params[0] = &outB;
        kernel_reduce_params[1] = &out_d;
        kernel_reduce_params[2] = &gridSize_y;
        kernel_reduce_params[3] = &nx;

        CUDA_SAFE_CALL(cuLaunchKernel(kernel_reduce,
                                      gridSize2_x, gridSize2_y, gridSize2_z,    // grid dim
                                      blockSize2_x, blockSize2_y, blockSize2_z,   // block dim
                                      0, NULL,             // shared mem and stream
                                      kernel_reduce_params, 0));


    } else if (RR.tagRanges == 1 && tagZero == 0) {
        // ranges mode

        gridSize_x = nblocks;

        CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "GpuConv1DOnDevice_ranges"));

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
                                      gridSize_x, gridSize_y, gridSize_z,       // grid dim
                                      blockSize_x, blockSize_y, blockSize_z,    // block dim
                                      blockSize_x * dimY * sizeof(TYPE), NULL,  // shared mem and stream
                                      kernel_params, 0));                       // arguments

    } else {
        // simple mode

        gridSize_x = nx / blockSize_x + (nx % blockSize_x == 0 ? 0 : 1);

        CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "GpuConv1DOnDevice"));

        void *kernel_params[4];
        kernel_params[0] = &nx;
        kernel_params[1] = &ny;
        kernel_params[2] = &out_d;
        kernel_params[3] = &arg_d;

        CUDA_SAFE_CALL(cuLaunchKernel(kernel,
                                      gridSize_x, gridSize_y, gridSize_z,        // grid dim
                                      blockSize_x, blockSize_y, blockSize_z,     // block dim
                                      blockSize_x * dimY * sizeof(TYPE), NULL,   // shared mem and stream
                                      kernel_params, 0));                        // arguments
    }
    
    CUDA_SAFE_CALL(cuCtxSynchronize());
    
    end = clock();
    std::cout << "time for exec kernel : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
    begin = clock();

    CUDA_SAFE_CALL(cuModuleUnload(module));
    
    end = clock();
    std::cout << "time for unloading kernel : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
    begin = clock();

    // Send data from device to host.

    if (tagHostDevice == 0)
        cuMemcpyDtoH(out, (CUdeviceptr) out_d, sizeof(TYPE) * sizeout);

    end = clock();
    std::cout << "time for copying result : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
    begin = clock();
    
    //cuMemFree(p_data);
    if (RR.tagRanges == 1) {
        cuMemFree((CUdeviceptr) lookup_d);
        cuMemFree((CUdeviceptr) slices_x_d);
        cuMemFree((CUdeviceptr) ranges_y_d);
        if (SS.nbatchdims > 0)
            cuMemFree((CUdeviceptr) offsets_d);
    }
    
    end = clock();
    std::cout << "time for freeing memory : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
    begin = clock();

    B = clock();
    std::cout << "total time : " << double(B - A) / CLOCKS_PER_SEC << std::endl;

    return 0;
}


extern "C" int launch_keops_float(const char *target_file_name, int tagHostDevice, int dimY, int nx, int ny,
                                  int device_id, int tagI, int tagZero, int use_half,
                                  int tag1D2D, int dimred,
                                  int cuda_block_size, int use_chunk_mode,
                                  int *indsi, int *indsj, int *indsp,
                                  int dimout,
                                  int *dimsx, int *dimsy, int *dimsp,
                                  int **ranges, int *shapeout, float *out, float *buffer, int nargs, ...) {
    // reading arguments
    va_list ap;
    va_start(ap, nargs);
    float *arg[nargs];
    for (int i = 0; i < nargs; i++)
        arg[i] = va_arg(ap, float*);
    int *argshape[nargs];
    for (int i = 0; i < nargs; i++)
        argshape[i] = va_arg(ap, int*);
    va_end(ap);

    return launch_keops(target_file_name, tagHostDevice, dimY, nx, ny, device_id, tagI, tagZero, use_half,
                        tag1D2D, dimred,
                        cuda_block_size, use_chunk_mode,
                        indsi, indsj, indsp,
                        dimout,
                        dimsx, dimsy, dimsp,
                        ranges, shapeout, out, nargs, buffer, arg, argshape);

}





extern "C" int launch_keops_double(const char *target_file_name, int tagHostDevice, int dimY, int nx, int ny,
                                   int device_id, int tagI, int tagZero, int use_half,
                                   int tag1D2D, int dimred,
                                   int cuda_block_size, int use_chunk_mode,
                                   int *indsi, int *indsj, int *indsp,
                                   int dimout,
                                   int *dimsx, int *dimsy, int *dimsp,
                                   int **ranges, int *shapeout, double *out, float *buffer, int nargs, ...) {
    // reading arguments
    va_list ap;
    va_start(ap, nargs);
    double *arg[nargs];
    for (int i = 0; i < nargs; i++)
        arg[i] = va_arg(ap, double*);
    int *argshape[nargs];
    for (int i = 0; i < nargs; i++)
        argshape[i] = va_arg(ap, int*);
    va_end(ap);

    return launch_keops(target_file_name, tagHostDevice, dimY, nx, ny, device_id, tagI, tagZero, use_half,
                        tag1D2D, dimred,
                        cuda_block_size, use_chunk_mode,
                        indsi, indsj, indsp,
                        dimout,
                        dimsx, dimsy, dimsp,
                        ranges, shapeout, out, nargs, buffer, arg, argshape);

}




extern "C" int launch_keops_half(const char *target_file_name, int tagHostDevice, int dimY, int nx, int ny,
                                 int device_id, int tagI, int tagZero, int use_half,
                                 int tag1D2D, int dimred,
                                 int cuda_block_size, int use_chunk_mode,
                                 int *indsi, int *indsj, int *indsp,
                                 int dimout,
                                 int *dimsx, int *dimsy, int *dimsp,
                                 int **ranges, int *shapeout, half2 *out, float *buffer, int nargs, ...) {
    // reading arguments
    va_list ap;
    va_start(ap, nargs);
    half2 *arg[nargs];
    for (int i = 0; i < nargs; i++)
        arg[i] = va_arg(ap, half2*);
    int *argshape[nargs];
    for (int i = 0; i < nargs; i++)
        argshape[i] = va_arg(ap, int*);
    va_end(ap);

    return launch_keops(target_file_name, tagHostDevice, dimY, nx, ny, device_id, tagI, tagZero, use_half,
                        tag1D2D, dimred,
                        cuda_block_size, use_chunk_mode,
                        indsi, indsj, indsp,
                        dimout,
                        dimsx, dimsy, dimsp,
                        ranges, shapeout, out, nargs, buffer, arg, argshape);

}
