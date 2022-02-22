
// nvcc -shared -Xcompiler -fPIC -lnvrtc -lcuda keops_nvrtc.cu -o keops_nvrtc.so
// g++ --verbose -L/opt/cuda/lib64 -L/opt/cuda/targets/x86_64-linux/lib/ -I/opt/cuda/targets/x86_64-linux/include/ -I../../include -shared -fPIC -lcuda -lnvrtc -fpermissive -DMAXIDGPU=0 -DMAXTHREADSPERBLOCK0=1024 -DSHAREDMEMPERBLOCK0=49152 -DnvrtcGetTARGET=nvrtcGetCUBIN -DnvrtcGetTARGETSize=nvrtcGetCUBINSize -DARCHTAG=\"sm\" keops_nvrtc.cpp -o keops_nvrtc.so
// g++ -std=c++11  -shared -fPIC -O3 -fpermissive -L /usr/lib -L /opt/cuda/lib64 -lcuda -lnvrtc -DnvrtcGetTARGET=nvrtcGetCUBIN -DnvrtcGetTARGETSize=nvrtcGetCUBINSize -DARCHTAG=\"sm\"  -I/home/bcharlier/projets/keops/keops/keops/include -I/opt/cuda/include -I/usr/include/python3.10/ -DMAXIDGPU=0 -DMAXTHREADSPERBLOCK0=1024 -DSHAREDMEMPERBLOCK0=49152  /home/bcharlier/projets/keops/keops/keops/binders/nvrtc/keops_nvrtc.cpp -o keops_nvrtc.cpython-310-x86_64-linux-gnu.so

#include <nvrtc.h>
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdarg.h>
#include <vector>
//#include <ctime>

#define __INDEX__ int
#define C_CONTIGUOUS 1
#define USE_HALF 0

#include "include/Sizes.h"
#include "include/Ranges.h"
#include "include/utils_pe.h"
#include "include/ranges_utils.h"


#include "include/CudaSizes.h"
#include <cuda_fp16.h>


template<typename TYPE>
class KeOps_module {
  public :

    CUdevice cuDevice;
    CUcontext ctx;
    CUmodule module;
    char *target;
    CUdeviceptr buffer;
    int nargs;

    void SetContext() {
        CUcontext current_ctx;
        CUDA_SAFE_CALL(cuCtxGetCurrent(&current_ctx));
        if (current_ctx != ctx)
            CUDA_SAFE_CALL(cuCtxPushCurrent(ctx));
        CUDA_SAFE_CALL(cuCtxGetCurrent(&current_ctx));
    }


    void Read_Target(const char *target_file_name) {
        std::ifstream rf(target_file_name, std::ifstream::binary);
        size_t targetSize;
        rf.read((char*)&targetSize, sizeof(size_t));
        target = new char[targetSize];
        rf.read(target, targetSize);
        rf.close();
    }


    KeOps_module(int device_id, int nargs_, const char *target_file_name) {

        nargs = nargs_;

        // init cuda in case not already done
        CUDA_SAFE_CALL(cuInit(0));

        // get the device and the primary context corresponding to device_id
        CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, device_id));
        CUDA_SAFE_CALL(cuDevicePrimaryCtxRetain(&ctx, cuDevice));

        // set the primary context as the active current context
        SetContext();

        // set global variables giving some properties of device
        SetGpuProps(device_id);

        // read the ptx or cubin file into a char array
        Read_Target(target_file_name);

        // load the corresponding module
        CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, target, 0, NULL, NULL));

        // allocate a small memory buffer for "on device" computation mode,
        // This is just used for storing the list of pointers to device data
        // as a device array ; it is better to allocate it here once for all,
        // otherwise allocating it at each call may cause a small overhead.
        CUDA_SAFE_CALL(cuMemAlloc(&buffer, nargs*sizeof(TYPE*)));

    }


    ~KeOps_module() {
        SetContext();
        CUDA_SAFE_CALL(cuMemFree(buffer));
        CUDA_SAFE_CALL(cuModuleUnload(module));
        CUDA_SAFE_CALL(cuDevicePrimaryCtxRelease(cuDevice));
        delete[] target;
    }

    int launch_kernel (int tagHostDevice, int dimY, int nx, int ny,
                       int tagI, int tagZero, int use_half,
                       int tag1D2D, int dimred,
                       int cuda_block_size, int use_chunk_mode,
                       std::vector< int > indsi, std::vector< int > indsj, std::vector< int > indsp,
                       int dimout,
                       int *dimsx, int *dimsy, int *dimsp,
                       int **ranges,
                       int *shapeout, TYPE *out,
                       TYPE **arg,
                       int **argshape
                      ) {


        //std::cout << nx << " " << ny << " " << dimsy[0] << " " << argshape[0][0] << " " << argshape[0][1]<< " " << argshape_ptr_v[1][1]<< std::endl;

        //clock_t start, end, start_, //end_;
        //start_ = start = clock();

        ////std::cout << "Entering launch_keops inner" << std::endl;

        //int **ranges = (int**) ranges_v.data();
        //TYPE **arg = (TYPE**) arg_v.data();
        //int **argshape = (int**) argshape_v.data();
        //TYPE *out = (TYPE*) out_void;

        ////end_ = clock();
        ////std::cout << "  time for converting std::vector : " << double(//end_ - start_) / CLOCKS_PER_SEC << std::endl;
        //start_ = clock();

        /*------------------------------------*/
        /*      End cast input args           */
        /*------------------------------------*/

        SetContext();

        ////end_ = clock();
        ////std::cout << "  time for set device : " << double(//end_ - start_) / CLOCKS_PER_SEC << std::endl;
        //start_ = clock();

        Sizes<TYPE> SS(nargs, arg, argshape, nx, ny,
                       tagI, use_half,
                       dimout,
                       indsi, indsj, indsp,
                       dimsx, dimsy, dimsp);

        //end_ = clock();
        //std::cout << "  time for Sizes : " << double(//end_ - start_) / CLOCKS_PER_SEC << std::endl;
        //start_ = clock();

        if (use_half)
            SS.switch_to_half2_indexing();

        Ranges<TYPE> RR(SS, ranges);
        nx = SS.nx;
        ny = SS.ny;

        //end_ = clock();
        //std::cout << "  time for Ranges : " << double(//end_ - start_) / CLOCKS_PER_SEC << std::endl;
        //start_ = clock();

        // now we switch (back...) indsi, indsj and dimsx, dimsy in case tagI=1.
        // This is to be consistent with the convention used in the old
        // bindings where i and j variables had different meanings in bindings
        // and in the core code. Clearly we could do better if we
        // carefully rewrite some parts of the code
        if (tagI == 1) {
            std::vector< int > tmp;
            tmp = indsj;
            indsj = indsi;
            indsi = tmp;

            int *tmp2;
            tmp2 = dimsy;
            dimsy = dimsx;
            dimsx = tmp2;
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

        if (RR.tagRanges==1) {
            if (tagHostDevice==1) {
                range_preprocess_from_device(nblocks, tagI, RR.nranges_x, RR.nranges_y, RR.castedranges,
                                             SS.nbatchdims, slices_x_d, ranges_y_d, lookup_d,
                                             offsets_d,
                                             blockSize_x, indsi, indsj, indsp, SS.shapes);
            } else { // tagHostDevice==0
                range_preprocess_from_host(nblocks, tagI, RR.nranges_x, RR.nranges_y, RR.nredranges_x, RR.nredranges_y, RR.castedranges,
                                           SS.nbatchdims, slices_x_d, ranges_y_d, lookup_d,
                                           offsets_d,
                                           blockSize_x, indsi, indsj, indsp, SS.shapes);
            }
        }

        ////end_ = clock();
        ////std::cout << "  time for interm : " << double(//end_ - start_) / CLOCKS_PER_SEC << std::endl;
        //start_ = clock();

        CUdeviceptr p_data;
        TYPE *out_d;
        TYPE **arg_d;
        int sizeout = get_sum(shapeout);

        if (tagHostDevice == 1) {
            p_data = buffer;
            load_args_FromDevice(p_data, out, out_d, nargs, arg, arg_d);
        } else
            load_args_FromHost(p_data, out, out_d, nargs, arg, arg_d, argshape, sizeout);

        ////end_ = clock();
        ////std::cout << "  time for load_args : " << double(//end_ - start_) / CLOCKS_PER_SEC << std::endl;
        //start_ = clock();

        CUfunction kernel;

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
            // std::cout << "GpuConv1DOnDevice_ranges " << nx << " " << gridSize_x ;
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

            //std::cout << "GpuConv1DOnDevice " << nx << " " << gridSize_x ;//<< " " << gridSize_y << " " << gridSize_z << " " <<  blockSize_x << " " << blockSize_y << " " <<  blockSize_z << " " <<   blockSize_x * dimY * sizeof(TYPE)  << std::endl;

            CUDA_SAFE_CALL(cuLaunchKernel(kernel,
                                          gridSize_x, gridSize_y, gridSize_z,        // grid dim
                                          blockSize_x, blockSize_y, blockSize_z,     // block dim
                                          blockSize_x * dimY * sizeof(TYPE), NULL,   // shared mem and stream
                                          kernel_params, 0));                        // arguments
        }

        CUDA_SAFE_CALL(cuCtxSynchronize());

        ////end_ = clock();
        ////std::cout << "  time for kernel : " << double(//end_ - start_) / CLOCKS_PER_SEC << std::endl;
        //start_ = clock();

        // Send data from device to host.

        if (tagHostDevice == 0) {
            cuMemcpyDtoH(out, (CUdeviceptr) out_d, sizeof(TYPE) * sizeout);
            cuMemFree(p_data);
        }
        if (RR.tagRanges == 1) {
            cuMemFree((CUdeviceptr) lookup_d);
            cuMemFree((CUdeviceptr) slices_x_d);
            cuMemFree((CUdeviceptr) ranges_y_d);
            if (SS.nbatchdims > 0)
                cuMemFree((CUdeviceptr) offsets_d);
        }

        //end_ = end = clock();
        ////std::cout << "  time for last part : " << double(//end_ - start_) / CLOCKS_PER_SEC << std::endl;
        ////std::cout << "time for launch_keops inner : " << double(end - start) / CLOCKS_PER_SEC << std::endl;

        return 0;
    }

};


template class KeOps_module<float>;
template class KeOps_module<double>;
template class KeOps_module<half2>;
