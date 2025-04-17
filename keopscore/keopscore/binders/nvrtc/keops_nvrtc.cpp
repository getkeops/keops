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
#include <numeric>
#include <nvrtc.h>
#include <sstream>
#include <stdarg.h>
#include <stdio.h>
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

// ------------------------------------------------------------------------
// DevicePointer wrapper
// Allocates memory on the device and links it to a boolean flag indicating if it was allocated by KeOps.
struct DevicePointer {
  signed long int* ptr;
  bool allocated_by_keops;

  DevicePointer() : ptr(nullptr), allocated_by_keops(false) {}

  // Allocates memory on the device and marks the pointer as owned.
  void allocate(size_t size) {
    CUDA_SAFE_CALL(cuMemAlloc((CUdeviceptr *)&ptr, size));
    allocated_by_keops = true;
  }

  // Copies data from host memory to the allocated device memory.
  void copyFromHost(const void *host_data, size_t size) {
    CUDA_SAFE_CALL(cuMemcpyHtoD((CUdeviceptr)ptr, host_data, size));
  }

  // Sets the pointer from an externally allocated device pointer.
  void setFromDevice(void *dev_ptr) {
    ptr = (signed long int*)dev_ptr;
    allocated_by_keops = false;
  }

  // Frees the memory only if it was allocated by this wrapper.
  void destroy() {
    if (allocated_by_keops && ptr != nullptr) {
      CUDA_SAFE_CALL_NO_EXCEPTION(cuMemFree((CUdeviceptr)ptr));
      ptr = nullptr;
    }
  }
};
// End DevicePointer wrapper
// ------------------------------------------------------------------------

signed long int *build_offset_tables(int nbatchdims, signed long int *shapes, signed long int nblocks,
                            signed long int *lookup_h, const std::vector<int> &indsi,
                            const std::vector<int> &indsj,
                            const std::vector<int> &indsp, int tagJ) {

  int sizei = indsi.size();
  int sizej = indsj.size();
  int sizep = indsp.size();

  // Support for broadcasting over batch dimensions
  // =============================================

  int sizevars = sizei + sizej + sizep;

  // Separate and store the shapes of the "i" and "j" variables + parameters
  // --------------
  //
  // shapes is an array of size (1+nargs)*(nbatchdims+3), which looks like:
  // [ A, .., B, M, N, D_out]  -> output
  // [ A, .., B, M, 1, D_1  ]  -> "i" variable
  // [ A, .., B, 1, N, D_2  ]  -> "j" variable
  // [ A, .., B, 1, 1, D_3  ]  -> "parameter"
  // [ A, .., 1, M, 1, D_4  ]  -> N.B.: we support broadcasting on the batch
  // dimensions! [ 1, .., 1, M, 1, D_5  ]  ->      (we'll just ask users to fill
  // in the shapes with *explicit* ones)

  std::vector<signed long int> shapes_i_vec(sizei * (nbatchdims + 1));
  signed long int *shapes_i = shapes_i_vec.data();
  std::vector<signed long int> shapes_j_vec(sizej * (nbatchdims + 1));
  signed long int *shapes_j = shapes_j_vec.data();
  std::vector<signed long int> shapes_p_vec(sizep * (nbatchdims + 1));
  signed long int *shapes_p = shapes_p_vec.data();

  // First, we fill shapes_i with the "relevant" shapes of the "i" variables,
  // making it look like, say:
  // [ A, .., B, M]
  // [ A, .., 1, M]
  // [ A, .., A, M]
  // Then, we do the same for shapes_j, but with "N" instead of "M".
  // And finally for the parameters, with "1" instead of "M".
  fill_shapes(nbatchdims, shapes, shapes_i, shapes_j, shapes_p, tagJ, indsi,
              indsj, indsp);

  int tagIJ =
      tagJ; // 1 if the reduction is made "over j", 0 if it is made "over i"
  signed long int M = shapes[nbatchdims], N = shapes[nbatchdims + 1];

  // We create a lookup table, "offsets", of shape (nblocks, SIZEVARS) --------
  signed long int *offsets_d = NULL;

  std::vector<signed long int> offsets_h_vec(nblocks * sizevars);
  signed long int *offsets_h = offsets_h_vec.data();

  for (signed long int k = 0; k < nblocks; k++) {
    signed long int range_id = (signed long int)lookup_h[3 * k];
    signed long int start_x = tagIJ ? range_id * M : range_id * N;
    signed long int start_y = tagIJ ? range_id * N : range_id * M;

    signed long int patch_offset = (signed long int)(lookup_h[3 * k + 1] - start_x);

    vect_broadcast_index(start_x, nbatchdims, sizei, shapes, shapes_i,
                         offsets_h + k * sizevars, patch_offset);
    vect_broadcast_index(start_y, nbatchdims, sizej, shapes, shapes_j,
                         offsets_h + k * sizevars + sizei);
    vect_broadcast_index(range_id, nbatchdims, sizep, shapes, shapes_p,
                         offsets_h + k * sizevars + sizei + sizej);
  }

  CUDA_SAFE_CALL(cuMemAlloc((CUdeviceptr *)&offsets_d,
                            sizeof(signed long int) * nblocks * sizevars));
  CUDA_SAFE_CALL(cuMemcpyHtoD((CUdeviceptr)offsets_d, offsets_h,
                              sizeof(signed long int) * nblocks * sizevars));

  return offsets_d;
}

// ------------------------------------------------------------------------
void range_preprocess_from_device(signed long int &nblocks, int tagI, signed long int nranges_x,
                                  signed long int nranges_y, signed long int **castedranges,
                                  int nbatchdims, DevicePointer &slices_x_d,
                                  DevicePointer &ranges_y_d, DevicePointer &lookup_d,
                                  DevicePointer &offsets_d, signed long int blockSize_x,
                                  const std::vector<int> &indsi,
                                  const std::vector<int> &indsj,
                                  const std::vector<int> &indsp,
                                  signed long int *shapes) {

  // Ranges pre-processing...
  // ==================================================================

  // N.B.: In the following code, we assume that the x-ranges do not overlap.
  //       Otherwise, we'd have to assume that DIMRED == DIMOUT
  //       or allocate a buffer of size nx * DIMRED. This may be done in the
  //       future.
  // Cf. reduction.h:
  //    FUN::tagJ = 1 for a reduction over j, result indexed by i
  //    FUN::tagJ = 0 for a reduction over i, result indexed by j

  int tagJ = 1 - tagI;
  signed long int nranges = tagJ ? nranges_x : nranges_y;

  signed long int *ranges_x = tagJ ? castedranges[0] : castedranges[3];
  signed long int *slices_x = tagJ ? castedranges[1] : castedranges[4];
  signed long int *ranges_y = tagJ ? castedranges[2] : castedranges[5];

  std::vector<signed long int> ranges_x_h_arr_vec(2 * nranges);
  signed long int *ranges_x_h_arr = ranges_x_h_arr_vec.data();
  signed long int *ranges_x_h;

  // The code below needs a pointer to ranges_x on *host* memory,
  // ------------------- as well as pointers to slices_x and ranges_y on
  // *device* memory.
  // -> Depending on the "ranges" location, we'll copy ranges_x *or* slices_x
  // and ranges_y
  //    to the appropriate memory:
  bool ranges_on_device = (nbatchdims == 0);
  // N.B.: We only support Host ranges with Device data when these ranges were
  // created
  //       to emulate block-sparse reductions.

  if (ranges_on_device) { // Ranges already on device.
    ranges_x_h = &ranges_x_h_arr[0];
    cuMemcpyDtoH(ranges_x_h, (CUdeviceptr)ranges_x, sizeof(signed long int) * 2 * nranges);
    // Set the pointers as borrowed.
    slices_x_d.setFromDevice(slices_x);
    ranges_y_d.setFromDevice(ranges_y);
  } else { // Ranges on host; allocate device memory.
    ranges_x_h = ranges_x;
    slices_x_d.allocate(sizeof(signed long int) * nranges);
    slices_x_d.copyFromHost(slices_x, sizeof(signed long int) * nranges);

    ranges_y_d.allocate(sizeof(signed long int) * 2 * nranges);
    ranges_y_d.copyFromHost(ranges_y, sizeof(signed long int) * 2 * nranges);
  }

  // Compute number of blocks needed.
  // ---------------------------------------------
  nblocks = 0;
  signed long int len_range = 0;
  for (signed long int i = 0; i < nranges; i++) {
    len_range = ranges_x_h[2 * i + 1] - ranges_x_h[2 * i];
    nblocks += (len_range / blockSize_x) + (len_range % blockSize_x == 0 ? 0 : 1);
  }

  // Create a lookup table for the blocks.
  // ---------------------------------------------
  std::vector<signed long int> lookup_h_vec(3 * nblocks);
  signed long int *lookup_h = lookup_h_vec.data();
  signed long int index = 0;
  for (signed long int i = 0; i < nranges; i++) {
    len_range = ranges_x_h[2 * i + 1] - ranges_x_h[2 * i];
    for (signed long int j = 0; j < len_range; j += blockSize_x) {
      lookup_h[3 * index] = i;
      lookup_h[3 * index + 1] = ranges_x_h[2 * i] + j;
      lookup_h[3 * index + 2] = ranges_x_h[2 * i] + j + std::min((signed long int)blockSize_x, len_range - j);
      index++;
    }
  }

  // Allocate and copy lookup table.
  // -----------------------------------------------------
  lookup_d.allocate(sizeof(signed long int) * 3 * nblocks);
  lookup_d.copyFromHost(lookup_h, sizeof(signed long int) * 3 * nblocks);

    // Support for broadcasting over batch dimensions
  // =============================================

  // We create a lookup table, "offsets", of shape (nblock, SIZEVARS):
  if (nbatchdims > 0) {
    offsets_d.ptr = build_offset_tables(nbatchdims, shapes, nblocks, lookup_h,
                                        indsi, indsj, indsp, tagJ);
    offsets_d.allocated_by_keops = true;
  }
}

void range_preprocess_from_host(signed long int &nblocks, int tagI, signed long int nranges_x,
                                signed long int nranges_y, signed long int nredranges_x,
                                signed long int nredranges_y, signed long int **castedranges,
                                int nbatchdims, DevicePointer &slices_x_d,
                                DevicePointer &ranges_y_d, DevicePointer &lookup_d,
                                DevicePointer &offsets_d, signed long int blockSize_x,
                                const std::vector<int> &indsi,
                                const std::vector<int> &indsj,
                                const std::vector<int> &indsp, signed long int *shapes) {

  // Ranges pre-processing...
  // ==================================================================
  
  // N.B.: In the following code, we assume that the x-ranges do not overlap.
  //       Otherwise, we'd have to assume that DIMRED == DIMOUT
  //       or allocate a buffer of size nx * DIMRED. This may be done in the
  //       future.
  // Cf. reduction.h:
  //    FUN::tagJ = 1 for a reduction over j, result indexed by i
  //    FUN::tagJ = 0 for a reduction over i, result indexed by j

  int tagJ = 1 - tagI;
  signed long int nranges = tagJ ? nranges_x : nranges_y;
  signed long int nredranges = tagJ ? nredranges_y : nredranges_x;

  signed long int *ranges_x = tagJ ? castedranges[0] : castedranges[3];
  signed long int *slices_x = tagJ ? castedranges[1] : castedranges[4];
  signed long int *ranges_y = tagJ ? castedranges[2] : castedranges[5];

  // Computes the number of blocks needed
  // ---------------------------------------------
  nblocks = 0;
  signed long int len_range = 0;
  for (signed long int i = 0; i < nranges; i++) {
    len_range = ranges_x[2 * i + 1] - ranges_x[2 * i];
    nblocks += (len_range / blockSize_x) + (len_range % blockSize_x == 0 ? 0 : 1);
  }

  // Create a lookup table for the blocks
  // --------------------------------------------
  std::vector<signed long int> lookup_h_vec(3 * nblocks);
  signed long int *lookup_h = lookup_h_vec.data();
  signed long int index = 0;
  for (signed long int i = 0; i < nranges; i++) {
    len_range = ranges_x[2 * i + 1] - ranges_x[2 * i];
    for (signed long int j = 0; j < len_range; j += blockSize_x) {
      lookup_h[3 * index] = i;
      lookup_h[3 * index + 1] = ranges_x[2 * i] + j;
      lookup_h[3 * index + 2] = ranges_x[2 * i] + j + std::min((signed long int)blockSize_x, len_range - j);
      index++;
    }
  }

  // Allocate and copy lookup table.
  // --------------------------------------------
  lookup_d.allocate(sizeof(signed long int) * 3 * nblocks);
  lookup_d.copyFromHost(lookup_h, sizeof(signed long int) * 3 * nblocks);

  // Allocate and copy slices_x_d.
  slices_x_d.allocate(sizeof(signed long int) * 2 * nranges);
  slices_x_d.copyFromHost(slices_x, sizeof(signed long int) * 2 * nranges);

  // Allocate and copy ranges_y_d.
  ranges_y_d.allocate(sizeof(signed long int) * 2 * nredranges);
  ranges_y_d.copyFromHost(ranges_y, sizeof(signed long int) * 2 * nredranges);

  // Support for broadcasting over batch dimensions
  // =============================================

  // We create a lookup table, "offsets", of shape (nblock, SIZEVARS):
  if (nbatchdims > 0) {
    offsets_d.ptr = build_offset_tables(nbatchdims, shapes, nblocks, lookup_h,
                                        indsi, indsj, indsp, tagJ);
    offsets_d.allocated_by_keops = true;
  }
}
// ------------------------------------------------------------------------

template <typename TYPE> class KeOps_module {
public:
  CUdevice cuDevice;
  CUcontext ctx;
  CUmodule module;
  char *target;
  CUdeviceptr buffer;
  int nargs;

  void SetContext() {
    CUcontext current_ctx;
    CUDA_SAFE_CALL_NO_EXCEPTION(cuCtxGetCurrent(&current_ctx));
    if (current_ctx != ctx)
      CUDA_SAFE_CALL_NO_EXCEPTION(cuCtxPushCurrent(ctx));
    CUDA_SAFE_CALL_NO_EXCEPTION(cuCtxGetCurrent(&current_ctx));
  }

  void Read_Target(const char *target_file_name) {
    std::ifstream rf(target_file_name, std::ifstream::binary);
    signed long int targetSize;
    rf.read((char *)&targetSize, sizeof(signed long int));
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
    // This is used for storing a list of pointers to device data.
    CUDA_SAFE_CALL(cuMemAlloc(&buffer, nargs * sizeof(TYPE *)));
  }

  ~KeOps_module() {
    SetContext();
    CUDA_SAFE_CALL_NO_EXCEPTION(cuMemFree(buffer));
    CUDA_SAFE_CALL_NO_EXCEPTION(cuModuleUnload(module));
    CUDA_SAFE_CALL_NO_EXCEPTION(cuDevicePrimaryCtxRelease(cuDevice));
    delete[] target;
  }

  int launch_kernel(int tagHostDevice, signed long int dimY, signed long int nx, signed long int ny,
                    int tagI, int tagZero, int use_half, int tag1D2D,
                    signed long int dimred, signed long int cuda_block_size, int use_chunk_mode,
                    std::vector<int> indsi, std::vector<int> indsj,
                    std::vector<int> indsp, signed long int dimout,
                    std::vector<signed long int> dimsx, std::vector<signed long int> dimsy,
                    std::vector<signed long int> dimsp, signed long int **ranges,
                    std::vector<signed long int> shapeout, TYPE *out, TYPE **arg,
                    std::vector<std::vector<signed long int>> argshape) {

    SetContext();

    ////end_ = clock();
    ////std::cout << "  time for set device : " << double(//end_ - start_) /
    /// CLOCKS_PER_SEC << std::endl;
    // start_ = clock();

    Sizes<TYPE> SS(nargs, arg, argshape, nx, ny, tagI, use_half, dimout, indsi,
                   indsj, indsp, dimsx, dimsy, dimsp);

    // end_ = clock();
    // std::cout << "  time for Sizes : " << double(//end_ - start_) /
    // CLOCKS_PER_SEC << std::endl; start_ = clock();

    if (use_half)
      SS.switch_to_half2_indexing();

    Ranges<TYPE> RR(SS, ranges);
    nx = SS.nx;
    ny = SS.ny;

    // end_ = clock();
    // std::cout << "  time for Ranges : " << double(//end_ - start_) /
    // CLOCKS_PER_SEC << std::endl; start_ = clock();

    // now we switch (back...) indsi, indsj and dimsx, dimsy in case tagI=1.
    // This is to be consistent with the convention used in the old
    // bindings where i and j variables had different meanings in bindings
    // and in the core code. Clearly we could do better if we
    // carefully rewrite some parts of the code

    if (tagI == 1) {
      std::vector<int> tmpind = indsj;
      indsj = indsi;
      indsi = tmpind;

      std::vector<signed long int> tmpdim = dimsy;
      dimsy = dimsx;
      dimsx = tmpdim;
    }

    unsigned int blockSize_x = 1, blockSize_y = 1, blockSize_z = 1;
    if (use_chunk_mode == 0) {
      // warning : blockSize.x was previously set to CUDA_BLOCK_SIZE; currently
      // CUDA_BLOCK_SIZE value is used as a bound.
      blockSize_x = std::min(
          cuda_block_size,
          std::min((signed long int)maxThreadsPerBlock,
                   (signed long int)(sharedMemPerBlock /
                     std::max((signed long int)1, (signed long int)(dimY * sizeof(TYPE))))));
    } else {
      // warning : the value here must match the one which is set in file
      // GpuReduc1D_chunks.py, line 59 and file GpuReduc1D_finalchunks.py, line
      // 67
      blockSize_x = std::min((signed long int)cuda_block_size,
                             std::min((signed long int)1024,
                                      (signed long int)(49152 /
                                        std::max((signed long int)1, (signed long int)(dimY * sizeof(TYPE))))));
    }

    signed long int nblocks;
    if (tagI == 1) {
      signed long int tmp = ny;
      ny = nx;
      nx = tmp;
    }

    // Declare DevicePointer wrappers.
    DevicePointer lookup_d, slices_x_d, ranges_y_d, offsets_d;
    if (RR.tagRanges == 1) {
      if (tagHostDevice == 1) {
        range_preprocess_from_device(
            nblocks, tagI, RR.nranges_x, RR.nranges_y, RR.castedranges,
            SS.nbatchdims, slices_x_d, ranges_y_d, lookup_d, offsets_d,
            blockSize_x, indsi, indsj, indsp, SS.shapes);
      } else { // Host mode
        range_preprocess_from_host(nblocks, tagI, RR.nranges_x, RR.nranges_y,
                                   RR.nredranges_x, RR.nredranges_y, RR.castedranges,
                                   SS.nbatchdims, slices_x_d, ranges_y_d, lookup_d, offsets_d,
                                   blockSize_x, indsi, indsj, indsp, SS.shapes);
      }
    }

    ////end_ = clock();
    ////std::cout << "  time for interm : " << double(//end_ - start_) /
    /// CLOCKS_PER_SEC << std::endl;
    // start_ = clock();

    CUdeviceptr p_data;
    TYPE *out_d;
    TYPE **arg_d;
    signed long int sizeout = std::accumulate(shapeout.begin(), shapeout.end(), 1,
                                std::multiplies<signed long int>());
    if (tagHostDevice == 1) {
      p_data = buffer;
      load_args_FromDevice(p_data, out, out_d, nargs, arg, arg_d);
    } else {
      load_args_FromHost(p_data, out, out_d, nargs, arg, arg_d, argshape, sizeout);
    }

    ////end_ = clock();
    ////std::cout << "  time for load_args : " << double(//end_ - start_) /
    /// CLOCKS_PER_SEC << std::endl;
    // start_ = clock();

    CUfunction kernel;

    unsigned int gridSize_x = 1, gridSize_y = 1, gridSize_z = 1;

    if (tag1D2D == 1) { // 2D scheme

      gridSize_x = nx / blockSize_x + (nx % blockSize_x == 0 ? 0 : 1);
      gridSize_y = ny / blockSize_x + (ny % blockSize_x == 0 ? 0 : 1);

      // Reduce : grid and block are both 1d
      unsigned int blockSize2_x = 1, blockSize2_y = 1, blockSize2_z = 1;
      blockSize2_x = blockSize_x; // number of threads in each block
      unsigned int gridSize2_x = 1, gridSize2_y = 1, gridSize2_z = 1;
      gridSize2_x = (nx * dimred) / blockSize2_x +
                    ((nx * dimred) % blockSize2_x == 0 ? 0 : 1);

      
      // Data on the device. We need an "inflated" outB, which contains
      // gridSize.y "copies" of out that will be reduced in the final pass.

      TYPE *outB;

      // single cudaMalloc
      CUdeviceptr p_data_outB;
      CUDA_SAFE_CALL(cuMemAlloc(&p_data_outB, sizeof(TYPE) * (nx * dimred * gridSize_y)));
      outB = (TYPE *)((TYPE **)p_data_outB);

      CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "GpuConv2DOnDevice"));

      void *kernel_params[4];
      kernel_params[0] = &nx;
      kernel_params[1] = &ny;
      kernel_params[2] = &outB;
      kernel_params[3] = &arg_d;

      // Size of the SharedData : blockSize.x*(DIMY)*sizeof(TYPE)

      CUDA_SAFE_CALL(cuLaunchKernel(
          kernel, gridSize_x, gridSize_y, gridSize_z,  // grid dim
          blockSize_x, blockSize_y, blockSize_z,      // block dim
          blockSize_x * dimY * sizeof(TYPE), NULL,    // shared mem and stream
          kernel_params, 0));
      // block until the device has completed
      CUDA_SAFE_CALL(cuCtxSynchronize());
      
      // Since we've used a 2D scheme, there's still a "blockwise" line
      // reduction to make on the output array px_d[0] = x1B. We go from shape (
      // gridSize.y * nx, DIMRED ) to (nx, DIMOUT)
      CUfunction kernel_reduce;
      CUDA_SAFE_CALL(cuModuleGetFunction(&kernel_reduce, module, "reduce2D"));
      void *kernel_reduce_params[4];
      kernel_reduce_params[0] = &outB;
      kernel_reduce_params[1] = &out_d;
      kernel_reduce_params[2] = &gridSize_y;
      kernel_reduce_params[3] = &nx;

      CUDA_SAFE_CALL(cuLaunchKernel(
          kernel_reduce, gridSize2_x, 1, 1,   // grid dim
          blockSize2_x, 1, 1,                 // block dim
          0, NULL, kernel_reduce_params, 0)); // shared mem and stream

      CUDA_SAFE_CALL(cuMemFree(p_data_outB));

    } else if (RR.tagRanges == 1 && tagZero == 0) {
      // ranges mode

      gridSize_x = nblocks;


      CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "GpuConv1DOnDevice_ranges"));
      void *kernel_params[9];
      kernel_params[0] = &nx;
      kernel_params[1] = &ny;
      kernel_params[2] = &SS.nbatchdims;
      CUdeviceptr offsets_ptr = (CUdeviceptr)(offsets_d.ptr);
      CUdeviceptr lookup_ptr  = (CUdeviceptr)(lookup_d.ptr);
      CUdeviceptr slices_ptr  = (CUdeviceptr)(slices_x_d.ptr);
      CUdeviceptr ranges_ptr  = (CUdeviceptr)(ranges_y_d.ptr);
      kernel_params[3] = &offsets_ptr;
      kernel_params[4] = &lookup_ptr;
      kernel_params[5] = &slices_ptr;
      kernel_params[6] = &ranges_ptr;
      kernel_params[7] = &out_d;
      kernel_params[8] = &arg_d;


      CUDA_SAFE_CALL(cuLaunchKernel(
          kernel, gridSize_x, gridSize_y, gridSize_z,     // grid dim
          blockSize_x, blockSize_y, blockSize_z,          // block dim
          blockSize_x * dimY * sizeof(TYPE), NULL,        // shared mem and stream
          kernel_params, 0));                             // arguments
    } else {
      // simple mode

      gridSize_x = nx / blockSize_x + (nx % blockSize_x == 0 ? 0 : 1);

      CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "GpuConv1DOnDevice"));

      void *kernel_params[4];
      kernel_params[0] = &nx;
      kernel_params[1] = &ny;
      kernel_params[2] = &out_d;
      kernel_params[3] = &arg_d;

      CUDA_SAFE_CALL(cuLaunchKernel(
          kernel, gridSize_x, gridSize_y, gridSize_z,
          blockSize_x, blockSize_y, blockSize_z,
          blockSize_x * dimY * sizeof(TYPE), NULL,
          kernel_params, 0));
    }

    CUDA_SAFE_CALL(cuCtxSynchronize());

    ////end_ = clock();
    ////std::cout << "  time for kernel : " << double(//end_ - start_) /
    /// CLOCKS_PER_SEC << std::endl;
    // start_ = clock();

    // Send data from device to host.

    if (tagHostDevice == 0) {
      CUDA_SAFE_CALL(cuMemcpyDtoH(out, (CUdeviceptr)out_d, sizeof(TYPE) * sizeout));
    }

    if (tagHostDevice == 0)
      CUDA_SAFE_CALL(cuMemFree(p_data));

    // Cleanup: free only the memory allocated by KeOps.
    if (RR.tagRanges == 1) {
      lookup_d.destroy();
      slices_x_d.destroy();
      ranges_y_d.destroy();
      if (SS.nbatchdims > 0) {
        offsets_d.destroy();
      }
    }

    return 0;
  }
};

template class KeOps_module<float>;
template class KeOps_module<double>;
template class KeOps_module<half2>;
