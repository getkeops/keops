#include <stdio.h>
#include <cuda.h>

#include "core/pack/Pack.h"
#include "core/pack/GetInds.h"
#include "core/pack/GetDims.h"
#include "core/utils/CudaErrorCheck.cu"
#include "core/utils/CudaSizes.h"

#include "core/mapreduce/GpuConv1D.h"

namespace keops {

struct GpuConv1D_FromDevice {
  template<typename TYPE, class FUN>
  static int Eval_(FUN fun, int nx, int ny, TYPE **phx_d, TYPE **phy_d, TYPE **php_d) {
    
    typedef typename FUN::DIMSX DIMSX;
    typedef typename FUN::DIMSY DIMSY;
    typedef typename FUN::DIMSP DIMSP;
    const int DIMY = DIMSY::SUM;
    const int SIZEI = DIMSX::SIZE;
    const int SIZEJ = DIMSY::SIZE;
    const int SIZEP = DIMSP::SIZE;
    
    // device arrays of pointers to device data
    TYPE **px_d, **py_d, **pp_d;
    
    // single cudaMalloc
    void **p_data;
    CudaSafeCall(cudaMalloc((void **) &p_data, sizeof(TYPE *) * (SIZEI + SIZEJ + SIZEP)));
    
    TYPE **p_data_a = (TYPE **) p_data;
    px_d = p_data_a;
    p_data_a += SIZEI;
    py_d = p_data_a;
    p_data_a += SIZEJ;
    pp_d = p_data_a;
    
    CudaSafeCall(cudaMemcpy(px_d, phx_d, SIZEI * sizeof(TYPE *), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(py_d, phy_d, SIZEJ * sizeof(TYPE *), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(pp_d, php_d, SIZEP * sizeof(TYPE *), cudaMemcpyHostToDevice));
    
    // Compute on device : grid and block are both 1d
    
    int dev = -1;
    CudaSafeCall(cudaGetDevice(&dev));
    
    SetGpuProps(dev);
    
    dim3 blockSize;
    // warning : blockSize.x was previously set to CUDA_BLOCK_SIZE; currently CUDA_BLOCK_SIZE value is used as a bound.
    blockSize.x = std::min(CUDA_BLOCK_SIZE,
                           std::min(maxThreadsPerBlock,
                                    (int) (sharedMemPerBlock / ::std::max(1, (int) (DIMY* sizeof(TYPE))))
                                   )
                          ); // number of threads in each block
    
    dim3 gridSize;
    gridSize.x = nx / blockSize.x + (nx % blockSize.x == 0 ? 0 : 1);
    
    // Size of the SharedData : blockSize.x*(DIMY)*sizeof(TYPE)
    GpuConv1DOnDevice<TYPE> << < gridSize, blockSize, blockSize.x * (DIMY) * sizeof(TYPE) >>
                                                      > (fun, nx, ny, px_d, py_d, pp_d);
    
    // block until the device has completed
    CudaSafeCall(cudaDeviceSynchronize());
    CudaCheckError();
    
    CudaSafeCall(cudaFree(p_data));
    
    return 0;
  }

// Same wrappers, but for data located on the device
  template<typename TYPE, class FUN, typename... Args>
  static int Eval(FUN fun, int nx, int ny, int device_id, TYPE *x1_d, Args... args) {
    
    // device_id is provided, so we set the GPU device accordingly
    // Warning : is has to be consistent with location of data
    CudaSafeCall(cudaSetDevice(device_id));
    
    typedef typename FUN::VARSI VARSI;
    typedef typename FUN::VARSJ VARSJ;
    typedef typename FUN::VARSP VARSP;
    
    const int SIZEI = VARSI::SIZE + 1;
    const int SIZEJ = VARSJ::SIZE;
    const int SIZEP = VARSP::SIZE;
    
    using DIMSX = GetDims<VARSI>;
    using DIMSY = GetDims<VARSJ>;
    using DIMSP = GetDims<VARSP>;
    
    using INDSI = GetInds<VARSI>;
    using INDSJ = GetInds<VARSJ>;
    using INDSP = GetInds<VARSP>;
    
    TYPE *phx_d[SIZEI];
    TYPE *phy_d[SIZEJ];
    TYPE *php_d[SIZEP];
    
    phx_d[0] = x1_d;
    
    getlist<INDSI>(phx_d + 1, args...);
    getlist<INDSJ>(phy_d, args...);
    getlist<INDSP>(php_d, args...);
    
    return Eval_(fun, nx, ny, phx_d, phy_d, php_d);
    
  }

// same without the device_id argument
  template<typename TYPE, class FUN, typename... Args>
  static int Eval(FUN fun, int nx, int ny, TYPE *x1_d, Args... args) {
    // We set the GPU device on which computations will be performed
    // to be the GPU on which data is located.
    // NB. we only check location of x1_d which is the output vector
    // so we assume that input data is on the same GPU
    // note : cudaPointerGetAttributes has a strange behaviour:
    // it looks like it makes a copy of the vector on the default GPU device (0) !!!
    // So we prefer to avoid this and provide directly the device_id as input (first function above)
    cudaPointerAttributes attributes;
    CudaSafeCall(cudaPointerGetAttributes(&attributes, x1_d));
    return Eval(fun, nx, ny, attributes.device, x1_d, args...);
  }
  
  template<typename TYPE, class FUN>
  static int Eval(FUN fun, int nx, int ny, TYPE *x1_d, TYPE **args, int device_id = -1) {
    
    if (device_id == -1) {
      // We set the GPU device on which computations will be performed
      // to be the GPU on which data is located.
      // NB. we only check location of x1_d which is the output vector
      // so we assume that input data is on the same GPU
      // note : cudaPointerGetAttributes has a strange behaviour:
      // it looks like it makes a copy of the vector on the default GPU device (0) !!!
      // So we prefer to avoid this and provide directly the device_id as input (else statement below)
      cudaPointerAttributes attributes;
      CudaSafeCall(cudaPointerGetAttributes(&attributes, x1_d));
      CudaSafeCall(cudaSetDevice(attributes.device));
    } else // device_id is provided, so we use it. Warning : is has to be consistent with location of data
      CudaSafeCall(cudaSetDevice(device_id));
    
    typedef typename FUN::VARSI VARSI;
    typedef typename FUN::VARSJ VARSJ;
    typedef typename FUN::VARSP VARSP;
    
    const int SIZEI = VARSI::SIZE + 1;
    const int SIZEJ = VARSJ::SIZE;
    const int SIZEP = VARSP::SIZE;
    
    using DIMSX = GetDims<VARSI>;
    using DIMSY = GetDims<VARSJ>;
    using DIMSP = GetDims<VARSP>;
    
    using INDSI = GetInds<VARSI>;
    using INDSJ = GetInds<VARSJ>;
    using INDSP = GetInds<VARSP>;
    
    TYPE *px_d[SIZEI];
    TYPE *py_d[SIZEJ];
    TYPE *pp_d[SIZEP];
    
    px_d[0] = x1_d;
    for (int i = 1; i < SIZEI; i++)
      px_d[i] = args[INDSI::VAL(i - 1)];
    for (int i = 0; i < SIZEJ; i++)
      py_d[i] = args[INDSJ::VAL(i)];
    for (int i = 0; i < SIZEP; i++)
      pp_d[i] = args[INDSP::VAL(i)];
    
    return Eval_(fun, nx, ny, px_d, py_d, pp_d);
    
  }
  
};

extern "C" int GpuReduc1D_FromDevice(int nx, int ny, __TYPE__ *gamma, __TYPE__ **args, int device_id = -1) {
  return Eval< F, GpuConv1D_FromDevice >::Run(nx, ny, gamma, args, device_id);
}

}
