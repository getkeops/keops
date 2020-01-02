#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <cuda.h>

#include "core/pack/Pack.h"
#include "core/pack/GetInds.h"
#include "core/pack/GetDims.h"
#include "core/mapreduce/broadcast_batch_dimensions.h"
#include "core/utils/CudaErrorCheck.cu"
#include "core/mapreduce/GpuConv1D_ranges.h"

namespace keops {

struct GpuConv1D_ranges_FromHost {
  
  template < typename TYPE, class FUN >
  static int Eval_(FUN fun, int nx, int ny,
                   int nbatchdims, int *shapes,
                   int nranges_x, int nranges_y, int nredranges_x, int nredranges_y, __INDEX__ **ranges,
                   TYPE** px_h, TYPE** py_h, TYPE** pp_h) {
    
    typedef typename FUN::DIMSX DIMSX;
    typedef typename FUN::DIMSY DIMSY;
    typedef typename FUN::DIMSP DIMSP;
    const int DIMY = DIMSY::SUM;
    const int DIMOUT = FUN::DIM; // dimension of output variable
    const int SIZEI = DIMSX::SIZE;
    const int SIZEJ = DIMSY::SIZE;
    const int SIZEP = DIMSP::SIZE;
    
    
    
    // Compute the memory footprints of all (broadcasted?) variables ===========
    typedef typename FUN::VARSI VARSI;
    typedef typename FUN::VARSJ VARSJ;
    typedef typename FUN::VARSP VARSP;
    
    const int SIZEVARSI = VARSI::SIZE+1;  // The usual convention is that the output "counts" in SIZEI
    const int SIZEVARSJ = VARSJ::SIZE;
    const int SIZEVARSP = VARSP::SIZE;
    
    static_assert(SIZEVARSI == SIZEI, "[Jean:] Looks like I misunderstood something... Sorry.");
    static_assert(SIZEVARSJ == SIZEJ, "[Jean:] Looks like I misunderstood something... Sorry.");
    static_assert(SIZEVARSP == SIZEP, "[Jean:] Looks like I misunderstood something... Sorry.");
    
    // Separate and store the shapes of the "i" and "j" variables + parameters --------------
    //
    // shapes is an array of size (1+nargs)*(nbatchdims+3), which looks like:
    // [ A, .., B, M, N, D_out]  -> output
    // [ A, .., B, M, 1, D_1  ]  -> "i" variable
    // [ A, .., B, 1, N, D_2  ]  -> "j" variable
    // [ A, .., B, 1, 1, D_3  ]  -> "parameter"
    // [ A, .., 1, M, 1, D_4  ]  -> N.B.: we support broadcasting on the batch dimensions!
    // [ 1, .., 1, M, 1, D_5  ]  ->      (we'll just ask users to fill in the shapes with *explicit* ones)
    
    int shapes_i[(SIZEVARSI-1)*(nbatchdims+1)], shapes_j[SIZEVARSJ*(nbatchdims+1)], shapes_p[SIZEVARSP*(nbatchdims+1)];
    
    // First, we fill shapes_i with the "relevant" shapes of the "i" variables,
    // making it look like, say:
    // [ A, .., B, M]
    // [ A, .., 1, M]
    // [ A, .., A, M]
    // Then, we do the same for shapes_j, but with "N" instead of "M".
    // And finally for the parameters, with "1" instead of "M".
    fill_shapes<FUN>(nbatchdims, shapes, shapes_i, shapes_j, shapes_p);
    
    int total_footprint_x = 0, total_footprint_y = 0, total_footprint_p = 0;
    int footprints_x[SIZEI], footprints_y[SIZEJ], footprints_p[SIZEP];
    int tmp = 0;
    
    // Footprints of the "x" variables: ----------------------------------------
    footprints_x[0] = nx * DIMOUT;  // First "x variable" is the output
    total_footprint_x = footprints_x[0];
    for (int k=1; k < SIZEI; k++) { // For the actual variables:
      tmp = DIMSX::VAL(k);  // use the product of the vector dimension...
      for (int l=0; l < nbatchdims+1; l++) {
        tmp *= shapes_i[ (k-1)*(nbatchdims+1) + l];  // with all the shape's dimensions
      }
      footprints_x[k] = tmp;
      total_footprint_x += tmp;
    }
    
    // Footprints of the "y" variables: ----------------------------------------
    for (int k=0; k < SIZEJ; k++) { // For the actual variables:
      tmp = DIMSY::VAL(k);  // use the product of the vector dimension...
      for (int l=0; l < nbatchdims+1; l++) {
        tmp *= shapes_j[ k*(nbatchdims+1) + l];  // with all the shape's dimensions
      }
      footprints_y[k] = tmp;
      total_footprint_y += tmp;
    }
    
    // Footprints of the "parameters": -----------------------------------------
    for (int k=0; k < SIZEP; k++) { // For the actual variables:
      tmp = DIMSP::VAL(k);  // use the product of the vector dimension...
      for (int l=0; l < nbatchdims+1; l++) {
        tmp *= shapes_p[ k*(nbatchdims+1) + l];  // with all the shape's dimensions
      }
      footprints_p[k] = tmp;
      total_footprint_p += tmp;
    }
    
    
    // Load data on the device =================================================
    
    // Setup pointers, allocate memory -----------------------------------------
    // pointers to device data
    TYPE *x_d, *y_d, *param_d;
    // device arrays of pointers to device data
    TYPE **px_d, **py_d, **pp_d;
    
    // single cudaMalloc
    void **p_data;
    CudaSafeCall(cudaMalloc((void**)&p_data,
                            sizeof(TYPE*) * (SIZEI+SIZEJ+SIZEP)      // pointers to the start of each variable
                            + sizeof(TYPE) * ( total_footprint_p       // parameters
                                               + total_footprint_x       // "i" variables if tagIJ==1, "j" otherwise
                                               + total_footprint_y )));  // "j" variables if tagIJ==1, "i" otherwise
    
    
    // Now, fill in our big, contiguous array: ---------------------------------
    // In the head, the pointers to the data:
    TYPE **p_data_a = (TYPE**)p_data;
    px_d = p_data_a;   // pointers to the "x" variables, later on in p_data
    p_data_a += SIZEI;
    py_d = p_data_a;   // pointers to the "y" variables, later on in p_data
    p_data_a += SIZEJ;
    pp_d = p_data_a;   // pointers to the "parameters", later on in p_data
    p_data_a += SIZEP;
    
    // In the tail, the actual data:
    TYPE *p_data_b = (TYPE*)p_data_a;  // Beware: Instead of storing TYPE*, we now store TYPE
    param_d = p_data_b; // Parameters
    p_data_b += total_footprint_p;
    x_d = p_data_b;     // "x" variables
    p_data_b += total_footprint_x;
    y_d = p_data_b;     // "y" variables
    
    // host arrays of pointers to device data
    TYPE *phx_d[SIZEI];  // Will be loaded to px_d
    TYPE *phy_d[SIZEJ];  // Will be loaded to py_d
    TYPE *php_d[SIZEP];  // Will be loaded to pp_d
    
    // parameters --------------------------------------------------------------
    int nvals;
    // if DIMSP is empty (i.e. no parameter), nvals = -1 which could result in a segfault
    if(SIZEP > 0){
      nvals = footprints_p[0];  // dimension of the first parameter
      php_d[0] = param_d;
      CudaSafeCall(cudaMemcpy(php_d[0], pp_h[0], sizeof(TYPE)*nvals, cudaMemcpyHostToDevice));
      
      for(int k=1; k<SIZEP; k++) {
        php_d[k] = php_d[k-1] + nvals;  // Move to the right...
        nvals = footprints_p[k]; // Memory footprint of the k-th parameter...
        CudaSafeCall(cudaMemcpy(php_d[k], pp_h[k], sizeof(TYPE)*nvals, cudaMemcpyHostToDevice));  // And load the data
      }
    }
    
    // "x" variables -----------------------------------------------------------
    if (SIZEI > 0) {
      phx_d[0] = x_d;
      nvals = footprints_x[0];  // First "x variable" is the output: no need to load anything
      for(int k=1; k<SIZEI; k++) {
        phx_d[k] = phx_d[k-1] + nvals;  // Move to the right...
        nvals = footprints_x[k]; // Memory footprint of the k-th x variable...
        CudaSafeCall(cudaMemcpy(phx_d[k], px_h[k], sizeof(TYPE)*nvals, cudaMemcpyHostToDevice));
      }
    }
    
    // "y" variables -----------------------------------------------------------
    if (SIZEJ > 0) {
      phy_d[0] = y_d;
      nvals = footprints_y[0];  // First "y" variable...
      CudaSafeCall(cudaMemcpy(phy_d[0], py_h[0], sizeof(TYPE)*nvals, cudaMemcpyHostToDevice)); // Should be loaded on the device!
      for(int k=1; k<SIZEJ; k++) {
        phy_d[k] = phy_d[k-1] + nvals;  // Move to the right...
        nvals = footprints_y[k];  // Memory footprint of the (k+1)-th y variable...
        CudaSafeCall(cudaMemcpy(phy_d[k], py_h[k], sizeof(TYPE)*nvals, cudaMemcpyHostToDevice));  // And load the data
      }
    }
    
    // Load on the device the pointer arrays: ----------------------------------
    CudaSafeCall(cudaMemcpy(pp_d, php_d, SIZEP*sizeof(TYPE*), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(px_d, phx_d, SIZEI*sizeof(TYPE*), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(py_d, phy_d, SIZEJ*sizeof(TYPE*), cudaMemcpyHostToDevice));
    
    
    // Setup the compute properties ==============================================================
    // Compute on device : grid and block are both 1d
    cudaDeviceProp deviceProp;
    int dev = -1;
    CudaSafeCall(cudaGetDevice(&dev));
    CudaSafeCall(cudaGetDeviceProperties(&deviceProp, dev));
    
    dim3 blockSize;
    // warning : blockSize.x was previously set to CUDA_BLOCK_SIZE; currently CUDA_BLOCK_SIZE value is used as a bound.
    blockSize.x = ::std::min(CUDA_BLOCK_SIZE,::std::min(deviceProp.maxThreadsPerBlock, (int) (deviceProp.sharedMemPerBlock / ::std::max(1, (int)(DIMY*sizeof(TYPE))) ))); // number of threads in each block
    
    
    // Ranges pre-processing... ==================================================================
    
    // N.B.: In the following code, we assume that the x-ranges do not overlap.
    //       Otherwise, we'd have to assume that DIMRED == DIMOUT
    //       or allocate a buffer of size nx * DIMRED. This may be done in the future.
    // Cf. reduction.h:
    //    FUN::tagJ = 1 for a reduction over j, result indexed by i
    //    FUN::tagJ = 0 for a reduction over i, result indexed by j
    
    int nranges    = FUN::tagJ ?    nranges_x :    nranges_y ;
    int nredranges = FUN::tagJ ? nredranges_y : nredranges_x ;
    __INDEX__ *ranges_x = FUN::tagJ ? ranges[0] : ranges[3] ;
    __INDEX__ *slices_x = FUN::tagJ ? ranges[1] : ranges[4] ;
    __INDEX__ *ranges_y = FUN::tagJ ? ranges[2] : ranges[5] ;
    
    // Computes the number of blocks needed ---------------------------------------------
    int nblocks = 0, len_range = 0;
    for(int i=0; i<nranges ; i++){
      len_range = ranges_x[2*i+1] - ranges_x[2*i] ;
      nblocks += (len_range/blockSize.x) + (len_range%blockSize.x==0 ? 0 : 1) ;
    }
    
    // Create a lookup table for the blocks --------------------------------------------
    __INDEX__ *lookup_h = NULL;
    lookup_h = new __INDEX__[3*nblocks] ;
    int index = 0;
    for(int i=0; i<nranges ; i++){
      len_range = ranges_x[2*i+1] - ranges_x[2*i] ;
      for(int j=0; j<len_range ; j+=blockSize.x) {
        lookup_h[3*index]   = i;
        lookup_h[3*index+1] = ranges_x[2*i] + j;
        lookup_h[3*index+2] = ranges_x[2*i] + j + ::std::min((int) blockSize.x, len_range-j ) ;
        index++;
      }
    }
    
    // Load the table on the device -----------------------------------------------------
    __INDEX__ *lookup_d = NULL;
    CudaSafeCall(cudaMalloc((__INDEX__**)&lookup_d, sizeof(__INDEX__)*3*nblocks));
    CudaSafeCall(cudaMemcpy(lookup_d, lookup_h, sizeof(__INDEX__)*3*nblocks, cudaMemcpyHostToDevice));
    
    // Load copies of slices_x and ranges_y on the device:
    __INDEX__ *slices_x_d = NULL, *ranges_y_d = NULL;
    
    // Send data from host device:
    CudaSafeCall(cudaMalloc((__INDEX__**) &slices_x_d, sizeof(__INDEX__)*2*nranges));
    CudaSafeCall(cudaMemcpy(slices_x_d, slices_x, sizeof(__INDEX__)*2*nranges, cudaMemcpyHostToDevice));
    
    CudaSafeCall(cudaMalloc((__INDEX__**) &ranges_y_d, sizeof(__INDEX__)*2*nredranges));
    CudaSafeCall(cudaMemcpy(ranges_y_d, ranges_y, sizeof(__INDEX__)*2*nredranges, cudaMemcpyHostToDevice));
    
    // Support for broadcasting over batch dimensions =============================================
    
    // We create a lookup table, "offsets", of shape (nblock, SIZEVARS):
    int *offsets_d = NULL;
    
    if (nbatchdims > 0) {
      offsets_d = build_offset_tables<FUN>( nbatchdims, shapes, nblocks, lookup_h );
    }
    
    // ============================================================================================
    
    
    dim3 gridSize;
    gridSize.x =  nblocks; //nx / blockSize.x + (nx%blockSize.x==0 ? 0 : 1);
    
    // Size of the SharedData : blockSize.x*(DIMY)*sizeof(TYPE)
    GpuConv1DOnDevice_ranges<TYPE><<<gridSize,blockSize,blockSize.x*(DIMY)*sizeof(TYPE)>>>(fun,nx,ny,
            nbatchdims,shapes, offsets_d,
            lookup_d,slices_x_d,ranges_y_d,
            px_d,py_d,pp_d);
    
    // block until the device has completed
    CudaSafeCall(cudaDeviceSynchronize());
    CudaCheckError();
    
    // Send data from device to host.
    CudaSafeCall(cudaMemcpy(*px_h, x_d, sizeof(TYPE)*(nx*DIMOUT),cudaMemcpyDeviceToHost));
    
    // Free memory.
    CudaSafeCall(cudaFree(p_data));
    
    // Free the block lookup table :
    delete [] lookup_h;
    CudaSafeCall(cudaFree(lookup_d));
    CudaSafeCall(cudaFree(slices_x_d));
    CudaSafeCall(cudaFree(ranges_y_d));
    
    if (nbatchdims > 0) {
      CudaSafeCall(cudaFree(offsets_d));
    }
    
    return 0;
  }


// and use getlist to enroll them into "pointers arrays" px and py.
  template < typename TYPE, class FUN, typename... Args >
  static int Eval(FUN fun, int nx, int ny,
                  int nbatchdims, int *shapes,
                  int nranges_x, int nranges_y, int nredranges_x, int nredranges_y, __INDEX__ **ranges,
                  int device_id, TYPE* x1_h, Args... args) {
    
    if(device_id!=-1)
      CudaSafeCall(cudaSetDevice(device_id));
    
    typedef typename FUN::VARSI VARSI;
    typedef typename FUN::VARSJ VARSJ;
    typedef typename FUN::VARSP VARSP;
    
    const int SIZEI = VARSI::SIZE+1;
    const int SIZEJ = VARSJ::SIZE;
    const int SIZEP = VARSP::SIZE;
    
    using DIMSX = GetDims<VARSI>;
    using DIMSY = GetDims<VARSJ>;
    using DIMSP = GetDims<VARSP>;
    
    using INDSI = GetInds<VARSI>;
    using INDSJ = GetInds<VARSJ>;
    using INDSP = GetInds<VARSP>;
    
    TYPE *px_h[SIZEI];
    TYPE *py_h[SIZEJ];
    TYPE *pp_h[SIZEP];
    
    px_h[0] = x1_h;
    getlist<INDSI>(px_h+1,args...);
    getlist<INDSJ>(py_h,args...);
    getlist<INDSP>(pp_h,args...);
    
    return Eval_(fun,nx,ny,nbatchdims,shapes,nranges_x,nranges_y,nredranges_x,nredranges_y,ranges,px_h,py_h,pp_h);
    
  }

// same without the device_id argument
  template < typename TYPE, class FUN, typename... Args >
  static int Eval(FUN fun, int nx, int ny,
                  int nbatchdims, int *shapes,
                  int nranges_x, int nranges_y, int nredranges_x, int nredranges_y, __INDEX__ **ranges,
                  TYPE* x1_h, Args... args) {
    return Eval(fun, nx, ny, nbatchdims, shapes, nranges_x, nranges_y, nredranges_x, nredranges_y, ranges, -1, x1_h, args...);
  }

// Idem, but with args given as an array of arrays, instead of an explicit list of arrays
  template < typename TYPE, class FUN >
  static int Eval(FUN fun, int nx, int ny,
                  int nbatchdims, int *shapes,
                  int nranges_x, int nranges_y, int nredranges_x, int nredranges_y, __INDEX__ **ranges,
                  TYPE* x1_h, TYPE** args, int device_id=-1) {
    
    // We set the GPU device on which computations will be performed
    if(device_id!=-1)
      CudaSafeCall(cudaSetDevice(device_id));
    
    typedef typename FUN::VARSI VARSI;
    typedef typename FUN::VARSJ VARSJ;
    typedef typename FUN::VARSP VARSP;
    
    const int SIZEI = VARSI::SIZE+1;
    const int SIZEJ = VARSJ::SIZE;
    const int SIZEP = VARSP::SIZE;
    
    using DIMSX = GetDims<VARSI>;
    using DIMSY = GetDims<VARSJ>;
    using DIMSP = GetDims<VARSP>;
    
    using INDSI = GetInds<VARSI>;
    using INDSJ = GetInds<VARSJ>;
    using INDSP = GetInds<VARSP>;
    
    TYPE *px_h[SIZEI];
    TYPE *py_h[SIZEJ];
    TYPE *pp_h[SIZEP];
    
    px_h[0] = x1_h;
    for(int i=1; i<SIZEI; i++)
      px_h[i] = args[INDSI::VAL(i-1)];
    for(int i=0; i<SIZEJ; i++)
      py_h[i] = args[INDSJ::VAL(i)];
    for(int i=0; i<SIZEP; i++)
      pp_h[i] = args[INDSP::VAL(i)];
    
    return Eval_(fun,nx,ny,nbatchdims,shapes,nranges_x,nranges_y,nredranges_x,nredranges_y,ranges,px_h,py_h,pp_h);
    
  }
  
  
};


}

using namespace keops;

extern "C" int GpuReduc1D_ranges_FromHost(int nx, int ny,
                                          int nbatchdims, int *shapes,
                                          int nranges_x, int nranges_y,
                                          int nredranges_x, int nredranges_y, __INDEX__ **castedranges,
                                          __TYPE__ *gamma, __TYPE__ **args, int device_id = -1) {
  return Eval< F, GpuConv1D_ranges_FromHost >::Run(nx,
                                                   ny,
                                                   nbatchdims,
                                                   shapes,
                                                   nranges_x,
                                                   nranges_y,
                                                   nredranges_x,
                                                   nredranges_y,
                                                   castedranges,
                                                   gamma,
                                                   args,
                                                   device_id);
}

