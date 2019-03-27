#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <cuda.h>

#include "core/Pack.h"

namespace keops {

template < typename TYPE, class FUN >
__global__ void GpuConv1DOnDevice_ranges(FUN fun, int nx, int ny,
    __INDEX__* lookup_d, __INDEX__* slices_x, __INDEX__* ranges_y,
    TYPE** px, TYPE** py, TYPE** pp) {

    __INDEX__ range_id= (lookup_d)[3*blockIdx.x] ;
    __INDEX__ start_x = (lookup_d)[3*blockIdx.x+1] ;
    __INDEX__ end_x   = (lookup_d)[3*blockIdx.x+2] ;
    
    __INDEX__ start_slice = range_id < 1 ? 0 : slices_x[range_id-1];
    __INDEX__ end_slice   = slices_x[range_id];

    // get the index of the current thread
    int i = start_x + threadIdx.x;

    // declare shared mem
    extern __shared__ TYPE yj[];

    // get templated dimensions :
    typedef typename FUN::DIMSX DIMSX;  // DIMSX is a "vector" of templates giving dimensions of xi variables
    typedef typename FUN::DIMSY DIMSY;  // DIMSY is a "vector" of templates giving dimensions of yj variables
    typedef typename FUN::DIMSP DIMSP;  // DIMSP is a "vector" of templates giving dimensions of parameters variables
    const int DIMX = DIMSX::SUM;        // DIMX  is sum of dimensions for xi variables
    const int DIMY = DIMSY::SUM;        // DIMY  is sum of dimensions for yj variables
    const int DIMP = DIMSP::SUM;        // DIMP  is sum of dimensions for parameters variables
    const int DIMOUT = FUN::DIM; // dimension of output variable
    const int DIMRED = FUN::DIMRED; // dimension of reduction operation
    const int DIMFOUT = DIMSX::FIRST;     // DIMFOUT is dimension of output variable of inner function

    // load parameter(s)
    TYPE param_loc[DIMP < 1 ? 1 : DIMP];
	load<DIMSP>(0,param_loc,pp); // load parameters variables from global memory to local thread memory

    // get the value of variable (index with i)
    TYPE xi[DIMX < 1 ? 1 : DIMX] ,tmp[DIMRED];
    if(i<end_x) {
        typename FUN::template InitializeReduction<TYPE>()(tmp); // tmp = 0
        load<typename DIMSX::NEXT>(i,xi+DIMFOUT,px+1); // load xi variables from global memory to local thread memory
    }

    __INDEX__ start_y = ranges_y[2*start_slice], end_y = 0;
    for( __INDEX__ index = start_slice ; index < end_slice ; index++ ) {
        if( (index+1 >= end_slice) || (ranges_y[2*index+2] != ranges_y[2*index+1]) ) {
            //start_y = ranges_y[2*index] ;
            end_y = ranges_y[2*index+1];
            //printf("%d,%d.", start_y, end_y) ;
            for(int jstart = start_y, tile = 0; jstart < end_y; jstart += blockDim.x, tile++) {

                // get the current column
                int j = jstart + threadIdx.x;

                if(j<end_y) { // we load yj from device global memory only if j<end_y
                    load<DIMSY>(j,yj+threadIdx.x*DIMY,py); // load yj variables from global memory to shared memory
                }
                __syncthreads();

                if(i<end_x) { // we compute x1i only if needed
                    TYPE* yjrel = yj; // Loop on the columns of the current block.
                    for(int jrel = 0; (jrel < blockDim.x) && (jrel<end_y-jstart); jrel++, yjrel+=DIMY) {
                        call<DIMSX,DIMSY,DIMSP>(fun,xi,yjrel,param_loc); // Call the function, which accumulates results in xi[0:DIMX1]
                        typename FUN::template ReducePairShort<TYPE>()(tmp, xi, jrel+tile*blockDim.x);     // tmp += xi
                    }
                }
                __syncthreads();
            }

            if(index+1 < end_slice) {
                start_y = ranges_y[2*index+2] ;
            }
        }
    }
    if(i<end_x) {
    	typename FUN::template FinalizeOutput<TYPE>()(tmp, px[0]+i*DIMOUT, px, i);
    }

}


struct GpuConv1D_ranges_FromHost {

template < typename TYPE, class FUN >
static int Eval_(FUN fun, int nx, int ny, 
    int nranges_x, int nranges_y, int nredranges_x, int nredranges_y, __INDEX__ **ranges, 
    TYPE** px_h, TYPE** py_h, TYPE** pp_h) {

    typedef typename FUN::DIMSX DIMSX;
    typedef typename FUN::DIMSY DIMSY;
    typedef typename FUN::DIMSP DIMSP;
    const int DIMX = DIMSX::SUM;
    const int DIMY = DIMSY::SUM;
    const int DIMP = DIMSP::SUM;
    const int DIMOUT = FUN::DIM; // dimension of output variable
    const int DIMFOUT = DIMSX::FIRST; // DIMFOUT is dimension of output variable of inner function
    const int SIZEI = DIMSX::SIZE;
    const int SIZEJ = DIMSY::SIZE;
    const int SIZEP = DIMSP::SIZE;

    // pointers to device data
    TYPE *x_d, *y_d, *param_d;

    // device arrays of pointers to device data
    TYPE **px_d, **py_d, **pp_d;

    // single cudaMalloc
    void **p_data;
    CudaSafeCall(cudaMalloc((void**)&p_data, sizeof(TYPE*)*(SIZEI+SIZEJ+SIZEP)+sizeof(TYPE)*(DIMP+nx*(DIMX-DIMFOUT+DIMOUT)+ny*DIMY)));

    TYPE **p_data_a = (TYPE**)p_data;
    px_d = p_data_a;
    p_data_a += SIZEI;
    py_d = p_data_a;
    p_data_a += SIZEJ;
    pp_d = p_data_a;
    p_data_a += SIZEP;
    TYPE *p_data_b = (TYPE*)p_data_a;
    param_d = p_data_b;
    p_data_b += DIMP;
    x_d = p_data_b;
    p_data_b += nx*(DIMX-DIMFOUT+DIMOUT);
    y_d = p_data_b;

    // host arrays of pointers to device data
    TYPE *phx_d[SIZEI];
    TYPE *phy_d[SIZEJ];
    TYPE *php_d[SIZEP];

    int nvals;    
    nvals = DIMSP::VAL(0);
    // if DIMSP is empty (i.e. no parameter), nvals = -1 which could result in a segfault
    if(nvals >= 0){ 
        php_d[0] = param_d;
        CudaSafeCall(cudaMemcpy(php_d[0], pp_h[0], sizeof(TYPE)*nvals, cudaMemcpyHostToDevice));
    }
    for(int k=1; k<SIZEP; k++) {
        php_d[k] = php_d[k-1] + nvals;
        nvals = DIMSP::VAL(k);
        CudaSafeCall(cudaMemcpy(php_d[k], pp_h[k], sizeof(TYPE)*nvals, cudaMemcpyHostToDevice));
    }    

    phx_d[0] = x_d;
    nvals = nx*DIMOUT;
    for(int k=1; k<SIZEI; k++) {
        phx_d[k] = phx_d[k-1] + nvals;
        nvals = nx*DIMSX::VAL(k);
        CudaSafeCall(cudaMemcpy(phx_d[k], px_h[k], sizeof(TYPE)*nvals, cudaMemcpyHostToDevice));
    }

    phy_d[0] = y_d;
    nvals = ny*DIMSY::VAL(0);
    CudaSafeCall(cudaMemcpy(phy_d[0], py_h[0], sizeof(TYPE)*nvals, cudaMemcpyHostToDevice));
    for(int k=1; k<SIZEJ; k++) {
        phy_d[k] = phy_d[k-1] + nvals;
        nvals = ny*DIMSY::VAL(k);
        CudaSafeCall(cudaMemcpy(phy_d[k], py_h[k], sizeof(TYPE)*nvals, cudaMemcpyHostToDevice));
    }

    // copy arrays of pointers
    CudaSafeCall(cudaMemcpy(pp_d, php_d, SIZEP*sizeof(TYPE*), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(px_d, phx_d, SIZEI*sizeof(TYPE*), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(py_d, phy_d, SIZEJ*sizeof(TYPE*), cudaMemcpyHostToDevice));

    // Compute on device : grid and block are both 1d
    cudaDeviceProp deviceProp;
    int dev = -1;
    CudaSafeCall(cudaGetDevice(&dev));
    CudaSafeCall(cudaGetDeviceProperties(&deviceProp, dev));

    dim3 blockSize;
    // warning : blockSize.x was previously set to CUDA_BLOCK_SIZE; currently CUDA_BLOCK_SIZE value is used as a bound.
    blockSize.x = min(CUDA_BLOCK_SIZE,min(deviceProp.maxThreadsPerBlock, (int) (deviceProp.sharedMemPerBlock / (DIMY*sizeof(TYPE))))); // number of threads in each block


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
            lookup_h[3*index+2] = ranges_x[2*i] + j + std::min((int) blockSize.x, len_range-j ) ;

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

    // ============================================================================================


    dim3 gridSize;
    gridSize.x =  nblocks; //nx / blockSize.x + (nx%blockSize.x==0 ? 0 : 1);

    // Size of the SharedData : blockSize.x*(DIMY)*sizeof(TYPE)
    GpuConv1DOnDevice_ranges<TYPE><<<gridSize,blockSize,blockSize.x*(DIMY)*sizeof(TYPE)>>>(fun,nx,ny,
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
    CudaSafeCall(cudaFree(lookup_d));
    CudaSafeCall(cudaFree(slices_x_d));
    CudaSafeCall(cudaFree(ranges_y_d));

    return 0;
}


// and use getlist to enroll them into "pointers arrays" px and py.
template < typename TYPE, class FUN, typename... Args >
static int Eval(FUN fun, int nx, int ny, 
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

    return Eval_(fun,nx,ny,nranges_x,nranges_y,nredranges_x,nredranges_y,ranges,px_h,py_h,pp_h);

}

// same without the device_id argument
template < typename TYPE, class FUN, typename... Args >
static int Eval(FUN fun, int nx, int ny, 
    int nranges_x, int nranges_y, int nredranges_x, int nredranges_y, __INDEX__ **ranges, 
    TYPE* x1_h, Args... args) {
    return Eval(fun, nx, ny, nranges_x, nranges_y, nredranges_x, nredranges_y, ranges, -1, x1_h, args...);
}

// Idem, but with args given as an array of arrays, instead of an explicit list of arrays
template < typename TYPE, class FUN >
static int Eval(FUN fun, int nx, int ny, 
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

    return Eval_(fun,nx,ny,nranges_x,nranges_y,nredranges_x,nredranges_y,ranges,px_h,py_h,pp_h);

}


};


struct GpuConv1D_ranges_FromDevice {
template < typename TYPE, class FUN >
static int Eval_(FUN fun, int nx, int ny, 
    int nranges_x, int nranges_y, __INDEX__ **ranges, 
    TYPE** phx_d, TYPE** phy_d, TYPE** php_d) {

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
    CudaSafeCall(cudaMalloc((void**)&p_data, sizeof(TYPE*)*(SIZEI+SIZEJ+SIZEP)));

    TYPE **p_data_a = (TYPE**)p_data;
    px_d = p_data_a;
    p_data_a += SIZEI;
    py_d = p_data_a;
    p_data_a += SIZEJ;
    pp_d = p_data_a;

    CudaSafeCall(cudaMemcpy(px_d, phx_d, SIZEI*sizeof(TYPE*), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(py_d, phy_d, SIZEJ*sizeof(TYPE*), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(pp_d, php_d, SIZEP*sizeof(TYPE*), cudaMemcpyHostToDevice));

    // Compute on device : grid and block are both 1d

    cudaDeviceProp deviceProp;
    int dev = -1;
    CudaSafeCall(cudaGetDevice(&dev));
    CudaSafeCall(cudaGetDeviceProperties(&deviceProp, dev));

    dim3 blockSize;
    // warning : blockSize.x was previously set to CUDA_BLOCK_SIZE; currently CUDA_BLOCK_SIZE value is used as a bound.
    blockSize.x = min(CUDA_BLOCK_SIZE,min(deviceProp.maxThreadsPerBlock, (int) (deviceProp.sharedMemPerBlock / (DIMY*sizeof(TYPE))))); // number of threads in each block


    // Ranges pre-processing... ==================================================================
    
    // N.B.: In the following code, we assume that the x-ranges do not overlap.
    //       Otherwise, we'd have to assume that DIMRED == DIMOUT
    //       or allocate a buffer of size nx * DIMRED. This may be done in the future.
    // Cf. reduction.h: 
    //    FUN::tagJ = 1 for a reduction over j, result indexed by i
    //    FUN::tagJ = 0 for a reduction over i, result indexed by j

    int nranges = FUN::tagJ ? nranges_x : nranges_y ;
    __INDEX__ *ranges_x = FUN::tagJ ? ranges[0] : ranges[3] ;
    __INDEX__ *slices_x = FUN::tagJ ? ranges[1] : ranges[4] ;
    __INDEX__ *ranges_y = FUN::tagJ ? ranges[2] : ranges[5] ;

    __INDEX__* ranges_x_h = NULL;
    ranges_x_h = new __INDEX__[2*nranges] ;
    // Send data from device to host.
    CudaSafeCall(cudaMemcpy(ranges_x_h, ranges_x, sizeof(__INDEX__)*2*nranges, cudaMemcpyDeviceToHost));

    // Computes the number of blocks needed ---------------------------------------------
    int nblocks = 0, len_range = 0;
    for(int i=0; i<nranges ; i++){
        len_range = ranges_x_h[2*i+1] - ranges_x_h[2*i] ;
        nblocks += (len_range/blockSize.x) + (len_range%blockSize.x==0 ? 0 : 1) ;
    }

    // Create a lookup table for the blocks --------------------------------------------
    __INDEX__ *lookup_h = NULL;
    lookup_h = new __INDEX__[3*nblocks] ;
    int index = 0;
    for(int i=0; i<nranges ; i++){
        len_range = ranges_x_h[2*i+1] - ranges_x_h[2*i] ;
        for(int j=0; j<len_range ; j+=blockSize.x) {
            lookup_h[3*index]   = i;
            lookup_h[3*index+1] = ranges_x_h[2*i] + j;
            lookup_h[3*index+2] = ranges_x_h[2*i] + j + std::min((int) blockSize.x, len_range-j ) ;

            index++;
        }
    }

    // Load the table on the device -----------------------------------------------------
    __INDEX__ *lookup_d = NULL;
    CudaSafeCall(cudaMalloc((__INDEX__**)&lookup_d, sizeof(__INDEX__)*3*nblocks));
    CudaSafeCall(cudaMemcpy(lookup_d, lookup_h, sizeof(__INDEX__)*3*nblocks, cudaMemcpyHostToDevice));

    // ============================================================================================

    dim3 gridSize;
    gridSize.x =  nblocks ; //nx / blockSize.x + (nx%blockSize.x==0 ? 0 : 1);

    // Size of the SharedData : blockSize.x*(DIMY)*sizeof(TYPE)
    GpuConv1DOnDevice_ranges<TYPE><<<gridSize,blockSize,blockSize.x*(DIMY)*sizeof(TYPE)>>>(fun,nx,ny,
        lookup_d,slices_x,ranges_y,
        px_d,py_d,pp_d);

    // block until the device has completed
    CudaSafeCall(cudaDeviceSynchronize());
    CudaCheckError();

    CudaSafeCall(cudaFree(p_data));

    // Free the block lookup table :
    delete [] ranges_x_h;
    delete [] lookup_h;
    CudaSafeCall(cudaFree(lookup_d));
    //CudaSafeCall(cudaFree(ranges_x_h));


    return 0;
}

// Same wrappers, but for data located on the device
template < typename TYPE, class FUN, typename... Args >
static int Eval(FUN fun, int nx, int ny, 
    int nranges_x, int nranges_y, __INDEX__ **ranges, 
    int device_id, TYPE* x1_d, Args... args) {

    // device_id is provided, so we set the GPU device accordingly
    // Warning : is has to be consistent with location of data
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

    TYPE *phx_d[SIZEI];
    TYPE *phy_d[SIZEJ];
    TYPE *php_d[SIZEP];

    phx_d[0] = x1_d;

    getlist<INDSI>(phx_d+1,args...);
    getlist<INDSJ>(phy_d,args...);
    getlist<INDSP>(php_d,args...);

    return Eval_(fun,nx,ny,nranges_x,nranges_y,ranges,phx_d,phy_d,php_d);

}

// same without the device_id argument
template < typename TYPE, class FUN, typename... Args >
static int Eval(FUN fun, int nx, int ny, 
    int nranges_x, int nranges_y, __INDEX__ **ranges, 
    TYPE* x1_d, Args... args) {
    // We set the GPU device on which computations will be performed
    // to be the GPU on which data is located.
    // NB. we only check location of x1_d which is the output vector
    // so we assume that input data is on the same GPU
    // note : cudaPointerGetAttributes has a strange behaviour:
    // it looks like it makes a copy of the vector on the default GPU device (0) !!! 
    // So we prefer to avoid this and provide directly the device_id as input (first function above)
    cudaPointerAttributes attributes;
    CudaSafeCall(cudaPointerGetAttributes(&attributes,x1_d));
    return Eval(fun, nx, ny, nranges_x,nranges_y,ranges, attributes.device, x1_d, args...);
}

template < typename TYPE, class FUN >
static int Eval(FUN fun, int nx, int ny, 
    int nranges_x, int nranges_y, __INDEX__ **ranges, 
    TYPE* x1_d, TYPE** args, int device_id=-1) {

    if(device_id==-1) {
        // We set the GPU device on which computations will be performed
        // to be the GPU on which data is located.
        // NB. we only check location of x1_d which is the output vector
        // so we assume that input data is on the same GPU
        // note : cudaPointerGetAttributes has a strange behaviour:
        // it looks like it makes a copy of the vector on the default GPU device (0) !!! 
	// So we prefer to avoid this and provide directly the device_id as input (else statement below)
        cudaPointerAttributes attributes;
        CudaSafeCall(cudaPointerGetAttributes(&attributes,x1_d));
        CudaSafeCall(cudaSetDevice(attributes.device));
    } else // device_id is provided, so we use it. Warning : is has to be consistent with location of data
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

    TYPE *px_d[SIZEI];
    TYPE *py_d[SIZEJ];
    TYPE *pp_d[SIZEP];

    px_d[0] = x1_d;
    for(int i=1; i<SIZEI; i++)
        px_d[i] = args[INDSI::VAL(i-1)];
    for(int i=0; i<SIZEJ; i++)
        py_d[i] = args[INDSJ::VAL(i)];
    for(int i=0; i<SIZEP; i++)
        pp_d[i] = args[INDSP::VAL(i)];

    return Eval_(fun,nx,ny,nranges_x,nranges_y,ranges,px_d,py_d,pp_d);

}

};

}
