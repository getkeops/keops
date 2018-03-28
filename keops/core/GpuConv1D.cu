#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <cuda.h>

#include "core/Pack.h"
#include "core/reductions/sum.h"
#include "core/reductions/log_sum_exp.h"


template < typename TYPE, class FUN >
__global__ void GpuConv1DOnDevice(FUN fun, int nx, int ny, TYPE** px, TYPE** py, TYPE** pp) {

    // get the index of the current thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // declare shared mem
    extern __shared__ TYPE yj[];

    // get templated dimensions :
    typedef typename FUN::DIMSX DIMSX;  // DIMSX is a "vector" of templates giving dimensions of xi variables
    typedef typename FUN::DIMSY DIMSY;  // DIMSY is a "vector" of templates giving dimensions of yj variables
    typedef typename FUN::DIMSP DIMSP;  // DIMSP is a "vector" of templates giving dimensions of parameters variables
    const int DIMX = DIMSX::SUM;        // DIMX  is sum of dimensions for xi variables
    const int DIMY = DIMSY::SUM;        // DIMY  is sum of dimensions for yj variables
    const int DIMP = DIMSP::SUM;        // DIMP  is sum of dimensions for parameters variables
    const int DIMX1 = DIMSX::FIRST;     // DIMX1 is dimension of output variable

    // load parameter(s)
    TYPE param_loc[DIMP < 1 ? 1 : DIMP];
	load<DIMSP>(0,param_loc,pp); // load parameters variables from global memory to local thread memory

    // get the value of variable (index with i)
    TYPE xi[DIMX < 1 ? 1 : DIMX] ,tmp[DIMX1];
    if(i<nx) {
        InitializeOutput<TYPE,DIMX1,typename FUN::FORM>()(tmp); // tmp = 0
        load<DIMSX::NEXT>(i,xi+DIMX1,px+1); // load xi variables from global memory to local thread memory
    }

    for(int jstart = 0, tile = 0; jstart < ny; jstart += blockDim.x, tile++) {

        // get the current column
        int j = tile * blockDim.x + threadIdx.x;

        if(j<ny) { // we load yj from device global memory only if j<ny
            load<DIMSY>(j,yj+threadIdx.x*DIMY,py); // load yj variables from global memory to shared memory
        }
        __syncthreads();

        if(i<nx) { // we compute x1i only if needed
            TYPE* yjrel = yj; // Loop on the columns of the current block.
            for(int jrel = 0; (jrel < blockDim.x) && (jrel<ny-jstart); jrel++, yjrel+=DIMY) {
                call<DIMSX,DIMSY,DIMSP>(fun,xi,yjrel,param_loc); // Call the function, which accumulates results in xi[0:DIMX1]
                ReducePair<TYPE,DIMX1,typename FUN::FORM>()(tmp, xi);     // tmp += xi
            }
        }
        __syncthreads();
    }
    if(i<nx) {
        for(int k=0; k<DIMX1; k++)
            (*px)[i*DIMX1+k] = tmp[k];
    }

}

template < typename TYPE, class FUN >
int GpuConv1D_FromHost(FUN fun, int nx, int ny, TYPE** px_h, TYPE** py_h, TYPE** pp_h) {

    typedef typename FUN::DIMSX DIMSX;
    typedef typename FUN::DIMSY DIMSY;
    typedef typename FUN::DIMSP DIMSP;
    const int DIMX = DIMSX::SUM;
    const int DIMY = DIMSY::SUM;
    const int DIMP = DIMSP::SUM;
    const int DIMX1 = DIMSX::FIRST;
    const int SIZEI = DIMSX::SIZE;
    const int SIZEJ = DIMSY::SIZE;
    const int SIZEP = DIMSP::SIZE;

    // pointers to device data
    TYPE *x_d, *y_d, *param_d;

    // device arrays of pointers to device data
    TYPE **px_d, **py_d, **pp_d;

    // single cudaMalloc
    void **p_data;
    cudaMalloc((void**)&p_data, sizeof(TYPE*)*(SIZEI+SIZEJ+SIZEP)+sizeof(TYPE)*(DIMP+nx*DIMX+ny*DIMY));

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
    p_data_b += nx*DIMX;
    y_d = p_data_b;

    // host arrays of pointers to device data
    TYPE *phx_d[SIZEI];
    TYPE *phy_d[SIZEJ];
    TYPE *php_d[SIZEP];

    int nvals;    
    php_d[0] = param_d;
    nvals = DIMSP::VAL(0);
    // if DIMSP is empty (i.e. no parameter), nvals = -1 which could result in a segfault
    if(nvals >= 0){ 
        cudaMemcpy(php_d[0], pp_h[0], sizeof(TYPE)*nvals, cudaMemcpyHostToDevice);
    }
    for(int k=1; k<SIZEP; k++) {
        php_d[k] = php_d[k-1] + nvals;
        nvals = DIMSP::VAL(k);
        cudaMemcpy(php_d[k], pp_h[k], sizeof(TYPE)*nvals, cudaMemcpyHostToDevice);
    }    

    phx_d[0] = x_d;
    nvals = nx*DIMSX::VAL(0);
    for(int k=1; k<SIZEI; k++) {
        phx_d[k] = phx_d[k-1] + nvals;
        nvals = nx*DIMSX::VAL(k);
        cudaMemcpy(phx_d[k], px_h[k], sizeof(TYPE)*nvals, cudaMemcpyHostToDevice);
    }

    phy_d[0] = y_d;
    nvals = ny*DIMSY::VAL(0);
    cudaMemcpy(phy_d[0], py_h[0], sizeof(TYPE)*nvals, cudaMemcpyHostToDevice);
    for(int k=1; k<SIZEJ; k++) {
        phy_d[k] = phy_d[k-1] + nvals;
        nvals = ny*DIMSY::VAL(k);
        cudaMemcpy(phy_d[k], py_h[k], sizeof(TYPE)*nvals, cudaMemcpyHostToDevice);
    }

    // copy arrays of pointers
    cudaMemcpy(pp_d, php_d, SIZEP*sizeof(TYPE*), cudaMemcpyHostToDevice);
    cudaMemcpy(px_d, phx_d, SIZEI*sizeof(TYPE*), cudaMemcpyHostToDevice);
    cudaMemcpy(py_d, phy_d, SIZEJ*sizeof(TYPE*), cudaMemcpyHostToDevice);

    // Compute on device : grid is 2d and block is 1d
    dim3 blockSize;
    blockSize.x = CUDA_BLOCK_SIZE; // number of threads in each block
    dim3 gridSize;
    gridSize.x =  nx / blockSize.x + (nx%blockSize.x==0 ? 0 : 1);

    // Size of the SharedData : blockSize.x*(DIMY)*sizeof(TYPE)
    GpuConv1DOnDevice<TYPE><<<gridSize,blockSize,blockSize.x*(DIMY)*sizeof(TYPE)>>>(fun,nx,ny,px_d,py_d,pp_d);

    // block until the device has completed
    cudaThreadSynchronize();

    // Send data from device to host.
    cudaMemcpy(*px_h, x_d, sizeof(TYPE)*(nx*DIMX1),cudaMemcpyDeviceToHost);

    // Free memory.
    cudaFree(p_data);

    return 0;
}

template < typename TYPE, class FUN >
int GpuConv1D_FromDevice(FUN fun, int nx, int ny, TYPE** phx_d, TYPE** phy_d, TYPE** php_d) {

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
    cudaMalloc((void**)&p_data, sizeof(TYPE*)*(SIZEI+SIZEJ+SIZEP));

    TYPE **p_data_a = (TYPE**)p_data;
    px_d = p_data_a;
    p_data_a += SIZEI;
    py_d = p_data_a;
    p_data_a += SIZEJ;
    pp_d = p_data_a;

    cudaMemcpy(px_d, phx_d, SIZEI*sizeof(TYPE*), cudaMemcpyHostToDevice);
    cudaMemcpy(py_d, phy_d, SIZEJ*sizeof(TYPE*), cudaMemcpyHostToDevice);
    cudaMemcpy(pp_d, php_d, SIZEP*sizeof(TYPE*), cudaMemcpyHostToDevice);

    // Compute on device : grid and block are both 1d
    dim3 blockSize;
    blockSize.x = CUDA_BLOCK_SIZE; // number of threads in each block
    dim3 gridSize;
    gridSize.x =  nx / blockSize.x + (nx%blockSize.x==0 ? 0 : 1);

    // Size of the SharedData : blockSize.x*(DIMY)*sizeof(TYPE)
    GpuConv1DOnDevice<TYPE><<<gridSize,blockSize,blockSize.x*(DIMY)*sizeof(TYPE)>>>(fun,nx,ny,px_d,py_d,pp_d);

    // block until the device has completed
    cudaThreadSynchronize();

    cudaFree(p_data);

    return 0;
}

// and use getlist to enroll them into "pointers arrays" px and py.
template < typename TYPE, class FUN, typename... Args >
int GpuConv1D(FUN fun, int nx, int ny, TYPE* x1_h, Args... args) {

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

    return GpuConv1D_FromHost(fun,nx,ny,px_h,py_h,pp_h);

}

// Idem, but with args given as an array of arrays, instead of an explicit list of arrays
template < typename TYPE, class FUN >
int GpuConv1D(FUN fun, int nx, int ny, TYPE* x1_h, TYPE** args) {
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

    return GpuConv1D_FromHost(fun,nx,ny,px_h,py_h,pp_h);

}

// Same wrappers, but for data located on the device
template < typename TYPE, class FUN, typename... Args >
int GpuConv1D_FromDevice(FUN fun, int nx, int ny, TYPE* x1_d, Args... args) {

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

    return GpuConv1D_FromDevice(fun,nx,ny,phx_d,phy_d,php_d);

}

template < typename TYPE, class FUN >
int GpuConv1D_FromDevice(FUN fun, int nx, int ny, TYPE* x1_d, TYPE** args) {
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

    return GpuConv1D_FromDevice(fun,nx,ny,px_d,py_d,pp_d);

}
