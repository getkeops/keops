#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <cuda.h>


#include "Pack.h"
#include "reductions/sum.h"
#include "reductions/log_sum_exp.h"

template < typename TYPE, class FUN, class PARAM >
__global__ void GpuConv1DOnDevice(FUN fun, PARAM param, int nx, int ny, TYPE** px, TYPE** py) {

    // get the index of the current thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // declare shared mem
    extern __shared__ TYPE yj[];

    // get templated dimensions :
    typedef typename FUN::DIMSX DIMSX;  // DIMSX is a "vector" of templates giving dimensions of xi variables
    typedef typename FUN::DIMSY DIMSY;  // DIMSY is a "vector" of templates giving dimensions of yj variables
    const int DIMPARAM = FUN::DIMPARAM; // DIMPARAM is the total size of the param vector
    const int DIMX = DIMSX::SUM;        // DIMX  is sum of dimensions for xi variables
    const int DIMY = DIMSY::SUM;        // DIMY  is sum of dimensions for yj variables
    const int DIMX1 = DIMSX::FIRST;     // DIMX1 is dimension of output variable

    // load parameter(s)
    TYPE param_loc[DIMPARAM < 1 ? 1 : DIMPARAM];
    for(int k=0; k<DIMPARAM; k++)
        param_loc[k] = param[k];

    // get the value of variable (index with i)
    TYPE xi[DIMX] ,tmp[DIMX1];
    if(i<nx) {
        InitializeOutput<TYPE,DIMX1,FUN>()(tmp); // tmp = 0
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
                call<DIMSX,DIMSY>(fun,xi,yjrel,param_loc); // Call the function, which accumulates results in xi[0:DIMX1]
                ReducePair<TYPE,DIMX1,FUN>()(tmp, xi);     // tmp += xi
            }
        }

        __syncthreads();
    }

    if(i<nx){
        for(int k=0; k<DIMX1; k++)
            (*px)[i*DIMX1+k] = tmp[k];
    }
}

template < typename TYPE, class FUN, class PARAM >
int GpuConv1D_FromHost(FUN fun, PARAM param_h, int nx, int ny, TYPE** px_h, TYPE** py_h) {

    typedef typename FUN::DIMSX DIMSX;
    typedef typename FUN::DIMSY DIMSY;
    const int DIMPARAM = FUN::DIMPARAM;
    const int DIMX = DIMSX::SUM;
    const int DIMY = DIMSY::SUM;
    const int DIMX1 = DIMSX::FIRST;
    const int SIZEI = DIMSX::SIZE;
    const int SIZEJ = DIMSY::SIZE;


    // pointers to device data
    TYPE *x_d, *y_d, *param_d;

    // device arrays of pointers to device data
    TYPE **px_d, **py_d;

    // single cudaMalloc
    void **p_data;
    cudaMalloc((void**)&p_data, sizeof(TYPE*)*(SIZEI+SIZEJ)+sizeof(TYPE)*(DIMPARAM+nx*DIMX+ny*DIMY));

    TYPE **p_data_a = (TYPE**)p_data;
    px_d = p_data_a;
    p_data_a += SIZEI;
    py_d = p_data_a;
    p_data_a += SIZEJ;
    TYPE *p_data_b = (TYPE*)p_data_a;
    param_d = p_data_b;
    p_data_b += DIMPARAM;
    x_d = p_data_b;
    p_data_b += nx*DIMX;
    y_d = p_data_b;

    // host arrays of pointers to device data
    TYPE *phx_d[SIZEI];
    TYPE *phy_d[SIZEJ];

    // Send data from host to device.
    cudaMemcpy(param_d, param_h, sizeof(TYPE)*DIMPARAM, cudaMemcpyHostToDevice);

    int nvals;
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
    cudaMemcpy(px_d, phx_d, SIZEI*sizeof(TYPE*), cudaMemcpyHostToDevice);
    cudaMemcpy(py_d, phy_d, SIZEJ*sizeof(TYPE*), cudaMemcpyHostToDevice);

    // Compute on device : grid is 2d and block is 1d
    dim3 blockSize;
    blockSize.x = 192; // number of threads in each block
    dim3 gridSize;
    gridSize.x =  nx / blockSize.x + (nx%blockSize.x==0 ? 0 : 1);

    // Size of the SharedData : blockSize.x*(DIMY)*sizeof(TYPE)
    GpuConv1DOnDevice<TYPE><<<gridSize,blockSize,blockSize.x*(DIMY)*sizeof(TYPE)>>>(fun,param_d,nx,ny,px_d,py_d);

    // block until the device has completed
    cudaThreadSynchronize();

    // Send data from device to host.
    cudaMemcpy(*px_h, x_d, sizeof(TYPE)*(nx*DIMX1),cudaMemcpyDeviceToHost);

    // Free memory.
    cudaFree(p_data);

    return 0;
}

template < typename TYPE, class FUN, class PARAM >
int GpuConv1D_FromDevice(FUN fun, PARAM param_d, int nx, int ny, TYPE** px_d, TYPE** py_d) {

    typedef typename FUN::DIMSX DIMSX;
    typedef typename FUN::DIMSY DIMSY;
    const int DIMY = DIMSY::SUM;
    const int DIMX1 = DIMSX::FIRST;

    /* // (Jean :) This portion of code seems to be useless,
    //             forgotten from a GpuConv2D copy-paste
    TYPE *out;
    cudaMalloc((void**)&out, sizeof(TYPE)*(nx*DIMX1));
    out = px_d[0]; // save the output location
    */

    // Compute on device : grid and block are both 1d
    dim3 blockSize;
    blockSize.x = 192; // number of threads in each block
    dim3 gridSize;
    gridSize.x =  nx / blockSize.x + (nx%blockSize.x==0 ? 0 : 1);

    // Size of the SharedData : blockSize.x*(DIMY)*sizeof(TYPE)
    GpuConv1DOnDevice<TYPE><<<gridSize,blockSize,blockSize.x*(DIMY)*sizeof(TYPE)>>>(fun,param_d,nx,ny,px_d,py_d);

    // block until the device has completed
    cudaThreadSynchronize();

    return 0;
}

// and use getlist to enroll them into "pointers arrays" px and py.
template < typename TYPE, class FUN, class PARAM, typename... Args >
int GpuConv1D(FUN fun, PARAM param, int nx, int ny, TYPE* x1_h, Args... args) {

    typedef typename FUN::VARSI VARSI;
    typedef typename FUN::VARSJ VARSJ;

    const int SIZEI = VARSI::SIZE+1;
    const int SIZEJ = VARSJ::SIZE;

    using DIMSX = GetDims<VARSI>;
    using DIMSY = GetDims<VARSJ>;

    using INDSI = GetInds<VARSI>;
    using INDSJ = GetInds<VARSJ>;

    TYPE *px_h[SIZEI];
    TYPE *py_h[SIZEJ];

    px_h[0] = x1_h;
    getlist<INDSI>(px_h+1,args...);
    getlist<INDSJ>(py_h,args...);

    return GpuConv1D_FromHost(fun,param,nx,ny,px_h,py_h);

}
// Idem, but with args given as an array of arrays, instead of an explicit list of arrays
template < typename TYPE, class FUN, class PARAM >
int GpuConv1D(FUN fun, PARAM param, int nx, int ny, TYPE* x1_h, TYPE** args) {
    typedef typename FUN::VARSI VARSI;
    typedef typename FUN::VARSJ VARSJ;

    const int SIZEI = VARSI::SIZE+1;
    const int SIZEJ = VARSJ::SIZE;

    using DIMSX = GetDims<VARSI>;
    using DIMSY = GetDims<VARSJ>;

    using INDSI = GetInds<VARSI>;
    using INDSJ = GetInds<VARSJ>;

    TYPE *px_h[SIZEI];
    TYPE *py_h[SIZEJ];

    px_h[0] = x1_h;
    for(int i=1; i<SIZEI; i++)
        px_h[i] = args[INDSI::VAL(i-1)];
    for(int i=0; i<SIZEJ; i++)
        py_h[i] = args[INDSJ::VAL(i)];

    return GpuConv1D_FromHost(fun,param,nx,ny,px_h,py_h);

}

// Same wrappers, but for data located on the device
template < typename TYPE, class FUN, class PARAM, typename... Args >
int GpuConv1D_FromDevice(FUN fun, PARAM param, int nx, int ny, TYPE* x1_d, Args... args) {

    typedef typename FUN::VARSI VARSI;
    typedef typename FUN::VARSJ VARSJ;

    const int SIZEI = VARSI::SIZE+1;
    const int SIZEJ = VARSJ::SIZE;

    using DIMSX = GetDims<VARSI>;
    using DIMSY = GetDims<VARSJ>;

    using INDSI = GetInds<VARSI>;
    using INDSJ = GetInds<VARSJ>;

    TYPE *px_d[SIZEI];
    TYPE *py_d[SIZEJ];

    px_d[0] = x1_d;
    getlist<INDSI>(px_d+1,args...);
    getlist<INDSJ>(py_d,args...);

    return GpuConv1D_FromDevice(fun,param,nx,ny,px_d,py_d);

}

template < typename TYPE, class FUN, class PARAM >
int GpuConv1D_FromDevice(FUN fun, PARAM param, int nx, int ny, TYPE* x1_d, TYPE** args) {
    typedef typename FUN::VARSI VARSI;
    typedef typename FUN::VARSJ VARSJ;

    const int SIZEI = VARSI::SIZE+1;
    const int SIZEJ = VARSJ::SIZE;

    using DIMSX = GetDims<VARSI>;
    using DIMSY = GetDims<VARSJ>;

    using INDSI = GetInds<VARSI>;
    using INDSJ = GetInds<VARSJ>;

    TYPE *px_d[SIZEI];
    TYPE *py_d[SIZEJ];

    px_d[0] = x1_d;
    for(int i=1; i<SIZEI; i++)
        px_d[i] = args[INDSI::VAL(i-1)];
    for(int i=0; i<SIZEJ; i++)
        py_d[i] = args[INDSJ::VAL(i)];

    return GpuConv1D_FromDevice(fun,param,nx,ny,px_d,py_d);

}
