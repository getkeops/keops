
#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <cuda.h>
#include <vector>

#include "Pack.h"

using namespace std;

template <typename TYPE, int DIMVECT>
__global__ void reduce0(TYPE* in, TYPE* out, int sizeY,int nx)
{
	TYPE res = 0;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < nx*DIMVECT)
    {
		for (int i = 0; i < sizeY; i++) 
            res += in[tid + i*nx*DIMVECT];
		/*res = in[tid+ nx* DIMVECT];*/
		out[tid] = res;
	}
}








// thread kernel: computation of x1i = sum_j k(x2i,x3i,...,y1j,y2j,...) for index i given by thread id.
template < typename TYPE, class FUN, class PARAM >
__global__ void GpuConv2DOnDevice(FUN fun, PARAM param, int nx, int ny, TYPE** px, TYPE** py)
{
	// gets dimensions and number of variables of inputs of function FUN
    typedef typename FUN::DIMSX DIMSX; // DIMSX is a "vector" of templates giving dimensions of xi variables
    typedef typename FUN::DIMSY DIMSY; // DIMSY is a "vector" of templates giving dimensions of yj variables
    const int DIMPARAM = FUN::DIMPARAM;
    const int DIMX = DIMSX::SUM; // DIMX is sum of dimensions for xi variables
    const int DIMY = DIMSY::SUM; // DIMY is sum of dimensions for yj variables
    const int DIMX1 = DIMSX::FIRST; // DIMX1 is dimension of output variable
    
    TYPE param_loc[DIMPARAM];
    for(int k=0; k<DIMPARAM; k++)
		param_loc[k] = param[k];
		
    extern __shared__ char yj_char[];
    TYPE* const yj = reinterpret_cast<TYPE*>(yj_char);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    TYPE xi[DIMX];
    TYPE tmp[DIMX1];
    if(i<nx)  // we will compute x1i only if i is in the range
    {
        for(int k=0; k<DIMX1; k++)
            tmp[k] = 0.0f; // initialize output
        // load xi from device global memory
		load<DIMSX::NEXT>(i,xi+DIMX1,px+1); // load xi variables from global memory to local thread memory
    }
    
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    if(j<ny) // we load yj from device global memory only if j<ny
		load<DIMSY>(j,yj+threadIdx.x*DIMY,py); // load yj variables from global memory to shared memory
		   	
    __syncthreads();
        
    if(i<nx) // we compute x1i only if needed
    {
    	TYPE* yjrel = yj;
        for(int jrel = 0; (jrel<blockDim.x) && ((blockDim.x*blockIdx.y+jrel)< ny); jrel++, yjrel+=DIMY)
        {
			call<DIMSX,DIMSY>(fun,xi,yjrel,param_loc); // call function
			for(int k=0; k<DIMX1; k++)
				tmp[k] += xi[k];
		}
        __syncthreads();
    }

    // Save the result in global memory.
    if(i<nx)
        for(int k=0; k<DIMX1; k++)
            (*px)[blockIdx.y*DIMX1*nx+i*DIMX1+k] = tmp[k];
}
///////////////////////////////////////////////////


template < typename TYPE, class FUN, class PARAM >
int GpuConv2D(FUN fun, PARAM param_h, int nx, int ny, TYPE** px_h, TYPE** py_h)
{
    typedef typename FUN::DIMSX DIMSX;
    typedef typename FUN::DIMSY DIMSY;
    const int DIMPARAM = FUN::DIMPARAM;
    const int DIMX = DIMSX::SUM;
    const int DIMY = DIMSY::SUM;
    const int DIMX1 = DIMSX::FIRST;
    const int SIZEX = DIMSX::SIZE;
    const int SIZEY = DIMSY::SIZE;
    
    // Data on the device.
    TYPE *x1B, *x_d, *y_d, *param_d;

    TYPE **px_d, **py_d;
    cudaHostAlloc((void**)&px_d, SIZEX*sizeof(TYPE*), cudaHostAllocMapped);
    cudaHostAlloc((void**)&py_d, SIZEY*sizeof(TYPE*), cudaHostAllocMapped);

    // Allocate arrays on device.
    cudaMalloc((void**)&x_d, sizeof(TYPE)*(nx*DIMX));
    cudaMalloc((void**)&y_d, sizeof(TYPE)*(ny*DIMY));
    cudaMalloc((void**)&param_d, sizeof(TYPE)*(DIMPARAM));

    // Send data from host to device.
    
    cudaMemcpy(param_d, param_h, sizeof(TYPE)*DIMPARAM, cudaMemcpyHostToDevice);

    int nvals;
    px_d[0] = x_d;
    nvals = nx*DIMSX::VAL(0);
    for(int k=1; k<SIZEX; k++)
    {
        px_d[k] = px_d[k-1] + nvals;
        nvals = nx*DIMSX::VAL(k);
        cudaMemcpy(px_d[k], px_h[k], sizeof(TYPE)*nvals, cudaMemcpyHostToDevice);
    }
    py_d[0] = y_d;
    nvals = ny*DIMSY::VAL(0);
    cudaMemcpy(py_d[0], py_h[0], sizeof(TYPE)*nvals, cudaMemcpyHostToDevice);
    for(int k=1; k<SIZEY; k++)
    {
        py_d[k] = py_d[k-1] + nvals;
        nvals = ny*DIMSY::VAL(k);
        cudaMemcpy(py_d[k], py_h[k], sizeof(TYPE)*nvals, cudaMemcpyHostToDevice);
    }

    // Compute on device.
    dim3 blockSize;
    blockSize.x = 192; // number of threads in each block
    int blockSizey = blockSize.x;
    dim3 gridSize;
    gridSize.x =  nx / blockSize.x + (nx%blockSize.x==0 ? 0 : 1);
	gridSize.y =  ny / blockSizey + (ny%blockSizey==0 ? 0 : 1);

    // Reduce  : grid and block are 1d
    dim3 blockSize2;
    blockSize2.x = 192; // number of threads in each block
    dim3 gridSize2;
    gridSize2.x =  (nx*DIMX1) / blockSize2.x + ((nx*DIMX1)%blockSize2.x==0 ? 0 : 1);

    cudaMalloc((void**)&x1B, sizeof(TYPE)*(nx*DIMX1*gridSize.y));
    px_d[0] = x1B;

    GpuConv2DOnDevice<TYPE><<<gridSize,blockSize,blockSize.x*(DIMY)*sizeof(TYPE)>>>(fun,param_d,nx,ny,px_d,py_d);
	
    reduce0<TYPE,DIMX1><<<gridSize2, blockSize2>>>(x1B, x_d, gridSize.y,nx);
        
    // block until the device has completed
    cudaThreadSynchronize();

    // Send data from device to host.
    cudaMemcpy(*px_h, x_d, sizeof(TYPE)*(nx*DIMX1),cudaMemcpyDeviceToHost);

    // Free memory.
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(x1B);
    cudaFreeHost(px_d);
    cudaFreeHost(py_d);

    return 0;
}

template < typename TYPE, class FUN, class PARAM, typename... Args >
int GpuConv2D(FUN fun, PARAM param, int nx, int ny, TYPE* x1_h, Args... args)
{

    typedef typename FUN::VARSI VARSI;
    typedef typename FUN::VARSJ VARSJ;
    
    const int SIZEX = VARSI::SIZE+1;
    const int SIZEY = VARSJ::SIZE;

    using DIMSX = GetDims<VARSI>; 
    using DIMSY = GetDims<VARSJ>; 
    
    using INDSX = GetInds<VARSI>; 
    using INDSY = GetInds<VARSJ>; 

    TYPE *px_h[SIZEX];
    TYPE *py_h[SIZEY];
    
    px_h[0] = x1_h;
    getlist_new<INDSX>(px_h+1,args...);
    getlist_new<INDSY>(py_h,args...);

	return GpuConv2D(fun,param,nx,ny,px_h,py_h);

}




// Host implementation of the convolution, for comparison


template < typename TYPE, class FUN, class PARAM >
int CpuConv(FUN fun, PARAM param, int nx, int ny, TYPE** px, TYPE** py)
{	
    typedef typename FUN::DIMSX DIMSX;
    typedef typename FUN::DIMSY DIMSY;
    const int DIMX = DIMSX::SUM;
    const int DIMY = DIMSY::SUM;
    const int DIMX1 = DIMSX::FIRST;

	TYPE xi[DIMX], yj[DIMY], tmp[DIMX1];
	for(int i=0; i<nx; i++)
	{
		load<DIMSX>(i,xi,px);
		for(int k=0; k<DIMX1; k++)
			tmp[k] = 0;		
		for(int j=0; j<ny; j++)
		{
			load<DIMSY>(j,yj,py);
			call<DIMSX,DIMSY>(fun,xi,yj,param);
			for(int k=0; k<DIMX1; k++)
				tmp[k] += xi[k];
		}
		for(int k=0; k<DIMX1; k++)
			px[0][i*DIMX1+k] = tmp[k];
	}   
	
	return 0;
}

template < typename TYPE, class FUN, class PARAM, typename... Args >
int CpuConv(FUN fun, PARAM param, int nx, int ny, TYPE* x1, Args... args)
{
    typedef typename FUN::VARSI VARSI;
    typedef typename FUN::VARSJ VARSJ;
    
    const int SIZEX = VARSI::SIZE+1;
    const int SIZEY = VARSJ::SIZE;

    using DIMSX = GetDims<VARSI>; 
    using DIMSY = GetDims<VARSJ>; 
    
    using INDSX = GetInds<VARSI>; 
    using INDSY = GetInds<VARSJ>; 

    TYPE *px[SIZEX];
    TYPE *py[SIZEY];

    px[0] = x1;
    getlist_new<INDSX>(px+1,args...);
    getlist_new<INDSY>(py,args...);

	return CpuConv(fun,param,nx,ny,px,py);
}



template < class F, class VARSI_, class VARSJ_, int DIMPARAM_ >
class Generic
{
	
	public :
	
	struct sEval // static wrapper
	{
		using VARSI = VARSI_;
		using VARSJ = VARSJ_;
		using DIMSX = typename GetDims<VARSI>::PUTLEFT<F::DIM>;	// dimensions of "i" variables 
		using DIMSY = GetDims<VARSJ>; 	// dimensions of "j" variables  
		static const int DIMPARAM = DIMPARAM_;
		
		template < typename... Args >
		__host__ __device__ __forceinline__ void operator()(Args... args)
		{
			F::Eval(args...);
		}
	};
	
};



