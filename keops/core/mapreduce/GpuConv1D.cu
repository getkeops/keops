#pragma once

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <cuda.h>

#include "core/pack/Pack.h"
#include "core/pack/Load.h"
#include "core/pack/Load_Chunks.h"
#include "core/pack/Call.h"
#include "core/pack/GetInds.h"
#include "core/pack/GetDims.h"
#include "core/utils/CudaErrorCheck.cu"
#include "core/utils/CudaSizes.h"
#include "core/utils/TypesUtils.h"

#include "Chunk_Mode_Constants.h"


namespace keops {


template < int USE_CHUNK_MODE, int BLOCKSIZE_CHUNKS, class VARFINAL=void > struct GpuConv1DOnDevice {};


#if USE_FINAL_CHUNKS==1


template < class VARFINAL, int DIMFINALCHUNK_CURR, typename TYPE >
__device__ void do_finalchunk_sub(TYPE *acc, int i, int j, int jstart, int chunk, int nx, int ny, 
			TYPE **args, TYPE *fout, TYPE *yj, TYPE *out) {
                
            static const int DIMOUT = VARFINAL::DIM;
                
            VectAssign<DIMFINALCHUNK>(acc,0.0f); //typename FUN::template InitializeReduction<__TYPEACC__, TYPE >()(acc); // acc = 0
            TYPE *yjrel = yj;
            if (j < ny) // we load yj from device global memory only if j<ny
                    load_chunks < pack<VARFINAL::N>, DIMFINALCHUNK, DIMFINALCHUNK_CURR, VARFINAL::DIM >(j, chunk, yj + threadIdx.x * DIMFINALCHUNK, args);
            __syncthreads();
            for (int jrel = 0; (jrel < blockDim.x) && (jrel < ny - jstart); jrel++, yjrel += DIMFINALCHUNK) {          
                if (i < nx) { // we compute only if needed
                    #pragma unroll
                    for (int k=0; k<DIMFINALCHUNK_CURR; k++)
                        acc[k] += yjrel[k] * fout[jrel];
                }
                __syncthreads();
            }
            if (i < nx) {
                //typename FUN::template FinalizeOutput<__TYPEACC__,TYPE>()(acc, out + i * DIMOUT, i);
                #pragma unroll
                for (int k=0; k<DIMFINALCHUNK_CURR; k++)
                    out[i*DIMOUT+chunk*DIMFINALCHUNK+k] += acc[k];
            }
            __syncthreads();
        }

template < int BLOCKSIZE_CHUNKS, class VARFINAL, typename TYPE, class FUN >
__global__ void GpuConv1DOnDevice_FinalChunks(FUN fun, int nx, int ny, TYPE *out, TYPE **args) {
    
    // get the index of the current thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // declare shared mem
    extern __shared__ TYPE yj[];
    
    const int NCHUNKS = 1 + (VARFINAL::DIM-1) / DIMFINALCHUNK;
    const int DIMLASTFINALCHUNK = VARFINAL::DIM - (NCHUNKS-1)*DIMFINALCHUNK;
    
    // get templated dimensions :
    typedef typename FUN::DIMSX DIMSX;  // DIMSX is a "vector" of templates giving dimensions of xi variables
    typedef typename FUN::DIMSY DIMSY;  // DIMSY is a "vector" of templates giving dimensions of yj variables
    typedef typename FUN::DIMSP DIMSP;  // DIMSP is a "vector" of templates giving dimensions of parameters variables
    typedef typename FUN::INDSI INDSI;
    typedef typename FUN::INDSJ INDSJ;
    typedef typename FUN::INDSP INDSP;
    const int DIMX = DIMSX::SUM;        // DIMX  is sum of dimensions for xi variables
    const int DIMY = DIMSY::SUM;        // DIMY  is sum of dimensions for yj variables
    const int DIMP = DIMSP::SUM;        // DIMP  is sum of dimensions for parameters variables
    const int DIMOUT = VARFINAL::DIM; // dimension of output variable
    const int DIMFOUT = FUN::F::DIM;     // DIMFOUT is dimension of output variable of inner function

    static_assert(DIMFOUT==1,"DIMFOUT should be 1");
    
    static_assert(SUM_SCHEME==BLOCK_SUM,"only BLOCK_SUM available");
            
    // load parameter(s)
    TYPE param_loc[DIMP < 1 ? 1 : DIMP];
    load<DIMSP, INDSP>(0, param_loc, args); // load parameters variables from global memory to local thread memory

    TYPE fout[DIMFOUT*BLOCKSIZE_CHUNKS];
    
    // get the value of variable (index with i)
    TYPE xi[DIMX < 1 ? 1 : DIMX];
    if (i < nx) {
        load<DIMSX, INDSI>(i, xi, args); // load xi variables from global memory to local thread memory
        
        #pragma unroll
        for (int k=0; k<DIMOUT; k++) {
            out[i*DIMOUT+k] = 0.0f;
        }
    }
    
    __TYPEACC__ acc[DIMFINALCHUNK];

    for (int jstart = 0, tile = 0; jstart < ny; jstart += blockDim.x, tile++) {

        // get the current column
        int j = tile * blockDim.x + threadIdx.x;

        if (j < ny) { // we load yj from device global memory only if j<ny
            load<DIMSY,INDSJ>(j, yj + threadIdx.x * DIMY, args); // load yj variables from global memory to shared memory
        }
        __syncthreads();

        if (i < nx) { // we compute x1i only if needed
            TYPE * yjrel = yj; // Loop on the columns of the current block.
            for (int jrel = 0; (jrel < BLOCKSIZE_CHUNKS) && (jrel < ny - jstart); jrel++, yjrel += DIMY) {
                call<DIMSX, DIMSY, DIMSP>(fun,
                                  fout+jrel*DIMFOUT,
                                  xi,
                                  yjrel,
                                  param_loc); // Call the function, which outputs results in fout
            }
        }
        
        __syncthreads();
        
        for (int chunk=0; chunk<NCHUNKS-1; chunk++) 
            do_finalchunk_sub < VARFINAL, DIMFINALCHUNK > (acc, i, j, jstart, chunk, nx, ny, args, fout, yj, out);
        do_finalchunk_sub < VARFINAL, DIMLASTFINALCHUNK > (acc, i, j, jstart, NCHUNKS-1, nx, ny, args, fout, yj, out);
    }
}

template < int BLOCKSIZE_CHUNKS, class VARFINAL > 
struct GpuConv1DOnDevice<2,BLOCKSIZE_CHUNKS,VARFINAL> {
    template < typename TYPE, class FUN >
    static void Eval(dim3 gridSize, dim3 blockSize, size_t SharedMem, FUN fun, int nx, int ny, TYPE *out, TYPE **args) {
        GpuConv1DOnDevice_FinalChunks < BLOCKSIZE_CHUNKS, VARFINAL > <<< gridSize, blockSize, SharedMem >>> (fun, nx, ny, out, args);
    }
};

#endif


template < class FUN, class FUN_CHUNKED_CURR, int DIMCHUNK_CURR, typename TYPE >
__device__ void do_chunk_sub(TYPE *acc, int tile, int i, int j, int jstart, int chunk, int nx, int ny, 
			TYPE **args, TYPE *fout, TYPE *xi, TYPE *yj, TYPE *param_loc) {

	using CHK = Chunk_Mode_Constants<FUN>;

	TYPE fout_tmp_chunk[CHK::FUN_CHUNKED::DIM];
	
	if (i < nx) 
		load_chunks < typename CHK::INDSI_CHUNKED, DIMCHUNK, DIMCHUNK_CURR, CHK::DIM_ORG >(i, chunk, xi + CHK::DIMX_NOTCHUNKED, args);

	if (j < ny) // we load yj from device global memory only if j<ny
		load_chunks < typename CHK::INDSJ_CHUNKED, DIMCHUNK, DIMCHUNK_CURR, CHK::DIM_ORG > (j, chunk, yj + threadIdx.x * CHK::DIMY + CHK::DIMY_NOTCHUNKED, args);

	__syncthreads();

	if (i < nx) { // we compute only if needed
		TYPE *yjrel = yj; // Loop on the columns of the current block.
		for (int jrel = 0; (jrel < blockDim.x) && (jrel < ny - jstart); jrel++, yjrel += CHK::DIMY) {
			TYPE *foutj = fout+jrel*CHK::FUN_CHUNKED::DIM;
			call < CHK::DIMSX, CHK::DIMSY, CHK::DIMSP > 
				(FUN_CHUNKED_CURR::template EvalFun<CHK::INDS>(), fout_tmp_chunk, xi, yjrel, param_loc);
			CHK::FUN_CHUNKED::acc_chunk(foutj, fout_tmp_chunk);
		}
	}
	__syncthreads();
}


template < int BLOCKSIZE_CHUNKS, typename TYPE, class FUN >
__global__ void GpuConv1DOnDevice_Chunks(FUN fun, int nx, int ny, TYPE *out, TYPE **args) {

	using CHK = Chunk_Mode_Constants<FUN>;

	// get the index of the current thread
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	// declare shared mem
	extern __shared__ TYPE yj[];

	// load parameter(s)
	TYPE param_loc[CHK::DIMP < 1 ? 1 : CHK::DIMP];
	load<CHK::DIMSP,CHK::INDSP>(0, param_loc, args); // load parameters variables from global memory to local thread memory

	__TYPEACC__ acc[CHK::DIMRED];

#if SUM_SCHEME == BLOCK_SUM
    // additional tmp vector to store intermediate results from each block
    TYPE tmp[CHK::DIMRED];
#elif SUM_SCHEME == KAHAN_SCHEME
    // additional tmp vector to accumulate errors
    static const int DIM_KAHAN = FUN::template KahanScheme<__TYPEACC__,TYPE>::DIMACC;
    TYPE tmp[DIM_KAHAN];
#endif

	if (i < nx) {
		typename FUN::template InitializeReduction<__TYPEACC__, TYPE >()(acc); // acc = 0
#if SUM_SCHEME == KAHAN_SCHEME
		VectAssign<DIM_KAHAN>(tmp,0.0f);
#endif		
	}

	TYPE xi[CHK::DIMX];

	TYPE fout_chunk[BLOCKSIZE_CHUNKS*CHK::DIMOUT_CHUNK];
	
	if (i < nx)
		load < CHK::DIMSX_NOTCHUNKED, CHK::INDSI_NOTCHUNKED > (i, xi, args); // load xi variables from global memory to local thread memory

	for (int jstart = 0, tile = 0; jstart < ny; jstart += blockDim.x, tile++) {

		// get the current column
		int j = tile * blockDim.x + threadIdx.x;
	
		if (j < ny) 
			load<CHK::DIMSY_NOTCHUNKED, CHK::INDSJ_NOTCHUNKED>(j, yj + threadIdx.x * CHK::DIMY, args);

		__syncthreads();

		if (i < nx) { // we compute only if needed
			for (int jrel = 0; (jrel < blockDim.x) && (jrel < ny - jstart); jrel++)
				CHK::FUN_CHUNKED::initacc_chunk(fout_chunk+jrel*CHK::DIMOUT_CHUNK);
#if SUM_SCHEME == BLOCK_SUM
			typename FUN::template InitializeReduction<TYPE,TYPE>()(tmp); // tmp = 0
#endif
		}

		// looping on chunks (except the last)
		#pragma unroll
		for (int chunk=0; chunk<CHK::NCHUNKS-1; chunk++)
			do_chunk_sub < FUN, CHK::FUN_CHUNKED, DIMCHUNK >
				(acc, tile, i, j, jstart, chunk, nx, ny, args, fout_chunk, xi, yj, param_loc);	
		// last chunk
		do_chunk_sub < FUN, CHK::FUN_LASTCHUNKED, CHK::DIMLASTCHUNK >
			(acc, tile, i, j, jstart, CHK::NCHUNKS-1, nx, ny, args, fout_chunk, xi, yj, param_loc);

		if (i < nx) { 
			TYPE * yjrel = yj; // Loop on the columns of the current block.
			for (int jrel = 0; (jrel < blockDim.x) && (jrel < ny - jstart); jrel++, yjrel += CHK::DIMY) {
#if SUM_SCHEME != KAHAN_SCHEME
				int ind = jrel + tile * blockDim.x;
#endif
				TYPE *foutj = fout_chunk + jrel*CHK::DIMOUT_CHUNK;
				TYPE fout_tmp[CHK::DIMFOUT];
				call<CHK::DIMSX, CHK::DIMSY, CHK::DIMSP, pack<CHK::DIMOUT_CHUNK> >
						(typename CHK::FUN_POSTCHUNK::template EvalFun<ConcatPacks<typename CHK::INDS,pack<FUN::NMINARGS>>>(), 
						fout_tmp,xi, yjrel, param_loc, foutj);
#if SUM_SCHEME == BLOCK_SUM
#if USE_HALF
        			typename FUN::template ReducePairShort<TYPE,TYPE>()(tmp, fout_tmp, __floats2half2_rn(2*ind,2*ind+1));     // tmp += fout_tmp
#else
				typename FUN::template ReducePairShort<TYPE,TYPE>()(tmp, fout_tmp, ind);     // tmp += fout_tmp
#endif
#elif SUM_SCHEME == KAHAN_SCHEME
				typename FUN::template KahanScheme<__TYPEACC__,TYPE>()(acc, fout_tmp, tmp);     
#else
#if USE_HALF
				typename FUN::template ReducePairShort<__TYPEACC__,TYPE>()(acc, fout_tmp, __floats2half2_rn(2*ind,2*ind+1));     // acc += fout_tmp
#else
				typename FUN::template ReducePairShort<__TYPEACC__,TYPE>()(acc, fout_tmp, ind);     // acc += fout_tmp
#endif
#endif
			}
#if SUM_SCHEME == BLOCK_SUM
			typename FUN::template ReducePair<__TYPEACC__,TYPE>()(acc, tmp);     // acc += tmp
#endif
		}
	__syncthreads();
	}
	
	if (i < nx) 
		typename FUN::template FinalizeOutput<__TYPEACC__,TYPE>()(acc, out + i * CHK::DIMOUT, i);
}


template < int BLOCKSIZE_CHUNKS > 
struct GpuConv1DOnDevice<1,BLOCKSIZE_CHUNKS> {
    template < typename TYPE, class FUN >
    static void Eval(dim3 gridSize, dim3 blockSize, size_t SharedMem, FUN fun, int nx, int ny, TYPE *out, TYPE **args) {
        GpuConv1DOnDevice_Chunks < BLOCKSIZE_CHUNKS > <<< gridSize, blockSize, SharedMem >>> (fun, nx, ny, out, args);
    }
};





template<typename TYPE, class FUN>
__global__ void GpuConv1DOnDevice_NoChunks(FUN fun, int nx, int ny, TYPE *out, TYPE **args) {

  // get the index of the current thread
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // declare shared mem
  extern __shared__ TYPE yj[];

  // get templated dimensions :
  typedef typename FUN::DIMSX DIMSX;  // DIMSX is a "vector" of templates giving dimensions of xi variables
  typedef typename FUN::DIMSY DIMSY;  // DIMSY is a "vector" of templates giving dimensions of yj variables
  typedef typename FUN::DIMSP DIMSP;  // DIMSP is a "vector" of templates giving dimensions of parameters variables
    typedef typename FUN::INDSI INDSI;
    typedef typename FUN::INDSJ INDSJ;
    typedef typename FUN::INDSP INDSP;
  const int DIMX = DIMSX::SUM;        // DIMX  is sum of dimensions for xi variables
  const int DIMY = DIMSY::SUM;        // DIMY  is sum of dimensions for yj variables
  const int DIMP = DIMSP::SUM;        // DIMP  is sum of dimensions for parameters variables
  const int DIMOUT = FUN::DIM; // dimension of output variable
  const int DIMRED = FUN::DIMRED; // dimension of reduction operation
  const int DIMFOUT = FUN::F::DIM;     // DIMFOUT is dimension of output variable of inner function

  // load parameter(s)
  TYPE param_loc[DIMP < 1 ? 1 : DIMP];
  load<DIMSP, INDSP>(0, param_loc, args); // load parameters variables from global memory to local thread memory

  TYPE fout[DIMFOUT];
  // get the value of variable (index with i)
  TYPE xi[DIMX < 1 ? 1 : DIMX];
  __TYPEACC__ acc[DIMRED];
#if SUM_SCHEME == BLOCK_SUM
    // additional tmp vector to store intermediate results from each block
    TYPE tmp[DIMRED];
#elif SUM_SCHEME == KAHAN_SCHEME
    // additional tmp vector to accumulate errors
    const int DIM_KAHAN = FUN::template KahanScheme<__TYPEACC__,TYPE>::DIMACC;
    TYPE tmp[DIM_KAHAN];
#endif
  if (i < nx) {
    typename FUN::template InitializeReduction<__TYPEACC__, TYPE >()(acc); // acc = 0
#if SUM_SCHEME == KAHAN_SCHEME
    VectAssign<DIM_KAHAN>(tmp,0.0f);
#endif
    load<DIMSX, INDSI>(i, xi, args); // load xi variables from global memory to local thread memory
  }

  for (int jstart = 0, tile = 0; jstart < ny; jstart += blockDim.x, tile++) {

    // get the current column
    int j = tile * blockDim.x + threadIdx.x;

    if (j < ny) { // we load yj from device global memory only if j<ny
      load<DIMSY,INDSJ>(j, yj + threadIdx.x * DIMY, args); // load yj variables from global memory to shared memory
    }
    __syncthreads();

    if (i < nx) { // we compute x1i only if needed
      TYPE * yjrel = yj; // Loop on the columns of the current block.
#if SUM_SCHEME == BLOCK_SUM
      typename FUN::template InitializeReduction<TYPE,TYPE>()(tmp); // tmp = 0
#endif
      for (int jrel = 0; (jrel < blockDim.x) && (jrel < ny - jstart); jrel++, yjrel += DIMY) {
        call<DIMSX, DIMSY, DIMSP>(fun,
				  fout,
                                  xi,
                                  yjrel,
                                  param_loc); // Call the function, which outputs results in fout
#if SUM_SCHEME == BLOCK_SUM
#if USE_HALF
        int ind = jrel + tile * blockDim.x;
        typename FUN::template ReducePairShort<TYPE,TYPE>()(tmp, fout, __floats2half2_rn(2*ind,2*ind+1));     // tmp += fout
#else
        typename FUN::template ReducePairShort<TYPE,TYPE>()(tmp, fout, jrel + tile * blockDim.x);     // tmp += fout
#endif
#elif SUM_SCHEME == KAHAN_SCHEME
        typename FUN::template KahanScheme<__TYPEACC__,TYPE>()(acc, fout, tmp);     
#else
#if USE_HALF
        int ind = jrel + tile * blockDim.x;
        typename FUN::template ReducePairShort<__TYPEACC__,TYPE>()(acc, fout, __floats2half2_rn(2*ind,2*ind+1));     // acc += fout
#else
	typename FUN::template ReducePairShort<__TYPEACC__,TYPE>()(acc, fout, jrel + tile * blockDim.x);     // acc += fout
#endif
#endif
      }
#if SUM_SCHEME == BLOCK_SUM
      typename FUN::template ReducePair<__TYPEACC__,TYPE>()(acc, tmp);     // acc += tmp
#endif
    }
    __syncthreads();
  }
  if (i < nx) {
    typename FUN::template FinalizeOutput<__TYPEACC__,TYPE>()(acc, out + i * DIMOUT, i);
  }

}


template < int DUMMY > 
struct GpuConv1DOnDevice<0,DUMMY> {
    template < typename TYPE, class FUN >
    static void Eval(dim3 gridSize, dim3 blockSize, size_t SharedMem, FUN fun, int nx, int ny, TYPE *out, TYPE **args) {
        GpuConv1DOnDevice_NoChunks <<< gridSize, blockSize, SharedMem >>> (fun, nx, ny, out, args);
    }
};



struct GpuConv1D_FromHost {

  template<typename TYPE, class FUN>
  static int Eval_(FUN fun, int nx, int ny, TYPE *out, TYPE **args_h) {

    typedef typename FUN::DIMSX DIMSX;
    typedef typename FUN::DIMSY DIMSY;
    typedef typename FUN::DIMSP DIMSP;
    typedef typename FUN::INDSI INDSI;
    typedef typename FUN::INDSJ INDSJ;
    typedef typename FUN::INDSP INDSP;
    const int DIMX = DIMSX::SUM;
    const int DIMY = DIMSY::SUM;
    const int DIMP = DIMSP::SUM;
    const int DIMOUT = FUN::DIM; // dimension of output variable
    const int SIZEI = DIMSX::SIZE;
    const int SIZEJ = DIMSY::SIZE;
    const int SIZEP = DIMSP::SIZE;
    static const int NMINARGS = FUN::NMINARGS;

    // pointer to device output array
    TYPE *out_d;

    // array of pointers to device input arrays
    TYPE **args_d;

    void *p_data;
    // single cudaMalloc
    CudaSafeCall(cudaMalloc(&p_data,
                            sizeof(TYPE *) * NMINARGS
                                + sizeof(TYPE) * (DIMP + nx * (DIMX + DIMOUT) + ny * DIMY)));

    args_d = (TYPE **) p_data;
    TYPE *dataloc = (TYPE *) (args_d + NMINARGS);
    out_d = dataloc;
    dataloc += nx*DIMOUT;

    // host array of pointers to device data
    TYPE *ph[NMINARGS];

      for (int k = 0; k < SIZEP; k++) {
        int indk = INDSP::VAL(k);
        int nvals = DIMSP::VAL(k);        
        CudaSafeCall(cudaMemcpy(dataloc, args_h[indk], sizeof(TYPE) * nvals, cudaMemcpyHostToDevice));
        ph[indk] = dataloc;
        dataloc += nvals;
      }

    for (int k = 0; k < SIZEI; k++) {
      int indk = INDSI::VAL(k);
      int nvals = nx * DIMSX::VAL(k);
      CudaSafeCall(cudaMemcpy(dataloc, args_h[indk], sizeof(TYPE) * nvals, cudaMemcpyHostToDevice));
      ph[indk] = dataloc;
      dataloc += nvals;
    }

      for (int k = 0; k < SIZEJ; k++) {
        int indk = INDSJ::VAL(k);
        int nvals = ny * DIMSY::VAL(k);
        CudaSafeCall(cudaMemcpy(dataloc, args_h[indk], sizeof(TYPE) * nvals, cudaMemcpyHostToDevice));
        ph[indk] = dataloc;
        dataloc += nvals;
      }


    // copy array of pointers
    CudaSafeCall(cudaMemcpy(args_d, ph, NMINARGS * sizeof(TYPE *), cudaMemcpyHostToDevice));

    // Compute on device : grid and block are both 1d
    int dev = -1;
    CudaSafeCall(cudaGetDevice(&dev));

    dim3 blockSize;

    SetGpuProps(dev);

    #if USE_FINAL_CHUNKS==1
        static const int USE_CHUNK_MODE = 2;
        using FUN_INTERNAL = Sum_Reduction<typename FUN::F::ARG1,FUN::tagI>;
        using VARFINAL = typename FUN::F::ARG2;
    #else
        static const int USE_CHUNK_MODE = ENABLECHUNK && ( FUN::F::template CHUNKED_FORMULAS<DIMCHUNK>::SIZE == 1 );
    #endif
    
    static const int DIMY_SHARED = Get_DIMY_SHARED<FUN,USE_CHUNK_MODE>::Value;

    static const int BLOCKSIZE_CHUNKS = ::std::min(CUDA_BLOCK_SIZE,
                             ::std::min(1024,
                                        (int) (49152 / ::std::max(1,
                                                    (int) (  DIMY_SHARED * sizeof(TYPE))))));

    int blocksize_nochunks = ::std::min(CUDA_BLOCK_SIZE,
                             ::std::min(maxThreadsPerBlock,
                                        (int) (sharedMemPerBlock / ::std::max(1,
                                                    (int) (  DIMY_SHARED * sizeof(TYPE))))));

    blockSize.x = USE_CHUNK_MODE ? BLOCKSIZE_CHUNKS : blocksize_nochunks;

    dim3 gridSize;
    gridSize.x = nx / blockSize.x + (nx % blockSize.x == 0 ? 0 : 1);

    #if USE_FINAL_CHUNKS==1
        GpuConv1DOnDevice<USE_CHUNK_MODE,BLOCKSIZE_CHUNKS,VARFINAL>::Eval(gridSize, blockSize, blockSize.x * DIMY_SHARED * sizeof(TYPE), FUN_INTERNAL(), nx, ny, out_d, args_d);
    #else
        GpuConv1DOnDevice<USE_CHUNK_MODE,BLOCKSIZE_CHUNKS>::Eval(gridSize, blockSize, blockSize.x * DIMY_SHARED * sizeof(TYPE), fun, nx, ny, out_d, args_d);
    #endif
    
    // block until the device has completed
    CudaSafeCall(cudaDeviceSynchronize());
    CudaCheckError();

    // Send data from device to host.
    CudaSafeCall(cudaMemcpy(out, out_d, sizeof(TYPE) * (nx * DIMOUT), cudaMemcpyDeviceToHost));

    // Free memory.
    CudaSafeCall(cudaFree(p_data));

    return 0;
  }

// and use getlist to enroll them into "pointers arrays" px and py.
  template<typename TYPE, class FUN, typename... Args>
  static int Eval(FUN fun, int nx, int ny, int device_id, TYPE *out, Args... args) {

    if (device_id != -1)
      CudaSafeCall(cudaSetDevice(device_id));

    static const int Nargs = sizeof...(Args);
    TYPE *pargs[Nargs];
    unpack(pargs, args...);

    return Eval_(fun, nx, ny, out, pargs);

  }

// same without the device_id argument
  template<typename TYPE, class FUN, typename... Args>
  static int Eval(FUN fun, int nx, int ny, TYPE *out, Args... args) {
    return Eval(fun, nx, ny, -1, out, args...);
  }

// Idem, but with args given as an array of arrays, instead of an explicit list of arrays
  template<typename TYPE, class FUN>
  static int Eval(FUN fun, int nx, int ny, TYPE *out, TYPE **pargs, int device_id = -1) {

    // We set the GPU device on which computations will be performed
    if (device_id != -1)
      CudaSafeCall(cudaSetDevice(device_id));

    return Eval_(fun, nx, ny, out, pargs);

  }

};

struct GpuConv1D_FromDevice {
  template<typename TYPE, class FUN>
  static int Eval_(FUN fun, int nx, int ny, TYPE *out, TYPE **args) {


    static const int NMINARGS = FUN::NMINARGS;

    // device array of pointers to device data
    TYPE **args_d;

    // single cudaMalloc
    CudaSafeCall(cudaMalloc(&args_d, sizeof(TYPE *) * NMINARGS));

    CudaSafeCall(cudaMemcpy(args_d, args, NMINARGS * sizeof(TYPE *), cudaMemcpyHostToDevice));

    // Compute on device : grid and block are both 1d

    int dev = -1;
    CudaSafeCall(cudaGetDevice(&dev));

    SetGpuProps(dev);

    dim3 blockSize;

    #if USE_FINAL_CHUNKS==1
        static const int USE_CHUNK_MODE = 2;
        using FUN_INTERNAL = Sum_Reduction<typename FUN::F::ARG1,FUN::tagI>;
        using VARFINAL = typename FUN::F::ARG2;
    #else
        static const int USE_CHUNK_MODE = ENABLECHUNK && ( FUN::F::template CHUNKED_FORMULAS<DIMCHUNK>::SIZE == 1 );
    #endif
    
    static const int DIMY_SHARED = Get_DIMY_SHARED<FUN,USE_CHUNK_MODE>::Value;

    static const int BLOCKSIZE_CHUNKS = ::std::min(CUDA_BLOCK_SIZE,
                             ::std::min(1024,
                                        (int) (49152 / ::std::max(1,
                                                    (int) (  DIMY_SHARED * sizeof(TYPE))))));

    int blocksize_nochunks = ::std::min(CUDA_BLOCK_SIZE,
                             ::std::min(maxThreadsPerBlock,
                                        (int) (sharedMemPerBlock / ::std::max(1,
                                                    (int) (  DIMY_SHARED * sizeof(TYPE))))));

    blockSize.x = USE_CHUNK_MODE ? BLOCKSIZE_CHUNKS : blocksize_nochunks;
	
    dim3 gridSize;
    gridSize.x = nx / blockSize.x + (nx % blockSize.x == 0 ? 0 : 1);

    #if USE_FINAL_CHUNKS==1
        GpuConv1DOnDevice<USE_CHUNK_MODE,BLOCKSIZE_CHUNKS,VARFINAL>::Eval(gridSize, blockSize, blockSize.x * DIMY_SHARED * sizeof(TYPE), FUN_INTERNAL(), nx, ny, out, args_d);
    #else
        GpuConv1DOnDevice<USE_CHUNK_MODE,BLOCKSIZE_CHUNKS>::Eval(gridSize, blockSize, blockSize.x * DIMY_SHARED * sizeof(TYPE), fun, nx, ny, out, args_d);
    #endif
    
    // block until the device has completed
    CudaSafeCall(cudaDeviceSynchronize());

    CudaCheckError();

    CudaSafeCall(cudaFree(args_d));

    return 0;
  }

// Same wrappers, but for data located on the device
  template<typename TYPE, class FUN, typename... Args>
  static int Eval(FUN fun, int nx, int ny, int device_id, TYPE *out, Args... args) {

    // device_id is provided, so we set the GPU device accordingly
    // Warning : is has to be consistent with location of data
    CudaSafeCall(cudaSetDevice(device_id));

    static const int Nargs = sizeof...(Args);
    TYPE *pargs[Nargs];
    unpack(pargs, args...);

    return Eval_(fun, nx, ny, out, pargs);

  }

// same without the device_id argument
  template<typename TYPE, class FUN, typename... Args>
  static int Eval(FUN fun, int nx, int ny, TYPE *out, Args... args) {
    // We set the GPU device on which computations will be performed
    // to be the GPU on which data is located.
    // NB. we only check location of x1_d which is the output vector
    // so we assume that input data is on the same GPU
    // note : cudaPointerGetAttributes has a strange behaviour:
    // it looks like it makes a copy of the vector on the default GPU device (0) !!! 
    // So we prefer to avoid this and provide directly the device_id as input (first function above)
    cudaPointerAttributes attributes;
    CudaSafeCall(cudaPointerGetAttributes(&attributes, out));
    return Eval(fun, nx, ny, attributes.device, out, args...);
  }

  template<typename TYPE, class FUN>
  static int Eval(FUN fun, int nx, int ny, TYPE *out, TYPE **pargs, int device_id = -1) {

    if (device_id == -1) {
      // We set the GPU device on which computations will be performed
      // to be the GPU on which data is located.
      // NB. we only check location of x1_d which is the output vector
      // so we assume that input data is on the same GPU
      // note : cudaPointerGetAttributes has a strange behaviour:
      // it looks like it makes a copy of the vector on the default GPU device (0) !!!
      // So we prefer to avoid this and provide directly the device_id as input (else statement below)
      cudaPointerAttributes attributes;
      CudaSafeCall(cudaPointerGetAttributes(&attributes, out));
      CudaSafeCall(cudaSetDevice(attributes.device));
    } else // device_id is provided, so we use it. Warning : is has to be consistent with location of data
      CudaSafeCall(cudaSetDevice(device_id));

    return Eval_(fun, nx, ny, out, pargs);

  }

};

}
