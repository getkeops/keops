#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <cuda.h>

#include "core/pack/Pack.h"
#include "core/pack/Load.h"
#include "core/pack/Load_Chunks.h"
#include "core/pack/Load_Indref.h"
#include "core/pack/Call.h"
#include "core/pack/GetInds.h"
#include "core/pack/GetDims.h"
#include "core/mapreduce/broadcast_batch_dimensions.h"
#include "core/utils/CudaErrorCheck.cu"
#include "core/utils/TypesUtils.h"

#include "Chunk_Mode_Constants.h"

namespace keops {


template < int USE_CHUNK_MODE, int BLOCKSIZE_CHUNKS, class FUN_GLOBAL=void, class VARFINAL=void > struct GpuConv1DOnDevice_ranges {};





#if USE_FINAL_CHUNKS==1


template < class FUN_GLOBAL, class VARFINAL, int DIMFINALCHUNK_CURR, typename TYPE >
__device__ void do_finalchunk_sub_ranges(TYPE *acc, int i, int j, int jstart, int start_y, int chunk, int end_x, int end_y, 
            int nbatchdims, int *indices_j,
			TYPE **args, TYPE *fout, TYPE *yj, TYPE *out) {
                
            static const int DIMOUT = VARFINAL::DIM;
                
            VectAssign<DIMFINALCHUNK>(acc,0.0f); //typename FUN::template InitializeReduction<__TYPEACC__, TYPE >()(acc); // acc = 0
            TYPE *yjrel = yj;
            if (j < end_y) // we load yj from device global memory only if j<end_y
                if (nbatchdims==0)
                    load_chunks < pack<VARFINAL::N>, DIMFINALCHUNK, DIMFINALCHUNK_CURR, VARFINAL::DIM >
                                    (j, chunk, yj + threadIdx.x * DIMFINALCHUNK, args);
                else
                    load_chunks < pack<VARFINAL::N>, typename FUN_GLOBAL::INDSJ, DIMFINALCHUNK, DIMFINALCHUNK_CURR, VARFINAL::DIM >
                                    (j-start_y, chunk, yj + threadIdx.x * DIMFINALCHUNK, args, indices_j);
            __syncthreads();
            for (int jrel = 0; (jrel < blockDim.x) && (jrel < end_y - jstart); jrel++, yjrel += DIMFINALCHUNK) {          
                if (i < end_x) { // we compute only if needed
                    #pragma unroll
                    for (int k=0; k<DIMFINALCHUNK_CURR; k++)                         
                        acc[k] += yjrel[k] * fout[jrel];
                }
                __syncthreads();
            }
            if (i < end_x) {
                //typename FUN::template FinalizeOutput<__TYPEACC__,TYPE>()(acc, out + i * DIMOUT, i);
                #pragma unroll
                for (int k=0; k<DIMFINALCHUNK_CURR; k++)
                    out[i*DIMOUT+chunk*DIMFINALCHUNK+k] += acc[k];
            }
            __syncthreads();
}

template < int BLOCKSIZE_CHUNKS, class VARFINAL, typename TYPE, class FUN_GLOBAL, class FUN >
__global__ void GpuConv1DOnDevice_ranges_FinalChunks(FUN_GLOBAL fun_global, FUN fun, int nx, int ny, 
    int nbatchdims, int *shapes, int *offsets_d,
    __INDEX__* lookup_d, __INDEX__* slices_x, __INDEX__* ranges_y,
    TYPE *out, TYPE **args) {
    
    // Buffers for the "broadcasted indices" -----------------------------------
    typedef typename FUN_GLOBAL::VARSI VARSI_GLOBAL;
    typedef typename FUN_GLOBAL::VARSJ VARSJ_GLOBAL;
    typedef typename FUN_GLOBAL::VARSP VARSP_GLOBAL;

    const int SIZEI_GLOBAL = VARSI_GLOBAL::SIZE;
    const int SIZEJ_GLOBAL = VARSJ_GLOBAL::SIZE;
    const int SIZEP_GLOBAL = VARSP_GLOBAL::SIZE;

    const int SIZEVARS_GLOBAL = SIZEI_GLOBAL + SIZEJ_GLOBAL + SIZEP_GLOBAL;

    int offsets[SIZEVARS_GLOBAL];
    int *indices_i = offsets, *indices_j = offsets + SIZEI_GLOBAL, *indices_p = offsets + SIZEI_GLOBAL + SIZEJ_GLOBAL;

    if (nbatchdims > 0) {
        for (int k = 0; k < SIZEVARS_GLOBAL; k++) {
            offsets[k] = offsets_d[ SIZEVARS_GLOBAL * blockIdx.x + k ];
        }
    }
    
    // Retrieve our position along the laaaaarge [1,~nx] axis: -----------------
    __INDEX__ range_id= (lookup_d)[3*blockIdx.x] ;
    __INDEX__ start_x = (lookup_d)[3*blockIdx.x+1] ;
    __INDEX__ end_x   = (lookup_d)[3*blockIdx.x+2] ;
    
    // The "slices_x" vector encodes a set of cutting points in
    // the "ranges_y" array of ranges.
    // As discussed in the Genred docstring, the first "0" is implicit:
    __INDEX__ start_slice = range_id < 1 ? 0 : slices_x[range_id-1];
    __INDEX__ end_slice   = slices_x[range_id];
    
    // get the index of the current thread
    int i = start_x + threadIdx.x;

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
    if (nbatchdims == 0)
        load<DIMSP,INDSP>(0, param_loc, args); // load parameters variables from global memory to local thread memory
    else
        load<DIMSP,INDSP>(0, param_loc, args, indices_p); // Possibly, with offsets as we support broadcasting over batch dimensions
        
    TYPE fout[DIMFOUT*BLOCKSIZE_CHUNKS];
    
    // get the value of variable (index with i)
    TYPE xi[DIMX < 1 ? 1 : DIMX];
    if (i < end_x) {
        if (nbatchdims == 0)
            load<DIMSX, INDSI>(i, xi, args); // load xi variables from global memory to local thread memory
        else
            load< DIMSX, INDSI>(threadIdx.x, xi, args, indices_i);  // Possibly, with offsets as we support broadcasting over batch dimensions
        #pragma unroll
        for (int k=0; k<DIMOUT; k++) {
            out[i*DIMOUT+k] = 0.0f;
        }
    }
    
    __TYPEACC__ acc[DIMFINALCHUNK];
    
    __INDEX__ start_y = ranges_y[2*start_slice], end_y = 0;
    for( __INDEX__ index = start_slice ; index < end_slice ; index++ ) {
        if( (index+1 >= end_slice) || (ranges_y[2*index+2] != ranges_y[2*index+1]) ) {
            end_y = ranges_y[2*index+1];
            for (int jstart = start_y, tile = 0; jstart < end_y; jstart += blockDim.x, tile++) {

                // get the current column
                int j = jstart + threadIdx.x;

                if (j < end_y) { // we load yj from device global memory only if j<end_y
                    if (nbatchdims == 0)
                        load<DIMSY,INDSJ>(j, yj + threadIdx.x * DIMY, args); // load yj variables from global memory to shared memory
                    else
                        load<DIMSY,INDSJ>(j-start_y, yj+threadIdx.x*DIMY, args, indices_j);  // Possibly, with offsets as we support broadcasting over batch dimensions
                }
                __syncthreads();

                if (i < end_x) { // we compute x1i only if needed
                    TYPE * yjrel = yj; // Loop on the columns of the current block.
                    for (int jrel = 0; (jrel < BLOCKSIZE_CHUNKS) && (jrel < end_y - jstart); jrel++, yjrel += DIMY)
                        call<DIMSX, DIMSY, DIMSP>(fun,
                                  fout+jrel*DIMFOUT,
                                  xi,
                                  yjrel,
                                  param_loc); // Call the function, which outputs results in fout
                }
                __syncthreads();
                
                for (int chunk=0; chunk<NCHUNKS-1; chunk++) 
                    do_finalchunk_sub_ranges < FUN_GLOBAL, VARFINAL, DIMFINALCHUNK > (acc, i, j, jstart, start_y, chunk, end_x, end_y, nbatchdims, indices_j, args, fout, yj, out);
                do_finalchunk_sub_ranges < FUN_GLOBAL, VARFINAL, DIMLASTFINALCHUNK > (acc, i, j, jstart, start_y, NCHUNKS-1, end_x, end_y, nbatchdims, indices_j, args, fout, yj, out);
                    
            }
            if(index+1 < end_slice) 
                start_y = ranges_y[2*index+2] ;
        }
    }
}


template < int BLOCKSIZE_CHUNKS, class FUN_GLOBAL, class VARFINAL > 
struct GpuConv1DOnDevice_ranges<2,BLOCKSIZE_CHUNKS,FUN_GLOBAL,VARFINAL> {
    template < typename TYPE, class FUN >
    static void Eval(dim3 gridSize, dim3 blockSize, size_t SharedMem, FUN fun, int nx, int ny, int nbatchdims, int *shapes, int *offsets_d,
    __INDEX__* lookup_d, __INDEX__* slices_x, __INDEX__* ranges_y, TYPE *out, TYPE **args) {
        GpuConv1DOnDevice_ranges_FinalChunks < BLOCKSIZE_CHUNKS, VARFINAL > <<< gridSize, blockSize, SharedMem >>> (FUN_GLOBAL(), fun, nx, ny, nbatchdims, shapes, offsets_d,
    		lookup_d, slices_x, ranges_y, out, args);
    }
};

#endif






template < class FUN, class FUN_CHUNKED_CURR, int DIMCHUNK_CURR, typename TYPE >
__device__ void do_chunk_sub_ranges(TYPE *acc, int tile, int i, int j, int jstart, int start_y, int chunk, int end_x, int end_y, 
			int nbatchdims, int *indices_i, int *indices_j, TYPE **args, TYPE *fout, TYPE *xi, TYPE *yj, TYPE *param_loc) {

	using CHK = Chunk_Mode_Constants<FUN>;

	TYPE fout_tmp_chunk[CHK::FUN_CHUNKED::DIM];
	
	if (i < end_x) {
            if (nbatchdims == 0) {
				load_chunks < typename CHK::INDSI_CHUNKED, DIMCHUNK, DIMCHUNK_CURR, CHK::DIM_ORG >
					(i, chunk, xi + CHK::DIMX_NOTCHUNKED, args);
            } else {
                load_chunks < typename CHK::INDSI_CHUNKED, typename FUN::INDSI, DIMCHUNK, DIMCHUNK_CURR, CHK::DIM_ORG >
					(threadIdx.x, chunk, xi + CHK::DIMX_NOTCHUNKED, args, indices_i);
            }
        }

	if (j < end_y) { // we load yj from device global memory only if j<ny
            if (nbatchdims == 0) {
				load_chunks < typename CHK::INDSJ_CHUNKED, DIMCHUNK, DIMCHUNK_CURR, CHK::DIM_ORG > 
						(j, chunk, yj + threadIdx.x * CHK::DIMY + CHK::DIMY_NOTCHUNKED, args);
            } else {
                load_chunks < typename CHK::INDSJ_CHUNKED, typename FUN::INDSJ, DIMCHUNK, DIMCHUNK_CURR, CHK::DIM_ORG > 
						(j-start_y, chunk, yj + threadIdx.x * CHK::DIMY + CHK::DIMY_NOTCHUNKED, args, indices_j);
            }
        }

	__syncthreads();

	if (i < end_x) { // we compute only if needed
		TYPE *yjrel = yj; // Loop on the columns of the current block.
        for (int jrel = 0; (jrel < blockDim.x) && (jrel < end_y - jstart); jrel++, yjrel += CHK::DIMY) {
				TYPE *foutj = fout+jrel*CHK::FUN_CHUNKED::DIM;
			call < CHK::DIMSX, CHK::DIMSY, CHK::DIMSP > 
				(FUN_CHUNKED_CURR::template EvalFun<CHK::INDS>(), fout_tmp_chunk, xi, yjrel, param_loc);
			CHK::FUN_CHUNKED::acc_chunk(foutj, fout_tmp_chunk);
		}
	}
	__syncthreads();
}





template < int BLOCKSIZE_CHUNKS, typename TYPE, class FUN >
__global__ void GpuConv1DOnDevice_ranges_Chunks(FUN fun, int nx, int ny,
    int nbatchdims, int *shapes, int *offsets_d,
    __INDEX__* lookup_d, __INDEX__* slices_x, __INDEX__* ranges_y,
    TYPE *out, TYPE **args) {

    using CHK = Chunk_Mode_Constants<FUN>;

    // Buffers for the "broadcasted indices" -----------------------------------
    typedef typename FUN::VARSI VARSI;
    typedef typename FUN::VARSJ VARSJ;
    typedef typename FUN::VARSP VARSP;

    const int SIZEI = VARSI::SIZE;
    const int SIZEJ = VARSJ::SIZE;
    const int SIZEP = VARSP::SIZE;

    const int SIZEVARS = SIZEI + SIZEJ + SIZEP;

    int offsets[SIZEVARS];
    int *indices_i = offsets, *indices_j = offsets + SIZEI, *indices_p = offsets + SIZEI + SIZEJ;

    if (nbatchdims > 0) {
        for (int k = 0; k < SIZEVARS; k++) {
            offsets[k] = offsets_d[ SIZEVARS * blockIdx.x + k ];
        }
    }


    // Retrieve our position along the laaaaarge [1,~nx] axis: -----------------
    __INDEX__ range_id= (lookup_d)[3*blockIdx.x] ;
    __INDEX__ start_x = (lookup_d)[3*blockIdx.x+1] ;
    __INDEX__ end_x   = (lookup_d)[3*blockIdx.x+2] ;
    
    // The "slices_x" vector encodes a set of cutting points in
    // the "ranges_y" array of ranges.
    // As discussed in the Genred docstring, the first "0" is implicit:
    __INDEX__ start_slice = range_id < 1 ? 0 : slices_x[range_id-1];
    __INDEX__ end_slice   = slices_x[range_id];

    // get the index of the current thread
    int i = start_x + threadIdx.x;

    // declare shared mem
    extern __shared__ TYPE yj[];

    TYPE param_loc[CHK::DIMP < 1 ? 1 : CHK::DIMP];
    if (nbatchdims == 0) {
        load<CHK::DIMSP,CHK::INDSP>(0, param_loc, args); // load parameters variables from global memory to local thread memory
    } else {
        load<CHK::DIMSP,CHK::INDSP>(0, param_loc, args, indices_p); // Possibly, with offsets as we support broadcasting over batch dimensions
    }




	__TYPEACC__ acc[CHK::DIMRED];

#if SUM_SCHEME == BLOCK_SUM
    // additional tmp vector to store intermediate results from each block
    TYPE tmp[CHK::DIMRED];
#elif SUM_SCHEME == KAHAN_SCHEME
    // additional tmp vector to accumulate errors
    static const int DIM_KAHAN = FUN::template KahanScheme<__TYPEACC__,TYPE>::DIMACC;
    TYPE tmp[DIM_KAHAN];
#endif

    if (i < end_x) {
	typename FUN::template InitializeReduction<__TYPEACC__, TYPE >()(acc); // acc = 0
#if SUM_SCHEME == KAHAN_SCHEME
	VectAssign<DIM_KAHAN>(tmp,0.0f);
#endif		
    }

    TYPE xi[CHK::DIMX];

    TYPE fout_chunk[BLOCKSIZE_CHUNKS*CHK::DIMOUT_CHUNK];
	
    if (i < end_x) {
        if (nbatchdims == 0) {
            load < CHK::DIMSX_NOTCHUNKED, CHK::INDSI_NOTCHUNKED > 
				(i, xi, args); // load xi variables from global memory to local thread memory
        } else {
            load_indref < CHK::DIMSX_NOTCHUNKED, CHK::INDSI_NOTCHUNKED, FUN::INDSI > 
				(threadIdx.x, xi, args, indices_i); // load xi variables from global memory to local thread memory
        }
    }

    __INDEX__ start_y = ranges_y[2*start_slice], end_y = 0;
    for( __INDEX__ index = start_slice ; index < end_slice ; index++ ) {
        if( (index+1 >= end_slice) || (ranges_y[2*index+2] != ranges_y[2*index+1]) ) {
            //start_y = ranges_y[2*index] ;
            end_y = ranges_y[2*index+1];

			for (int jstart = start_y, tile = 0; jstart < end_y; jstart += blockDim.x, tile++) {

				// get the current column
				int j = jstart + threadIdx.x;
	
				if (j < end_y) {
                    if (nbatchdims == 0) {
						load < CHK::DIMSY_NOTCHUNKED, CHK::INDSJ_NOTCHUNKED >
							(j, yj + threadIdx.x * CHK::DIMY, args);
                    } else {
                        load_indref < CHK::DIMSY_NOTCHUNKED, CHK::INDSJ_NOTCHUNKED, FUN::INDSJ >
							(j-start_y, yj + threadIdx.x * CHK::DIMY, args, indices_j);
                    }
				}
				__syncthreads();

				if (i < end_x) { // we compute only if needed
					for (int jrel = 0; (jrel < blockDim.x) && (jrel < end_y - jstart); jrel++)
						CHK::FUN_CHUNKED::initacc_chunk(fout_chunk+jrel*CHK::DIMOUT_CHUNK);
#if SUM_SCHEME == BLOCK_SUM
					typename FUN::template InitializeReduction<TYPE,TYPE>()(tmp); // tmp = 0
#endif
				}

				// looping on chunks (except the last)
				#pragma unroll
				for (int chunk=0; chunk<CHK::NCHUNKS-1; chunk++)
					do_chunk_sub_ranges < FUN, CHK::FUN_CHUNKED, DIMCHUNK >
						(acc, tile, i, j, jstart, start_y, chunk, end_x, end_y, nbatchdims, indices_i, indices_j, args, fout_chunk, xi, yj, param_loc);	
				// last chunk
				do_chunk_sub_ranges < FUN, CHK::FUN_LASTCHUNKED, CHK::DIMLASTCHUNK >
					(acc, tile, i, j, jstart, start_y, CHK::NCHUNKS-1, end_x,end_y, nbatchdims, indices_i, indices_j, args, fout_chunk, xi, yj, param_loc);

			if (i < end_x) { 
				TYPE *yjrel = yj; // Loop on the columns of the current block.
                if (nbatchdims == 0) {
					for (int jrel = 0; (jrel < blockDim.x) && (jrel <end_y - jstart); jrel++, yjrel += CHK::DIMY) {
#if SUM_SCHEME != KAHAN_SCHEME
						int ind = jrel + tile * blockDim.x + start_y;
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
				} 
                else {
					for (int jrel = 0; (jrel < blockDim.x) && (jrel <end_y - jstart); jrel++, yjrel += CHK::DIMY) {
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
				}
#if SUM_SCHEME == BLOCK_SUM
				typename FUN::template ReducePair<__TYPEACC__,TYPE>()(acc, tmp);     // acc += tmp
#endif
			}
	__syncthreads();
	}
	
			if(index+1 < end_slice) {
                start_y = ranges_y[2*index+2] ;
			}
		}
	}

    if (i < end_x) 
		typename FUN::template FinalizeOutput<__TYPEACC__,TYPE>()(acc, out + i * CHK::DIMOUT, i);

}


template < int BLOCKSIZE_CHUNKS > 
struct GpuConv1DOnDevice_ranges<1,BLOCKSIZE_CHUNKS> {
    template < typename TYPE, class FUN >
    static void Eval(dim3 gridSize, dim3 blockSize, size_t SharedMem, FUN fun, int nx, int ny, int nbatchdims, int *shapes, int *offsets_d,
    __INDEX__* lookup_d, __INDEX__* slices_x, __INDEX__* ranges_y,
    TYPE *out, TYPE **args) {
        GpuConv1DOnDevice_ranges_Chunks < BLOCKSIZE_CHUNKS > <<< gridSize, blockSize, SharedMem >>> (fun, nx, ny, nbatchdims, shapes, offsets_d,
    		lookup_d, slices_x, ranges_y, out, args);
    }
};




template < typename TYPE, class FUN >
__global__ void GpuConv1DOnDevice_ranges_NoChunks(FUN fun, int nx, int ny,
    int nbatchdims, int *shapes, int *offsets_d,
    __INDEX__* lookup_d, __INDEX__* slices_x, __INDEX__* ranges_y,
    TYPE *out, TYPE **args) {


    // Buffers for the "broadcasted indices" -----------------------------------
    typedef typename FUN::VARSI VARSI;
    typedef typename FUN::VARSJ VARSJ;
    typedef typename FUN::VARSP VARSP;

    const int SIZEI = VARSI::SIZE;
    const int SIZEJ = VARSJ::SIZE;
    const int SIZEP = VARSP::SIZE;

    const int SIZEVARS = SIZEI + SIZEJ + SIZEP;

    int offsets[SIZEVARS];
    int *indices_i = offsets, *indices_j = offsets + SIZEI, *indices_p = offsets + SIZEI + SIZEJ;

    if (nbatchdims > 0) {
        for (int k = 0; k < SIZEVARS; k++) {
            offsets[k] = offsets_d[ SIZEVARS * blockIdx.x + k ];
        }
    }


    // Retrieve our position along the laaaaarge [1,~nx] axis: -----------------
    __INDEX__ range_id= (lookup_d)[3*blockIdx.x] ;
    __INDEX__ start_x = (lookup_d)[3*blockIdx.x+1] ;
    __INDEX__ end_x   = (lookup_d)[3*blockIdx.x+2] ;
    
    // The "slices_x" vector encodes a set of cutting points in
    // the "ranges_y" array of ranges.
    // As discussed in the Genred docstring, the first "0" is implicit:
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

    if (nbatchdims == 0) {
	    load<DIMSP, INDSP>(0, param_loc, args); // load parameters variables from global memory to local thread memory
    } else {
        load<DIMSP,INDSP>(0, param_loc, args, indices_p); // Possibly, with offsets as we support broadcasting over batch dimensions
    }

    TYPE fout[DIMFOUT < 1 ? 1 : DIMFOUT];
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
    if(i<end_x) {
        typename FUN::template InitializeReduction<__TYPEACC__,TYPE>()(acc); // acc = 0
#if SUM_SCHEME == KAHAN_SCHEME
        VectAssign<DIM_KAHAN>(tmp,0.0f);
#endif
        if (nbatchdims == 0) {
            load< DIMSX, INDSI>(i, xi, args); // load xi variables from global memory to local thread memory
        } else {
            load< DIMSX, INDSI>(threadIdx.x, xi, args, indices_i);  // Possibly, with offsets as we support broadcasting over batch dimensions
        }
    }

    __INDEX__ start_y = ranges_y[2*start_slice], end_y = 0;
    for( __INDEX__ index = start_slice ; index < end_slice ; index++ ) {
        if( (index+1 >= end_slice) || (ranges_y[2*index+2] != ranges_y[2*index+1]) ) {
            //start_y = ranges_y[2*index] ;
            end_y = ranges_y[2*index+1];

            for(int jstart = start_y, tile = 0; jstart < end_y; jstart += blockDim.x, tile++) {

                // get the current column
                int j = jstart + threadIdx.x;

                if(j<end_y) { // we load yj from device global memory only if j<end_y
                    if (nbatchdims == 0) {
                        load<DIMSY,INDSJ>(j, yj+threadIdx.x*DIMY, args); // load yj variables from global memory to shared memory
                    } else {
                        load<DIMSY,INDSJ>(j-start_y, yj+threadIdx.x*DIMY, args, indices_j);  // Possibly, with offsets as we support broadcasting over batch dimensions
                    }
                }
                __syncthreads();

                if(i<end_x) { // we compute x1i only if needed
                    TYPE* yjrel = yj; // Loop on the columns of the current block.
#if SUM_SCHEME == BLOCK_SUM
      	            typename FUN::template InitializeReduction<TYPE,TYPE>()(tmp); // tmp = 0
#endif
                    if (nbatchdims == 0) {
                        for(int jrel = 0; (jrel < blockDim.x) && (jrel<end_y-jstart); jrel++, yjrel+=DIMY) {
                            call<DIMSX,DIMSY,DIMSP>(fun,fout,xi,yjrel,param_loc); // Call the function, which outputs results in xi[0:DIMX1]
#if SUM_SCHEME == BLOCK_SUM
#if USE_HALF
        int ind = jrel+tile*blockDim.x + start_y;
        typename FUN::template ReducePairShort<TYPE,TYPE>()(tmp, fout, __floats2half2_rn(2*ind,2*ind+1));     // tmp += fout
#else
        typename FUN::template ReducePairShort<TYPE,TYPE>()(tmp, fout, jrel+tile*blockDim.x + start_y);     // tmp += fout
#endif                           
#elif SUM_SCHEME == KAHAN_SCHEME
                            typename FUN::template KahanScheme<__TYPEACC__,TYPE>()(acc, fout, tmp);
#else
#if USE_HALF
        int ind = jrel+tile*blockDim.x + start_y;
        typename FUN::template ReducePairShort<__TYPEACC__,TYPE>()(acc, fout, __floats2half2_rn(2*ind,2*ind+1));     // acc += fout
#else
        typename FUN::template ReducePairShort<__TYPEACC__,TYPE>()(acc, fout, jrel+tile*blockDim.x + start_y);     // acc += fout
#endif                              
#endif
                        } 
                    }
                    else {
                        for(int jrel = 0; (jrel < blockDim.x) && (jrel<end_y-jstart); jrel++, yjrel+=DIMY) {
                            call<DIMSX,DIMSY,DIMSP>(fun,fout,xi,yjrel,param_loc); // Call the function, which outputs results in fout
#if SUM_SCHEME == BLOCK_SUM
#if USE_HALF
       			    int ind = jrel+tile*blockDim.x;
        		    typename FUN::template ReducePairShort<TYPE,TYPE>()(tmp, fout, __floats2half2_rn(2*ind,2*ind+1));     // tmp += fout
#else
                            typename FUN::template ReducePairShort<TYPE,TYPE>()(tmp, fout, jrel+tile*blockDim.x);     // tmp += fout
#endif
#elif SUM_SCHEME == KAHAN_SCHEME
                            typename FUN::template KahanScheme<__TYPEACC__,TYPE>()(acc, fout, tmp);
#else
#if USE_HALF
       			    int ind = jrel+tile*blockDim.x;
        		    typename FUN::template ReducePairShort<__TYPEACC__,TYPE>()(acc, fout, __floats2half2_rn(2*ind,2*ind+1));     // acc += fout
#else
                            typename FUN::template ReducePairShort<__TYPEACC__,TYPE>()(acc, fout, jrel+tile*blockDim.x);     // acc += fout
#endif
#endif
                        }
                    }
#if SUM_SCHEME == BLOCK_SUM
                    typename FUN::template ReducePair<__TYPEACC__,TYPE>()(acc, tmp);     // acc += tmp
#endif
                }
                __syncthreads();
            }

            if(index+1 < end_slice) {
                start_y = ranges_y[2*index+2] ;
            }
        }
    }
    if(i<end_x) {
    	typename FUN::template FinalizeOutput<__TYPEACC__,TYPE>()(acc, out+i*DIMOUT, i);
    }

}


template < int DUMMY > 
struct GpuConv1DOnDevice_ranges<0,DUMMY> {
    template < typename TYPE, class FUN >
    static void Eval(dim3 gridSize, dim3 blockSize, size_t SharedMem, FUN fun, int nx, int ny, int nbatchdims, int *shapes, int *offsets_d,
    __INDEX__* lookup_d, __INDEX__* slices_x, __INDEX__* ranges_y,
    TYPE *out, TYPE **args) {
        GpuConv1DOnDevice_ranges_NoChunks <<< gridSize, blockSize, SharedMem >>> (fun, nx, ny, nbatchdims, shapes, offsets_d,
    		lookup_d, slices_x, ranges_y, out, args);
    }
};





template < class FUN >
int* build_offset_tables( int nbatchdims, int *shapes, int nblocks, __INDEX__ *lookup_h ) {

        // Support for broadcasting over batch dimensions =============================================
        typedef typename FUN::VARSI VARSI;
        typedef typename FUN::VARSJ VARSJ;
        typedef typename FUN::VARSP VARSP;
    
        const int SIZEI = VARSI::SIZE;
        const int SIZEJ = VARSJ::SIZE;
        const int SIZEP = VARSP::SIZE;
    
        const int SIZEVARS = SIZEI + SIZEJ + SIZEP;
    
        // Separate and store the shapes of the "i" and "j" variables + parameters --------------
        //
        // shapes is an array of size (1+nargs)*(nbatchdims+3), which looks like:
        // [ A, .., B, M, N, D_out]  -> output
        // [ A, .., B, M, 1, D_1  ]  -> "i" variable
        // [ A, .., B, 1, N, D_2  ]  -> "j" variable
        // [ A, .., B, 1, 1, D_3  ]  -> "parameter"
        // [ A, .., 1, M, 1, D_4  ]  -> N.B.: we support broadcasting on the batch dimensions!
        // [ 1, .., 1, M, 1, D_5  ]  ->      (we'll just ask users to fill in the shapes with *explicit* ones)
    
        int shapes_i[SIZEI*(nbatchdims+1)], shapes_j[SIZEJ*(nbatchdims+1)], shapes_p[SIZEP*(nbatchdims+1)];
    
        // First, we fill shapes_i with the "relevant" shapes of the "i" variables,
        // making it look like, say:
        // [ A, .., B, M]
        // [ A, .., 1, M]
        // [ A, .., A, M]
        // Then, we do the same for shapes_j, but with "N" instead of "M".
        // And finally for the parameters, with "1" instead of "M".
        fill_shapes<FUN>(nbatchdims, shapes, shapes_i, shapes_j, shapes_p);
    
        const int tagIJ = FUN::tagJ; // 1 if the reduction is made "over j", 0 if it is made "over i"
        int M = shapes[nbatchdims], N = shapes[nbatchdims+1];

        // We create a lookup table, "offsets", of shape (nblocks, SIZEVARS) --------
        int *offsets_h = NULL, *offsets_d = NULL;
    
        offsets_h = new int[nblocks * SIZEVARS] ;

        for (int k=0; k < nblocks; k++) {
            int range_id = (int) lookup_h[3*k] ;
            int start_x  = tagIJ ? range_id * M : range_id * N;
            int start_y  = tagIJ ? range_id * N : range_id * M;

            int patch_offset = (int) (lookup_h[3*k+1]-start_x);
            
            vect_broadcast_index(start_x, nbatchdims, SIZEI, shapes, shapes_i, offsets_h + k*SIZEVARS, patch_offset);
            vect_broadcast_index(start_y, nbatchdims, SIZEJ,   shapes, shapes_j, offsets_h + k*SIZEVARS + SIZEI);
            vect_broadcast_index(range_id, nbatchdims, SIZEP, shapes, shapes_p, offsets_h + k*SIZEVARS + SIZEI + SIZEJ);
        }

        CudaSafeCall(cudaMalloc((int**)&offsets_d, sizeof(int)*nblocks*SIZEVARS));
        CudaSafeCall(cudaMemcpy(offsets_d, offsets_h, sizeof(int)*nblocks*SIZEVARS, cudaMemcpyHostToDevice));
    
        delete [] offsets_h;
        return offsets_d;
}








struct GpuConv1D_ranges_FromHost {

template < typename TYPE, class FUN >
static int Eval_(FUN fun, int nx, int ny, 
    int nbatchdims, int *shapes, 
    int nranges_x, int nranges_y, int nredranges_x, int nredranges_y, __INDEX__ **ranges, 
    TYPE *out, TYPE **args_h) {

    typedef typename FUN::DIMSX DIMSX;
    typedef typename FUN::DIMSY DIMSY;
    typedef typename FUN::DIMSP DIMSP;
    typedef typename FUN::INDSI INDSI;
    typedef typename FUN::INDSJ INDSJ;
    typedef typename FUN::INDSP INDSP;
    const int DIMOUT = FUN::DIM; // dimension of output variable
    const int SIZEI = DIMSX::SIZE;
    const int SIZEJ = DIMSY::SIZE;
    const int SIZEP = DIMSP::SIZE;
    static const int NMINARGS = FUN::NMINARGS;



    // Compute the memory footprints of all (broadcasted?) variables ===========
    typedef typename FUN::VARSI VARSI;
    typedef typename FUN::VARSJ VARSJ;
    typedef typename FUN::VARSP VARSP;

    // Separate and store the shapes of the "i" and "j" variables + parameters --------------
    //
    // shapes is an array of size (1+nargs)*(nbatchdims+3), which looks like:
    // [ A, .., B, M, N, D_out]  -> output
    // [ A, .., B, M, 1, D_1  ]  -> "i" variable
    // [ A, .., B, 1, N, D_2  ]  -> "j" variable
    // [ A, .., B, 1, 1, D_3  ]  -> "parameter"
    // [ A, .., 1, M, 1, D_4  ]  -> N.B.: we support broadcasting on the batch dimensions!
    // [ 1, .., 1, M, 1, D_5  ]  ->      (we'll just ask users to fill in the shapes with *explicit* ones)

    int shapes_i[(SIZEI)*(nbatchdims+1)], shapes_j[SIZEJ*(nbatchdims+1)], shapes_p[SIZEP*(nbatchdims+1)];

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
    for (int k=0; k < SIZEI; k++) { // For the actual variables:
        tmp = DIMSX::VAL(k);  // use the product of the vector dimension...
        for (int l=0; l < nbatchdims+1; l++) {
            tmp *= shapes_i[ k*(nbatchdims+1) + l];  // with all the shape's dimensions
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

    // pointer to device output array
    TYPE *out_d;

    // array of pointers to device input arrays
    TYPE **args_d;

    // single cudaMalloc
    void *p_data;
    CudaSafeCall(cudaMalloc(&p_data, 
                             sizeof(TYPE*) * NMINARGS      // pointers to the start of each variable
                           + sizeof(TYPE) * ( nx*DIMOUT				  // output
											+ total_footprint_p       // parameters
                                            + total_footprint_x       // "i" variables if tagIJ==1, "j" otherwise
                                            + total_footprint_y )));  // "j" variables if tagIJ==1, "i" otherwise
    

    // Now, fill in our big, contiguous array: ---------------------------------
    // In the head, the pointer to the data:
    args_d = (TYPE **) p_data;

    // In the tail, the actual data:
    TYPE *dataloc = (TYPE *) (args_d + NMINARGS);  // Beware: Instead of storing TYPE*, we now store TYPE
    out_d = dataloc;
    dataloc += nx*DIMOUT;

    // host array of pointers to device data
    TYPE *ph[NMINARGS];

      for (int k = 0; k < SIZEP; k++) {
        int indk = INDSP::VAL(k);
        int nvals = footprints_p[k];        
        CudaSafeCall(cudaMemcpy(dataloc, args_h[indk], sizeof(TYPE) * nvals, cudaMemcpyHostToDevice));
        ph[indk] = dataloc;
        dataloc += nvals;
      }

   for (int k = 0; k < SIZEI; k++) {
      int indk = INDSI::VAL(k);
      int nvals = footprints_x[k];
      CudaSafeCall(cudaMemcpy(dataloc, args_h[indk], sizeof(TYPE) * nvals, cudaMemcpyHostToDevice));
      ph[indk] = dataloc;
      dataloc += nvals;
    }

      for (int k = 0; k < SIZEJ; k++) {
        int indk = INDSJ::VAL(k);
        int nvals = footprints_y[k];
        CudaSafeCall(cudaMemcpy(dataloc, args_h[indk], sizeof(TYPE) * nvals, cudaMemcpyHostToDevice));
        ph[indk] = dataloc;
        dataloc += nvals;
      }

    // Load on the device the pointer arrays: ----------------------------------
    CudaSafeCall(cudaMemcpy(args_d, ph, NMINARGS * sizeof(TYPE *), cudaMemcpyHostToDevice));

    // Setup the compute properties ==============================================================
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


    #if USE_FINAL_CHUNKS==1
        GpuConv1DOnDevice_ranges<USE_CHUNK_MODE,BLOCKSIZE_CHUNKS,FUN,VARFINAL>::Eval(gridSize, blockSize, blockSize.x * DIMY_SHARED * sizeof(TYPE), 
									FUN_INTERNAL(), nx, ny, nbatchdims,shapes, offsets_d,
								        lookup_d,slices_x_d,ranges_y_d,
								        out_d, args_d);
    #else
        GpuConv1DOnDevice_ranges<USE_CHUNK_MODE,BLOCKSIZE_CHUNKS>::Eval(gridSize, blockSize, blockSize.x * DIMY_SHARED * sizeof(TYPE), 
									fun, nx, ny, nbatchdims,shapes, offsets_d,
								        lookup_d,slices_x_d,ranges_y_d,
								        out_d, args_d);
    #endif
    
    // block until the device has completed
    CudaSafeCall(cudaDeviceSynchronize());
    CudaCheckError();

    // Send data from device to host.
    CudaSafeCall(cudaMemcpy(out, out_d, sizeof(TYPE)*(nx*DIMOUT),cudaMemcpyDeviceToHost));

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
    int device_id, TYPE *out, Args... args) {

    if(device_id!=-1)
        CudaSafeCall(cudaSetDevice(device_id));
    
    static const int Nargs = sizeof...(Args);
    TYPE *pargs[Nargs];
    unpack(pargs, args...);

    return Eval_(fun,nx,ny,nbatchdims,shapes,nranges_x,nranges_y,nredranges_x,nredranges_y,ranges,out, pargs);

}

// same without the device_id argument
template < typename TYPE, class FUN, typename... Args >
static int Eval(FUN fun, int nx, int ny, 
    int nbatchdims, int *shapes, 
    int nranges_x, int nranges_y, int nredranges_x, int nredranges_y, __INDEX__ **ranges, 
    TYPE *out, Args... args) {
    return Eval(fun, nx, ny, nbatchdims, shapes, nranges_x, nranges_y, nredranges_x, nredranges_y, ranges, -1, out, args...);
}

// Idem, but with args given as an array of arrays, instead of an explicit list of arrays
template < typename TYPE, class FUN >
static int Eval(FUN fun, int nx, int ny, 
    int nbatchdims, int *shapes, 
    int nranges_x, int nranges_y, int nredranges_x, int nredranges_y, __INDEX__ **ranges, 
    TYPE* out, TYPE** pargs, int device_id=-1) {

    // We set the GPU device on which computations will be performed
    if(device_id!=-1)
        CudaSafeCall(cudaSetDevice(device_id));

    return Eval_(fun,nx,ny,nbatchdims,shapes,nranges_x,nranges_y,nredranges_x,nredranges_y,ranges,out,pargs);

}


};


struct GpuConv1D_ranges_FromDevice {
template < typename TYPE, class FUN >
static int Eval_(FUN fun, int nx, int ny,
    int nbatchdims, int *shapes,  
    int nranges_x, int nranges_y, __INDEX__ **ranges, 
    TYPE *out, TYPE** args) {

    static const int NMINARGS = FUN::NMINARGS;

    // device array of pointers to device data
    TYPE **args_d;

    // single cudaMalloc
    CudaSafeCall(cudaMalloc(&args_d, sizeof(TYPE*)*NMINARGS));

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

    __INDEX__ *ranges_x_h = NULL, *slices_x_d = NULL, *ranges_y_d = NULL;

    // The code below needs a pointer to ranges_x on *host* memory,  -------------------
    // as well as pointers to slices_x and ranges_y on *device* memory.
    // -> Depending on the "ranges" location, we'll copy ranges_x *or* slices_x and ranges_y
    //    to the appropriate memory:
    bool ranges_on_device = (nbatchdims==0);  
    // N.B.: We only support Host ranges with Device data when these ranges were created 
    //       to emulate block-sparse reductions.

    if ( ranges_on_device ) {  // The ranges are on the device
        ranges_x_h = new __INDEX__[2*nranges] ;
        // Send data from device to host.
        CudaSafeCall(cudaMemcpy(ranges_x_h, ranges_x, sizeof(__INDEX__)*2*nranges, cudaMemcpyDeviceToHost));
        slices_x_d = slices_x;
        ranges_y_d = ranges_y;
    }
    else {  // The ranges are on host memory; this is typically what happens with **batch processing**,
            // with ranges generated by keops_io.h:
        ranges_x_h = ranges_x;

        // Copy "slices_x" to the device:
        CudaSafeCall(cudaMalloc((__INDEX__**)&slices_x_d, sizeof(__INDEX__)*nranges));
        CudaSafeCall(cudaMemcpy(slices_x_d, slices_x, sizeof(__INDEX__)*nranges, cudaMemcpyHostToDevice));

        // Copy "redranges_y" to the device: with batch processing, we KNOW that they have the same shape as ranges_x
        CudaSafeCall(cudaMalloc((__INDEX__**)&ranges_y_d, sizeof(__INDEX__)*2*nranges));
        CudaSafeCall(cudaMemcpy(ranges_y_d, ranges_y, sizeof(__INDEX__)*2*nranges, cudaMemcpyHostToDevice));
    }

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
            lookup_h[3*index+2] = ranges_x_h[2*i] + j + ::std::min((int) blockSize.x, len_range-j ) ;

            index++;
        }
    }

    // Load the table on the device -----------------------------------------------------
    __INDEX__ *lookup_d = NULL;
    CudaSafeCall(cudaMalloc((__INDEX__**)&lookup_d, sizeof(__INDEX__)*3*nblocks));
    CudaSafeCall(cudaMemcpy(lookup_d, lookup_h, sizeof(__INDEX__)*3*nblocks, cudaMemcpyHostToDevice));

    // Support for broadcasting over batch dimensions =============================================

    // We create a lookup table, "offsets", of shape (nblock, SIZEVARS):
    int *offsets_d = NULL;

    if (nbatchdims > 0) {
        offsets_d = build_offset_tables<FUN>( nbatchdims, shapes, nblocks, lookup_h );
    }

    // ============================================================================================

    dim3 gridSize;
    gridSize.x =  nblocks ; //nx / blockSize.x + (nx%blockSize.x==0 ? 0 : 1);

    #if USE_FINAL_CHUNKS==1
        GpuConv1DOnDevice_ranges<USE_CHUNK_MODE,BLOCKSIZE_CHUNKS,FUN,VARFINAL>::Eval(gridSize, blockSize, blockSize.x * DIMY_SHARED * sizeof(TYPE), 
									FUN_INTERNAL(), nx, ny, nbatchdims,shapes, offsets_d,
								        lookup_d,slices_x_d,ranges_y_d,
								        out, args_d);
    #else
        GpuConv1DOnDevice_ranges<USE_CHUNK_MODE,BLOCKSIZE_CHUNKS>::Eval(gridSize, blockSize, blockSize.x * DIMY_SHARED * sizeof(TYPE), 
									fun, nx, ny, nbatchdims,shapes, offsets_d,
								        lookup_d,slices_x_d,ranges_y_d,
								        out, args_d);
    #endif
    
    // block until the device has completed
    CudaSafeCall(cudaDeviceSynchronize());
    CudaCheckError();

    CudaSafeCall(cudaFree(args_d));

    // Free the block lookup table :
    delete [] lookup_h;
    CudaSafeCall(cudaFree(lookup_d));

    // Free the host or device "ranges" copies:
    if (ranges_on_device) {
        delete [] ranges_x_h;
    } else {
        CudaSafeCall(cudaFree(slices_x_d));
        CudaSafeCall(cudaFree(ranges_y_d));
    }
    
    if (nbatchdims > 0) {
        CudaSafeCall(cudaFree(offsets_d));
    }


    return 0;
}

// Same wrappers, but for data located on the device
template < typename TYPE, class FUN, typename... Args >
static int Eval(FUN fun, int nx, int ny, 
    int nbatchdims, int *shapes, 
    int nranges_x, int nranges_y, __INDEX__ **ranges, 
    int device_id, TYPE* out, Args... args) {

    // device_id is provided, so we set the GPU device accordingly
    // Warning : is has to be consistent with location of data
    CudaSafeCall(cudaSetDevice(device_id));

    static const int Nargs = sizeof...(Args);
    TYPE *pargs[Nargs];
    unpack(pargs, args...);
    return Eval_(fun,nx,ny,nbatchdims,shapes,nranges_x,nranges_y,ranges,out, pargs);

}

// same without the device_id argument
template < typename TYPE, class FUN, typename... Args >
static int Eval(FUN fun, int nx, int ny, 
    int nbatchdims, int *shapes, 
    int nranges_x, int nranges_y, __INDEX__ **ranges, 
    TYPE* out, Args... args) {
    // We set the GPU device on which computations will be performed
    // to be the GPU on which data is located.
    // NB. we only check location of out which is the output vector
    // so we assume that input data is on the same GPU
    // note : cudaPointerGetAttributes has a strange behaviour:
    // it looks like it makes a copy of the vector on the default GPU device (0) !!! 
    // So we prefer to avoid this and provide directly the device_id as input (first function above)
    cudaPointerAttributes attributes;
    CudaSafeCall(cudaPointerGetAttributes(&attributes,out));
    return Eval(fun, nx, ny, nbatchdims, shapes, nranges_x,nranges_y,ranges, attributes.device, out, args...);
}

template < typename TYPE, class FUN >
static int Eval(FUN fun, int nx, int ny, 
    int nbatchdims, int *shapes, 
    int nranges_x, int nranges_y, __INDEX__ **ranges, 
    TYPE* out, TYPE** args, int device_id=-1) {

    if(device_id==-1) {
        // We set the GPU device on which computations will be performed
        // to be the GPU on which data is located.
        // NB. we only check location of x1_d which is the output vector
        // so we assume that input data is on the same GPU
        // note : cudaPointerGetAttributes has a strange behaviour:
        // it looks like it makes a copy of the vector on the default GPU device (0) !!! 
	// So we prefer to avoid this and provide directly the device_id as input (else statement below)
        cudaPointerAttributes attributes;
        CudaSafeCall(cudaPointerGetAttributes(&attributes,out));
        CudaSafeCall(cudaSetDevice(attributes.device));
    } else // device_id is provided, so we use it. Warning : is has to be consistent with location of data
        CudaSafeCall(cudaSetDevice(device_id));

    return Eval_(fun,nx,ny,nbatchdims,shapes,nranges_x,nranges_y,ranges,out,args);

}

};

}
