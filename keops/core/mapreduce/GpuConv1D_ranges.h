#pragma once

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <cuda.h>

#include "core/pack/Pack.h"
#include "core/pack/GetInds.h"
#include "core/pack/GetDims.h"
#include "core/mapreduce/broadcast_batch_dimensions.h"
#include "core/utils/CudaErrorCheck.cu"

namespace keops {

template < typename TYPE, class FUN >
__global__ void GpuConv1DOnDevice_ranges(FUN fun, int nx, int ny,
    int nbatchdims, int *shapes, int *offsets_d,
    __INDEX__* lookup_d, __INDEX__* slices_x, __INDEX__* ranges_y,
    TYPE** px, TYPE** py, TYPE** pp) {


    // Buffers for the "broadcasted indices" -----------------------------------
    typedef typename FUN::VARSI VARSI;
    typedef typename FUN::VARSJ VARSJ;
    typedef typename FUN::VARSP VARSP;

    const int SIZEI = VARSI::SIZE+1;  // The usual convention is that the output "counts" in SIZEI
    const int SIZEJ = VARSJ::SIZE;
    const int SIZEP = VARSP::SIZE;

    const int SIZEVARS = SIZEI-1 + SIZEJ + SIZEP;

    int offsets[SIZEVARS];
    int *indices_i = offsets, *indices_j = offsets + SIZEI-1, *indices_p = offsets + SIZEI-1 + SIZEJ;

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
    const int DIMX = DIMSX::SUM;        // DIMX  is sum of dimensions for xi variables
    const int DIMY = DIMSY::SUM;        // DIMY  is sum of dimensions for yj variables
    const int DIMP = DIMSP::SUM;        // DIMP  is sum of dimensions for parameters variables
    const int DIMOUT = FUN::DIM; // dimension of output variable
    const int DIMRED = FUN::DIMRED; // dimension of reduction operation
    const int DIMFOUT = DIMSX::FIRST;     // DIMFOUT is dimension of output variable of inner function

    // load parameter(s)
    TYPE param_loc[DIMP < 1 ? 1 : DIMP];

    if (nbatchdims == 0) {
	    load<DIMSP>(0, param_loc, pp); // load parameters variables from global memory to local thread memory
    } else {
        load<DIMSP>(0, param_loc, pp, indices_p); // Possibly, with offsets as we support broadcasting over batch dimensions
    }

    // get the value of variable (index with i)
    TYPE xi[DIMX < 1 ? 1 : DIMX];
    __TYPEACC__ acc[DIMRED];
#if USE_BLOCKRED
    // additional tmp vector to store intermediate results from each block
    TYPE tmp[DIMRED];
#elif USE_KAHAN
    // additional tmp vector to accumulate errors
    const int DIM_KAHAN = FUN::template KahanScheme<__TYPEACC__,TYPE>::DIMACC;
    TYPE tmp[DIM_KAHAN];
#endif
    if(i<end_x) {
        typename FUN::template InitializeReduction<__TYPEACC__>()(acc); // acc = 0
#if USE_KAHAN
#pragma unroll
        for (int k = 0; k < DIM_KAHAN; k++)
          tmp[k] = 0.0f;
#endif
        if (nbatchdims == 0) {
            load<typename DIMSX::NEXT>(i, xi+DIMFOUT, px+1); // load xi variables from global memory to local thread memory
        } else {
            load<typename DIMSX::NEXT>(threadIdx.x, xi+DIMFOUT, px+1, indices_i);  // Possibly, with offsets as we support broadcasting over batch dimensions
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
                        load<DIMSY>(j, yj+threadIdx.x*DIMY, py); // load yj variables from global memory to shared memory
                    } else {
                        load<DIMSY>(j-start_y, yj+threadIdx.x*DIMY, py, indices_j);  // Possibly, with offsets as we support broadcasting over batch dimensions
                    }
                }
                __syncthreads();

                if(i<end_x) { // we compute x1i only if needed
                    TYPE* yjrel = yj; // Loop on the columns of the current block.
#if USE_BLOCKRED
      	            typename FUN::template InitializeReduction<TYPE>()(tmp); // tmp = 0
#endif
                    if (nbatchdims == 0) {
                        for(int jrel = 0; (jrel < blockDim.x) && (jrel<end_y-jstart); jrel++, yjrel+=DIMY) {
                            call<DIMSX,DIMSY,DIMSP>(fun,xi,yjrel,param_loc); // Call the function, which accumulates results in xi[0:DIMX1]
#if USE_BLOCKRED
                            typename FUN::template ReducePairShort<TYPE,TYPE>()(tmp, xi, jrel+tile*blockDim.x + start_y);     // tmp += xi
#elif USE_KAHAN
                            typename FUN::template KahanScheme<__TYPEACC__,TYPE>()(acc, xi, tmp);
#else
                            typename FUN::template ReducePairShort<__TYPEACC__,TYPE>()(acc, xi, jrel+tile*blockDim.x + start_y);     // acc += xi
#endif
                        } 
                    }
                    else {
                        for(int jrel = 0; (jrel < blockDim.x) && (jrel<end_y-jstart); jrel++, yjrel+=DIMY) {
                            call<DIMSX,DIMSY,DIMSP>(fun,xi,yjrel,param_loc); // Call the function, which accumulates results in xi[0:DIMX1]
#if USE_BLOCKRED
                            typename FUN::template ReducePairShort<TYPE,TYPE>()(tmp, xi, jrel+tile*blockDim.x);     // tmp += xi
#elif USE_KAHAN
                            typename FUN::template KahanScheme<__TYPEACC__,TYPE>()(acc, xi, tmp);
#else
                            typename FUN::template ReducePairShort<__TYPEACC__,TYPE>()(acc, xi, jrel+tile*blockDim.x);     // acc += xi
#endif
                        }
                    }
#if USE_BLOCKRED
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
    	typename FUN::template FinalizeOutput<__TYPEACC__,TYPE>()(acc, px[0]+i*DIMOUT, px, i);
//printf("blockIdx.x=%d, threadIdx.x=%d, i=%d, start_x=%d, end_x=%d, *acc=%f, *(px[0]+i*DIMOUT)=%f\n",blockIdx.x,threadIdx.x,i,start_x,end_x,*acc,*(px[0]+i*DIMOUT));
    }

}




template < class FUN >
int* build_offset_tables( int nbatchdims, int *shapes, int nblocks, __INDEX__ *lookup_h ) {

        // Support for broadcasting over batch dimensions =============================================
        typedef typename FUN::VARSI VARSI;
        typedef typename FUN::VARSJ VARSJ;
        typedef typename FUN::VARSP VARSP;
    
        const int SIZEI = VARSI::SIZE+1;  // The usual convention is that the output "counts" in SIZEI
        const int SIZEJ = VARSJ::SIZE;
        const int SIZEP = VARSP::SIZE;
    
        const int SIZEVARS = SIZEI-1 + SIZEJ + SIZEP;
    
        // Separate and store the shapes of the "i" and "j" variables + parameters --------------
        //
        // shapes is an array of size (1+nargs)*(nbatchdims+3), which looks like:
        // [ A, .., B, M, N, D_out]  -> output
        // [ A, .., B, M, 1, D_1  ]  -> "i" variable
        // [ A, .., B, 1, N, D_2  ]  -> "j" variable
        // [ A, .., B, 1, 1, D_3  ]  -> "parameter"
        // [ A, .., 1, M, 1, D_4  ]  -> N.B.: we support broadcasting on the batch dimensions!
        // [ 1, .., 1, M, 1, D_5  ]  ->      (we'll just ask users to fill in the shapes with *explicit* ones)
    
        int shapes_i[(SIZEI-1)*(nbatchdims+1)], shapes_j[SIZEJ*(nbatchdims+1)], shapes_p[SIZEP*(nbatchdims+1)];
    
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
            
            vect_broadcast_index(start_x, nbatchdims, SIZEI-1, shapes, shapes_i, offsets_h + k*SIZEVARS, patch_offset);
            vect_broadcast_index(start_y, nbatchdims, SIZEJ,   shapes, shapes_j, offsets_h + k*SIZEVARS + SIZEI-1);
            vect_broadcast_index(range_id, nbatchdims, SIZEP, shapes, shapes_p, offsets_h + k*SIZEVARS + SIZEI-1 + SIZEJ);
        }

        CudaSafeCall(cudaMalloc((int**)&offsets_d, sizeof(int)*nblocks*SIZEVARS));
        CudaSafeCall(cudaMemcpy(offsets_d, offsets_h, sizeof(int)*nblocks*SIZEVARS, cudaMemcpyHostToDevice));
    
        delete [] offsets_h;
        return offsets_d;
}

}
