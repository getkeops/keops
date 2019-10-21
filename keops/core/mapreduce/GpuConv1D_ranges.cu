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


struct GpuConv1D_ranges_FromDevice {
template < typename TYPE, class FUN >
static int Eval_(FUN fun, int nx, int ny,
    int nbatchdims, int *shapes,  
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
    blockSize.x = ::std::min(CUDA_BLOCK_SIZE,::std::min(deviceProp.maxThreadsPerBlock, (int) (deviceProp.sharedMemPerBlock / ::std::max(1, (int)(DIMY*sizeof(TYPE))) ))); // number of threads in each block


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

    // Size of the SharedData : blockSize.x*(DIMY)*sizeof(TYPE)
    GpuConv1DOnDevice_ranges<TYPE><<<gridSize,blockSize,blockSize.x*(DIMY)*sizeof(TYPE)>>>(fun,nx,ny,
        nbatchdims,shapes, offsets_d,
        lookup_d,slices_x_d,ranges_y_d,
        px_d,py_d,pp_d);

    // block until the device has completed
    CudaSafeCall(cudaDeviceSynchronize());
    CudaCheckError();

    CudaSafeCall(cudaFree(p_data));

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

    return Eval_(fun,nx,ny,nbatchdims,shapes,nranges_x,nranges_y,ranges,phx_d,phy_d,php_d);

}

// same without the device_id argument
template < typename TYPE, class FUN, typename... Args >
static int Eval(FUN fun, int nx, int ny, 
    int nbatchdims, int *shapes, 
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
    return Eval(fun, nx, ny, nbatchdims, shapes, nranges_x,nranges_y,ranges, attributes.device, x1_d, args...);
}

template < typename TYPE, class FUN >
static int Eval(FUN fun, int nx, int ny, 
    int nbatchdims, int *shapes, 
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

    return Eval_(fun,nx,ny,nbatchdims,shapes,nranges_x,nranges_y,ranges,px_d,py_d,pp_d);

}

};

}
