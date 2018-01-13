
#ifndef REDUCTION_SUM
#define REDUCTION_SUM


// Sum is the generic template, with arbitrary formula F.
// These functions can be overloaded if, for instance,
//                 F = LogSumExp<G>.

template <typename TYPE, int DIM, class F>
struct InitializeOutput{
HOST_DEVICE INLINE void operator()(TYPE *tmp) {
    for(int k=0; k<DIM; k++)
        tmp[k] = 0.0f; // initialize output
}
};

// equivalent of the += operation
template <typename TYPE, int DIM, class F>
struct ReducePair{
HOST_DEVICE INLINE void operator()(TYPE *tmp, TYPE *xi) {
    for(int k=0; k<DIM; k++) { 
        tmp[k] += xi[k];
    }
}
};

template <typename TYPE, int DIMVECT, class F>
__global__ void reduce2D(TYPE* in, TYPE* out, int sizeY,int nx) {
    /* Function used as a final reduction pass in the 2D scheme,
     * once the block reductions have been made.
     * Takes as input:
     * - in,  a  sizeY * (nx * DIMVECT ) array
     * - out, an          nx * DIMVECT   array
     *
     * Computes, in parallel, the "columnwise"-sum (which correspond to lines of blocks)
     * of *in and stores the result in out.
     */
    TYPE res = 0;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < nx*DIMVECT) {
        for (int i = 0; i < sizeY; i++)
            res += in[tid + i*nx*DIMVECT]; // We use "+=" as a reduction op. But it could be anything, really!
        /*res = in[tid+ nx* DIMVECT];*/
        out[tid] = res;
    }
}

#endif