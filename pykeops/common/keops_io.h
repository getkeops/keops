#include <vector>
#include <string>
#include <type_traits>
#include <limits>
#include <stdexcept>
#include <tuple>

// #include "formula.h" done by cmake

extern "C" {
    int CpuReduc(int, int, __TYPE__*, __TYPE__**);
    int CpuReduc_ranges(int, int, int, int*, int, int, __INDEX__**, __TYPE__*, __TYPE__**);
};

#if USE_CUDA
extern "C" {
    int GpuReduc1D_FromHost(int, int, __TYPE__*, __TYPE__**, int);
    int GpuReduc1D_FromDevice(int, int, __TYPE__*, __TYPE__**, int);
    int GpuReduc2D_FromHost(int, int, __TYPE__*, __TYPE__**, int);
    int GpuReduc2D_FromDevice(int, int, __TYPE__*, __TYPE__**, int);
    int GpuReduc1D_ranges_FromHost(int, int, int, int*, int, int, int, int, __INDEX__**, __TYPE__*, __TYPE__**, int);
    int GpuReduc1D_ranges_FromDevice(int, int, int, int*, int, int, __INDEX__**, __TYPE__*, __TYPE__**, int);
};
#endif

namespace pykeops {

using namespace keops;
namespace py = pybind11;

/////////////////////////////////////////////////////////////////////////////////
//                           Keops
/////////////////////////////////////////////////////////////////////////////////


using FF = F::F; // F::F is formula inside reduction (ex if F is SumReduction<Form> then F::F is Form)

using VARSI = typename FF::template VARS<0>;    // list variables of type I used in formula F
using VARSJ = typename FF::template VARS<1>;    // list variables of type J used in formula F
using VARSP = typename FF::template VARS<2>;    // list variables of type parameter used in formula F

using DIMSX = GetDims<VARSI>;
using DIMSY = GetDims<VARSJ>;
using DIMSP = GetDims<VARSP>;

using INDSI = GetInds<VARSI>;
using INDSJ = GetInds<VARSJ>;
using INDSP = GetInds<VARSP>;

using INDS = ConcatPacks<ConcatPacks<INDSI, INDSJ>, INDSP>;

const int NARGSI = VARSI::SIZE; // number of I variables used in formula F
const int NARGSJ = VARSJ::SIZE; // number of J variables used in formula F
const int NARGSP = VARSP::SIZE; // number of parameters variables used in formula F

const int NARGS = F::NMINARGS;
const int DIMOUT = F::DIM;

const int TAGIJ = F::tagI;

const std::string f =  PrintReduction<F>();

const auto Error_msg_no_cuda = "[KeOps] This KeOps shared object has been compiled without cuda support: \n 1) to perform computations on CPU, simply set tagHostDevice to 0\n 2) to perform computations on GPU, please recompile the formula with a working version of cuda.";

/////////////////////////////////////////////////////////////////////////////////
//                             Utils
/////////////////////////////////////////////////////////////////////////////////

template< typename array_t >
int get_ndim(array_t obj_ptri);  // len( a.shape )

template< typename array_t >
int get_size(array_t obj_ptri, int l);  // a.shape[l]

template< typename array_t >
__TYPE__* get_data(array_t obj_ptri);   // raw pointer to "a.data"

template< typename array_t >
__INDEX__* get_rangedata(array_t obj_ptri);  // raw pointer to "a.data", casted as integer

template< typename array_t >
bool is_contiguous(array_t obj_ptri);  // is "a" ordered properly? KeOps does *not* support strides!


/////////////////////////////////////////////////////////////////////////////////
//                    Sanity checks on args
/////////////////////////////////////////////////////////////////////////////////


void check_tag(int tag, std::string msg){
    if ((tag < 0) || (tag > 1)) {
        throw std::runtime_error("[KeOps] tag" + msg + " should be (0 or 1) but is " + std::to_string(tag));
    }
}

template<typename array_t>
std::tuple<int, int, int, int*> check_args(size_t nargs, std::vector<int> categories, std::vector<int> dimensions, std::vector<array_t> obj_ptr) {

    if (nargs < NARGS) {  // given vs. expected number of arguments
        throw std::runtime_error("[KeOps] Not enough arguments: received " + std::to_string(nargs)
                                +" but expected at least " + std::to_string(NARGS) + ".");
    }

    int *typeargs, *dimargs;

    if (false) { //(NARGS>0) {  
        // Jean: This dimcheck may fail with second derivatives, and is somewhat useless
        //       if Genred(...) is implemented correctly... So I removed it. May be a bad idea!

        // Expected categories and dimensions, from the formula's signature =============
        
        typeargs = new int[NARGS];  // Expected categories
        dimargs = new int[NARGS];   // Expected dimenstions

        // Fill typeargs and dimargs with -1's:
        for (int k = 0; k < NARGS; k++) {  
            typeargs[k] = -1;
            dimargs[k] = -1;
        }
        // Fill in all the positions that correspond to "i" variables:
        for (int k = 0; k < NARGSI; k++) {
            typeargs[INDSI::VAL(k)] = 0;
            dimargs[INDSI::VAL(k)] = DIMSX::VAL(k);
        }
        // Fill in all the positions that correspond to "j" variables:
        for (int k = 0; k < NARGSJ; k++) {
            typeargs[INDSJ::VAL(k)] = 1;
            dimargs[INDSJ::VAL(k)] = DIMSY::VAL(k);
        }
        // Fill in all the positions that correspond to "parameters":
        for (int k = 0; k < NARGSP; k++) {
            typeargs[INDSP::VAL(k)] = 2;
            dimargs[INDSP::VAL(k)] = DIMSP::VAL(k);
        }

        // Check vs. the user-given categories and dimensions ==================
        for (int k = 0; k < NARGS; k++) {
            if (typeargs[k] != categories[k]) {
                throw std::runtime_error("[KeOps] Wrong variable category (0 = Vi, 1 = Vj, 2 = Pm) at position " + std::to_string(k)
                                        +": received " + std::to_string(categories[k]) +
                                        +" but expected " + std::to_string(typeargs[k]) + ".");
            }

            if (dimargs[k] != dimensions[k]) {
                throw std::runtime_error("[KeOps] Wrong dimension for variable at position " + std::to_string(k)
                                        +": received " + std::to_string(dimargs[k]) +
                                        +" but expected " + std::to_string(dimensions[k]) + ".");
            }
        }
    }

    // Are we working in batch mode? Infer the answer from the first arg =============
    int nbatchdims = get_ndim( obj_ptr[0] );  // Number of dims of the first tensor
    // Remove the "trailing" dim (.., D) if the first arg is a parameter,
    // or the last two (.., M/N, D) if it is an "i" or "j" variable:
    nbatchdims -= (categories[0] == 2) ? 1 : 2;  

    if (nbatchdims < 0) {
        throw std::runtime_error("[KeOps] Wrong number of dimensions for the first arg: is "
                + std::to_string( get_ndim( obj_ptr[0] ) ) + " but should be at least "
                + std::to_string( (categories[0] == 2) ? 1 : 2 ) );
    }

    // Now, we'll keep track of the output + all arguments' shapes in a large array:
    int *shapes = new int[(nargs + 1) * (nbatchdims + 3)]; // N.B.: shapes will be destroyed at the very end of generic_red
    // Eventually, with "batch dimensions" [A, .., B], the table will look like:
    //
    // [ A, .., B, M, N, D_out]  -> output
    // [ A, .., B, M, 1, D_1  ]  -> "i" variable
    // [ A, .., B, 1, N, D_2  ]  -> "j" variable
    // [ A, .., B, 1, 1, D_3  ]  -> "parameter"
    // [ A, .., 1, M, 1, D_4  ]  -> N.B.: we support broadcasting on the batch dimensions!
    // [ 1, .., 1, M, 1, D_5  ]  ->      (we'll just ask users to fill in the shapes with *explicit* ones)

    // Fill in the first line with
    // [ 1, ..., 1, -1, -1, D_out]
    for (int b = 0; b < nbatchdims; b++) {
        shapes[b] = 1;  // 1 = default option
    }
    shapes[nbatchdims]     = -1;  // M is still unknown
    shapes[nbatchdims + 1] = -1;  // N is still unknown
    shapes[nbatchdims + 2] = DIMOUT;  // Top right corner: dimension of the output


    // Check the compatibility of all tensor shapes ==================================
    
    for (size_t i = 0; i < nargs; i++) {

        // Check the number of dimensions --------------------------------------------
        int ndims = get_ndim( obj_ptr[i] );  // Number of dims of the i-th tensor
        
        // N.B.: CAT=2 -> "Parameter" -> 1 extra dim ; otherwise, CAT=0 or 1 -> 2 extra dims
        if( ndims != nbatchdims + ((categories[i] == 2) ? 1 : 2) ) {
            throw std::runtime_error("[KeOps] Wrong number of dimensions for arg number " + std::to_string(i) 
                        + " : KeOps detected " + std::to_string(nbatchdims)
                        + " batch dimensions from the first argument 0, and thus expected "
                        + std::to_string( nbatchdims + ((categories[i] == 2) ? 1 : 2) )
                        + " dimenstions here, but only received "
                        + std::to_string( ndims ) 
                        + ". Note that KeOps supports broadcasting on batch dimensions, "
                        + "but still expects 'dummy' unit dimensions in the input shapes, "
                        + "for the sake of clarity.");
        }  

        // Fill in the (i+1)-th line of the "shapes" array ---------------------------
        int off_i = (i + 1) * (nbatchdims + 3);

        // First, the batch dimensions:
        for (int b = 0; b < nbatchdims; b++) {
            shapes[off_i+b] = get_size(obj_ptr[i], b);

            // Check that the current value is compatible with what
            // we've encountered so far, as stored in the first line of "shapes"
            if (shapes[off_i+b] != 1) {  // This dimension is not "broadcasted"
                if (shapes[b] == 1) {
                    shapes[b] = shapes[off_i+b];  // -> it becomes the new standard
                } else if ( shapes[b] != shapes[off_i+b] ) {
                    throw std::runtime_error("[KeOps] Wrong value of the batch dimension " 
                        + std::to_string(b) + " for argument number " + std::to_string(i) 
                        + " : is " + std::to_string(shapes[off_i+b]) 
                        + " but was " + std::to_string(shapes[b]) 
                        + " or 1 in previous arguments.");
                }
            }
        }

        // Then, the numbers "M", "N" and "D":
        if (categories[i] == 0) {  // "i" variable --------------------------------------
            shapes[off_i+nbatchdims] = get_size(obj_ptr[i], nbatchdims);  // = "M"
            if (shapes[nbatchdims] == -1) {  // This is the first "i" variable that we encounter
                shapes[nbatchdims] = shapes[off_i+nbatchdims];  // -> Fill in the "M" coefficient in the first line
            }

            shapes[off_i+nbatchdims+1] = 1;
            shapes[off_i+nbatchdims+2] = get_size(obj_ptr[i], nbatchdims+1);  // = "D"

            // Check the number of "lines":
            if (shapes[nbatchdims] != shapes[off_i+nbatchdims]) {
                throw std::runtime_error("[KeOps] Wrong value of the 'i' dimension "
                        + std::to_string(nbatchdims) + "for arg number " + std::to_string(i) 
                        + " : is " + std::to_string(shapes[off_i+nbatchdims])
                        + " but was " + std::to_string(shapes[nbatchdims]) 
                        + " in previous 'i' arguments.");
            }

            // And the number of "columns":
            if (shapes[off_i+nbatchdims+2] != dimensions[i]) {
                throw std::runtime_error("[KeOps] Wrong value of the 'vector size' dimension "
                        + std::to_string(nbatchdims+1) + "for arg number " + std::to_string(i) 
                        + " : is " + std::to_string(shapes[off_i+nbatchdims+2]) 
                        + " but should be " + std::to_string(dimensions[i])) ;
            }
        } 
        else if (categories[i] == 1) {  // "j" variable ----------------------------------
            shapes[off_i+nbatchdims] = 1;
            shapes[off_i+nbatchdims+1] = get_size(obj_ptr[i], nbatchdims);  // = "N"
            if (shapes[nbatchdims+1] == -1) {  // This is the first "j" variable that we encounter
                shapes[nbatchdims+1] = shapes[off_i+nbatchdims+1];  // -> Fill in the "N" coefficient in the first line
            }

            shapes[off_i+nbatchdims+2] = get_size(obj_ptr[i], nbatchdims+1);  // = "D"

            // Check the number of "lines":
            if (shapes[nbatchdims+1] != shapes[off_i+nbatchdims+1]) {
                throw std::runtime_error("[KeOps] Wrong value of the 'j' dimension "
                        + std::to_string(nbatchdims) + "for arg number " + std::to_string(i) 
                        + " : is " + std::to_string(shapes[off_i+nbatchdims+1])
                        + " but was " + std::to_string(shapes[nbatchdims+1]) 
                        + " in previous 'j' arguments.");
            }

            // And the number of "columns":
            if (shapes[off_i+nbatchdims+2] != dimensions[i]) {
                throw std::runtime_error("[KeOps] Wrong value of the 'vector size' dimension "
                        + std::to_string(nbatchdims+1) + "for arg number " + std::to_string(i) 
                        + " : is " + std::to_string(shapes[off_i+nbatchdims+2]) 
                        + " but should be " + std::to_string(dimensions[i])) ;
            }

        } 
        else if (categories[i] == 2) {  // "parameters" -------------------------------
            shapes[off_i+nbatchdims]   = 1;
            shapes[off_i+nbatchdims+1] = 1;
            shapes[off_i+nbatchdims+2] = get_size(obj_ptr[i], nbatchdims);  // = "D"

            if (shapes[off_i+nbatchdims+2] != dimensions[i]) {
                throw std::runtime_error("[KeOps] Wrong value of the 'vector size' dimension "
                        + std::to_string(nbatchdims) + "for arg number " + std::to_string(i) 
                        + " : is " + std::to_string(shapes[off_i+nbatchdims+2]) 
                        + " but should be " + std::to_string(dimensions[i])) ;
            }
        }

        if (!is_contiguous(obj_ptr[i])) {
            throw std::runtime_error("[KeOps] Arg number " + std::to_string(i) + " : is not contiguous. "
                    + "Please provide 'contiguous' dara array, as KeOps does not support strides. "
                    + "If you're getting this error in the 'backward' pass of a code using torch.sum() "
                    + "on the output of a KeOps routine, you should consider replacing 'a.sum()' with "
                    + "'(1. * a).sum()' or 'torch.dot(a.view(-1), torch.ones_like(a).view(-1))'. ") ;
        }
    }

    // Compute the total numbers nx and ny of "i" and "j" indices ==========
    // Remember that the first line of "shapes" is given by:
    //
    // [ A, .., B, M, N, D_out]  -> output

    
    int nx = shapes[nbatchdims];      // = M
    int ny = shapes[nbatchdims + 1];  // = N

    if (nx == -1 or ny == -1) {
        throw std::runtime_error("[KeOps] [Jean:] This formula only referred to one type ('i' or 'j') of variable: we should find a way to handle this situation properly.");
    }

    int nbatches = 1;
    for (int b = 0; b < nbatchdims; b++) {
        nbatches *= shapes[b];  // Compute the product of all "batch dimensions"
    }
    nx *= nbatches;  // = A * ... * B * M
    ny *= nbatches;  // = A * ... * B * N
    
    // Free the allocated memory (but *not* shapes) ========================
    if (false) { //(NARGS>0) {
        delete[] dimargs;
        delete[] typeargs;
    }

    return std::make_tuple(nx, ny, nbatchdims, shapes);

}

template<typename _T>
short int cast_Device_Id(_T Device_Id) {
    static_assert(std::is_integral<_T>::value, "Device_Id must be of integral type.");
    if(Device_Id < std::numeric_limits<short int>::max()) {
        return(static_cast<short int>(Device_Id));
    } else {
        throw std::runtime_error("[KeOps] Device_Id exceeded short int limit");
    }
}

/////////////////////////////////////////////////////////////////////////////////
//                    Call Cuda functions
/////////////////////////////////////////////////////////////////////////////////

// Implemented by pykeops/torch/generic_red.cpp or pykeops/numpy/generic_red.cpp
template < typename array_t >
array_t launch_keops(int tag1D2D, int tagCpuGpu, int tagHostDevice, short int Device_Id,
                        int nx, int ny, int nbatchdims, int *shapes, int *shape_out,
                        int tagRanges, int nranges_x, int nranges_y, int nredranges_x, int nredranges_y, __INDEX__ **castedranges,
                        __TYPE__ ** castedargs);


/////////////////////////////////////////////////////////////////////////////////
//                    Main function
/////////////////////////////////////////////////////////////////////////////////

template < typename array_t, typename index_t >
array_t generic_red(int tagCpuGpu,        // tagCpuGpu=0     means Reduction on Cpu, tagCpuGpu=1       means Reduction on Gpu, tagCpuGpu=2 means Reduction on Gpu from device data
                    int tag1D2D,          // tag1D2D=0       means 1D Gpu scheme,      tag1D2D=1       means 2D Gpu scheme
                    int tagHostDevice,    // tagHostDevice=1 means _fromDevice suffix. tagHostDevice=0 means _fromHost suffix
                    int Device_Id,        // id of GPU device
                    py::tuple ranges,     // () if no "sparsity" ranges are given (default behavior)
                                          // Otherwise, ranges is a 6-uple of (integer) array_t
                                          // ranges = (ranges_i, slices_i, redranges_j, ranges_j, slices_j, redranges_i)
                                          // as documented in the doc on sparstiy and clustering.
                    py::tuple categories,
                    py::tuple dimensions,
                    py::args py_args) {

    // Check that we have enough arguments:
    size_t nargs = py_args.size();
    if (nargs < NARGS) {
        throw std::runtime_error(
        "[KeOps] Wrong number of args : is " + std::to_string(py_args.size())
        + " but should be at least " + std::to_string(NARGS)
        + " in " + f
        );
    }

    check_tag(tag1D2D, "1D2D");
    check_tag(tagCpuGpu, "CpuGpu");
    check_tag(tagHostDevice, "HostDevice");

    short int Device_Id_s = cast_Device_Id(Device_Id);

    // Cast the input variable : It may be a copy here...
    std::vector<array_t> obj_ptr(py_args.size());
    for (size_t i = 0; i < py_args.size(); i++)
        obj_ptr[i] = py::cast<array_t> (py_args[i]);
    // If torch.h is included, the last 3 lines could be replaced by : auto obj_ptr = py::cast<std::vector<array_t>>(py_args);

    // get the pointers to data to avoid a copy
    __TYPE__ **castedargs = new __TYPE__ *[NARGS];
    for(size_t i=0; i<NARGS; i++)
        castedargs[i] = get_data(obj_ptr[i]);

    // Cast the input signature:
    std::vector<int> cats(categories.size());
    for (size_t i = 0; i < categories.size(); i++)
        cats[i] = py::cast<int> (categories[i]);

    std::vector<int> dims(dimensions.size());
    for (size_t i = 0; i < dimensions.size(); i++)
        dims[i] = py::cast<int> (dimensions[i]);

    if ((categories.size() != nargs) or (dimensions.size() != nargs)) {
        throw std::runtime_error(
        "[KeOps] The lengths of the 'categories', 'dimensions' and '*args' tuples mismatch."
        );
    }

    // Check the aguments' dimensions, and retrieve all the shape information:
    std::tuple<int, int, int, int*> nx_ny_nbatch_shapes = check_args<array_t>(nargs, cats, dims, obj_ptr);
    int nx = std::get<0>(nx_ny_nbatch_shapes), ny = std::get<1>(nx_ny_nbatch_shapes);
    int nbatchdims = std::get<2>(nx_ny_nbatch_shapes);
    int *shapes = std::get<3>(nx_ny_nbatch_shapes);

    int tagRanges, nranges_x, nranges_y, nredranges_x, nredranges_y ;
    __INDEX__ **castedranges;
    // N.B.: This vector is only used if ranges.size() == 6,
    //       but should *absolutely* be declared in all cases.
    //       Otherwise, a silent error (not detected by the compiler) provokes
    //       a silent buffer re-allocation, with a random corruption (in some edge cases)
    //       of the "ranges" arrays and, eventually, a segmentation fault. 
    std::vector<index_t> ranges_ptr(ranges.size());  

    // Sparsity: should we handle ranges? ======================================

    if (nbatchdims == 0) {  // Standard M-by-N computation
        if(ranges.size() == 0) {
            tagRanges = 0; 
            nranges_x = 0; nranges_y = 0 ;
            nredranges_x = 0; nredranges_y = 0 ;
            castedranges = new __INDEX__ *[1];
        }
        else if(ranges.size() == 6) {
            // Cast the six integer arrays
            for (size_t i = 0; i < ranges.size(); i++)
                ranges_ptr[i] = py::cast<index_t> (ranges[i]);
            
            // get the pointers to data to avoid a copy
            castedranges = new __INDEX__ *[ranges.size()];
            for(size_t i=0; i<ranges.size(); i++)
                castedranges[i] = get_rangedata(ranges_ptr[i]);

            tagRanges = 1;
            nranges_x = get_size(ranges_ptr[0], 0) ;
            nranges_y = get_size(ranges_ptr[3], 0) ;

            nredranges_x = get_size(ranges_ptr[5], 0) ;
            nredranges_y = get_size(ranges_ptr[2], 0) ;
        }
        else {
            throw std::runtime_error(
                "[KeOps] the 'ranges' argument should be a tuple of size 0 or 6, "
                "but is of size " + std::to_string(ranges.size()) + "."
            );
        }
    } else if ( ranges.size() == 0 ) {  // Batch processing: we'll have to generate a custom, block-diagonal sparsity pattern
        tagRanges = 1;  // Batch processing is emulated through the block-sparse mode

        // We compute/read the number and size of our diagonal blocks ----------
        int nbatches = 1;
        for (int b = 0; b < nbatchdims; b++) {
            nbatches *= shapes[b];  // Compute the product of all "batch dimensions"
        }
        int M = shapes[nbatchdims], N = shapes[nbatchdims+1];

        // Create new "castedranges" from scratch ------------------------------
        // With pythonic notations, we'll have:
        //   castedranges = (ranges_i, slices_i, redranges_j,   ranges_j, slices_j, redranges_i)
        // with:
        // - ranges_i    = redranges_i = [ [0,M], [M,2M], ..., [(nbatches-1)M, nbatches*M] ]
        // - slices_i    = slices_j    = [    1,     2,   ...,   nbatches-1,   nbatches    ]
        // - redranges_j = ranges_j    = [ [0,N], [N,2N], ..., [(nbatches-1)N, nbatches*N] ]

        castedranges = new __INDEX__ *[6];
        castedranges[0] = new __INDEX__[2*nbatches];  // ranges_i
        castedranges[1] = new __INDEX__[nbatches];    // slices_i
        castedranges[2] = new __INDEX__[2*nbatches];  // redranges_j
        castedranges[3] = castedranges[2];            // ranges_j
        castedranges[4] = castedranges[1];            // slices_j
        castedranges[5] = castedranges[0];            // redranges_i

        for (int b = 0; b < nbatches; b++) {
            castedranges[0][2*b] = b*M; castedranges[0][2*b+1] = (b+1)*M;
            castedranges[1][b]   = b+1;
            castedranges[2][2*b] = b*N; castedranges[2][2*b+1] = (b+1)*N;
        }

        nranges_x = nbatches; nredranges_x = nbatches;
        nranges_y = nbatches; nredranges_y = nbatches;

    } else {
            throw std::runtime_error(
                "[KeOps] The 'ranges' argument (block-sparse mode) is not supported with batch processing, "
                "but we detected " + std::to_string(nbatchdims) + " > 0 batch dimensions."
            );
    }
    

    // Store, in a raw int array, the shape of the output: =====================
    // [A, .., B, M, D]  if TAGIJ==0
    //  or
    // [A, .., B, N, D]  if TAGIJ==1

    int *shape_output = new int[nbatchdims+2];
    for (int b = 0; b < nbatchdims; b++) {
            shape_output[b] = shapes[b];  // Copy the "batch dimensions"
    }
    shape_output[nbatchdims]   = shapes[nbatchdims+TAGIJ];  // M or N
    shape_output[nbatchdims+1] = shapes[nbatchdims+2];      // D


    // Call Cuda codes =========================================================
    array_t result = launch_keops<array_t>(tag1D2D, tagCpuGpu, tagHostDevice, Device_Id_s,
                            nx, ny,
                            nbatchdims, shapes, shape_output,
                            tagRanges, nranges_x, nranges_y, nredranges_x, nredranges_y, castedranges,
                            castedargs);


    // Free the allocated memory, return our output array ======================
    if ( nbatchdims != 0 ) {
        delete[] castedranges[0];  // ranges_i = redranges_i
        delete[] castedranges[1];  // slices_i = slices_j
        delete[] castedranges[2];  // redranges_j = ranges_j
    }

    delete[] castedargs;
    delete[] castedranges;
    delete[] shapes;
    delete[] shape_output;

    return result;

}


}
