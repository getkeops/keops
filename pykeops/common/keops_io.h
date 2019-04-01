#include <vector>
#include <string>
// #include "formula.h" done by cmake

extern "C" {
    int CpuReduc(int, int, __TYPE__*, __TYPE__**);
    int CpuReduc_ranges(int, int, int, int, __INDEX__**, __TYPE__*, __TYPE__**);
};

#if USE_CUDA
extern "C" {
    int GpuReduc1D_FromHost(int, int, __TYPE__*, __TYPE__**, int);
    int GpuReduc1D_FromDevice(int, int, __TYPE__*, __TYPE__**, int);
    int GpuReduc2D_FromHost(int, int, __TYPE__*, __TYPE__**, int);
    int GpuReduc2D_FromDevice(int, int, __TYPE__*, __TYPE__**, int);
    int GpuReduc1D_ranges_FromHost(int, int, int, int, int, int, __INDEX__**, __TYPE__*, __TYPE__**, int);
    int GpuReduc1D_ranges_FromDevice(int, int, int, int, __INDEX__**, __TYPE__*, __TYPE__**, int);
};
#endif

namespace pykeops {

using namespace keops;
namespace py = pybind11;

/////////////////////////////////////////////////////////////////////////////////
//                           Keops
/////////////////////////////////////////////////////////////////////////////////


using FF = F::F; // F::F is formula inside reduction (ex if F is Sum_Reduction<Form> then F::F is Form)

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
int get_size(array_t obj_ptri, int l);

template< typename array_t >
__TYPE__* get_data(array_t obj_ptri);

template< typename array_t >
__INDEX__* get_rangedata(array_t obj_ptri);

template< typename array_t >
bool is_contiguous(array_t obj_ptri);


/////////////////////////////////////////////////////////////////////////////////
//                    Sanity checks on args
/////////////////////////////////////////////////////////////////////////////////


void check_tag(int tag, std::string msg){
    if ((tag < 0) || (tag > 1)) {
        throw std::runtime_error("[KeOps] tag" + msg + " should be (0 or 1) but is " + std::to_string(tag));
    }
}

template<typename array_t>
void check_args(int nx, int ny, std::vector<array_t> obj_ptr) {

    if (NARGS>0) {
        // ------ check the dimensions ------------//
        int *typeargs = new int[NARGS];
        int *dimargs = new int[NARGS];

        for (int k = 0; k < NARGS; k++) {
            typeargs[k] = -1;
            dimargs[k] = -1;
        }
        for (int k = 0; k < NARGSI; k++) {
            typeargs[INDSI::VAL(k)] = 0;
            dimargs[INDSI::VAL(k)] = DIMSX::VAL(k);
        }
        for (int k = 0; k < NARGSJ; k++) {
            typeargs[INDSJ::VAL(k)] = 1;
            dimargs[INDSJ::VAL(k)] = DIMSY::VAL(k);
        }
        for (int k = 0; k < NARGSP; k++) {
            typeargs[INDSP::VAL(k)] = 2;
            dimargs[INDSP::VAL(k)] = DIMSP::VAL(k);
        }

        // check  the dimension :
        for (size_t i = 0; i < NARGS; i++) {
            if (typeargs[i] == 0) {
                if (nx != get_size(obj_ptr[i],0)) {
                    throw std::runtime_error("[KeOps] Wrong number of rows for arg number " + std::to_string(i) + " : is "
                            + std::to_string(get_size(obj_ptr[i],0)) + " but should be " + std::to_string(nx));
                }

                // column
                if (get_size(obj_ptr[i],1) != dimargs[i]) {
                    throw std::runtime_error("[KeOps] Wrong number of column for arg number " + std::to_string(i) + " : is "
                            + std::to_string(get_size(obj_ptr[i],1)) + " but should be " + std::to_string(dimargs[i])) ;
                }
            } else if (typeargs[i] == 1) {
                if (ny != get_size(obj_ptr[i],0) ) {
                    throw std::runtime_error("[KeOps] Wrong number of rows for arg number " + std::to_string(i) + " : is "
                            + std::to_string(get_size(obj_ptr[i],0)) + " but should be " + std::to_string(ny));
                }
                // column
                if (get_size(obj_ptr[i],1) != dimargs[i]) {
                    throw std::runtime_error("[KeOps] Wrong number of column for arg number " + std::to_string(i) + " : is "
                            + std::to_string(get_size(obj_ptr[i],1)) + " but should be " + std::to_string(dimargs[i])) ;
                }

            } else if (typeargs[i] == 2) {
                if (get_size(obj_ptr[i],0) != dimargs[i]) {
                    throw std::runtime_error("[KeOps] Wrong number of elements for arg number " + std::to_string(i) + " : is "
                            + std::to_string(get_size(obj_ptr[i],0)) + " but should be " + std::to_string(dimargs[i])) ;
                }
            }

            if (!is_contiguous(obj_ptr[i])) {
                throw std::runtime_error("[KeOps] Arg number " + std::to_string(i) + " : is not contiguous. "
                        + "Please provide 'contiguous' dara array, as KeOps does not support strides. "
                        + "If you're getting this error in the 'backward' pass of a code using torch.sum() "
                        + "on the output of a KeOps routine, you should consider replacing 'a.sum()' with "
                        + "'torch.dot(a.view(-1), torch.ones_like(a).view(-1))'. ") ;
            }
        }

        delete[] dimargs;
        delete[] typeargs;
    }
}

short int cast_Device_Id(int Device_Id){
  if (Device_Id <std::numeric_limits<short int>::max()) {
    return(Device_Id);
  } else {
    throw std::runtime_error("[keOps] Device_Id exceeded short int limit");
  }
}

/////////////////////////////////////////////////////////////////////////////////
//                    Call Cuda functions
/////////////////////////////////////////////////////////////////////////////////

// Implemented by pykeops/torch/generic_red.cpp or pykeops/numpy/generic_red.cpp
template < typename array_t >
array_t launch_keops(int tag1D2D, int tagCpuGpu, int tagHostDevice, short int Device_Id,
                        int nx, int ny, int nout, int dimout,
                        int tagRanges, int nranges_x, int nranges_y, int nredranges_x, int nredranges_y, __INDEX__ **castedranges,
                        __TYPE__ ** castedargs);


/////////////////////////////////////////////////////////////////////////////////
//                    Main function
/////////////////////////////////////////////////////////////////////////////////

template < typename array_t, typename index_t >
array_t generic_red(int nx, int ny,
                    int tagCpuGpu,        // tagCpuGpu=0     means Reduction on Cpu, tagCpuGpu=1       means Reduction on Gpu, tagCpuGpu=2 means Reduction on Gpu from device data
                    int tag1D2D,          // tag1D2D=0       means 1D Gpu scheme,      tag1D2D=1       means 2D Gpu scheme
                    int tagHostDevice,    // tagHostDevice=1 means _fromDevice suffix. tagHostDevice=0 means _fromHost suffix
                    int Device_Id,        // id of GPU device
                    py::tuple ranges,     // () if no "sparsity" ranges are given (default behavior)
                                          // Otherwise, ranges is a 6-uple of (integer) array_t
                                          // ranges = (ranges_i, slices_i, redranges_j, ranges_j, slices_j, redranges_i)
                                          // as documented in the doc on sparstiy and clustering.
                    py::args py_args) {

    // Checks
    if (py_args.size() < NARGS) {
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

    // Check all the dimensions
    check_args<array_t>(nx, ny, obj_ptr);

    // dimension Output : nout is the nbr of rows of the result
    int nout = (TAGIJ == 0)? nx : ny;


    int tagRanges, nranges_x, nranges_y, nredranges_x, nredranges_y ;
    __INDEX__ **castedranges;

    // Sparsity: should we handle ranges?
    if(ranges.size() == 0) {
        tagRanges = 0; 
        nranges_x = 0; nranges_y = 0 ;
        nredranges_x = 0; nredranges_y = 0 ;
        castedranges = new __INDEX__ *[1];
    }
    else if(ranges.size() == 6) {
        // Cast the six integer arrays
        std::vector<index_t> ranges_ptr(ranges.size());
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



    // Call Cuda codes
    array_t result = launch_keops<array_t>(tag1D2D, tagCpuGpu, tagHostDevice, Device_Id_s,
                            nx, ny,
                            nout, F::DIM,      // dimout, nout
                            tagRanges, nranges_x, nranges_y, nredranges_x, nredranges_y, castedranges,
                            castedargs);

    delete[] castedargs;

    return result;

}


}
