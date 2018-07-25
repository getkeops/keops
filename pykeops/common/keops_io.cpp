#include <vector>
#include <string>
// #include "formula.h" done by cmake

#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


extern "C" {
    int CpuConv(int, int, __TYPE__*, __TYPE__**);
    int CpuTransConv(int, int, __TYPE__*, __TYPE__**);
};

#if USE_CUDA
extern "C" {
    int GpuConv1D(int, int, __TYPE__*, __TYPE__**);
    int GpuConv1D_FromDevice(int, int, __TYPE__*, __TYPE__**);
    int GpuConv2D(int, int, __TYPE__*, __TYPE__**);
    int GpuConv2D_FromDevice(int, int, __TYPE__*, __TYPE__**);
    int GpuTransConv1D(int, int, __TYPE__*, __TYPE__**);
    int GpuTransConv1D_FromDevice(int, int, __TYPE__*, __TYPE__**);
    int GpuTransConv2D(int, int, __TYPE__*, __TYPE__**);
    int GpuTransConv2D_FromDevice(int, int, __TYPE__*, __TYPE__**);
};
#endif

namespace pykeops {

using namespace keops;
namespace py = pybind11;

/////////////////////////////////////////////////////////////////////////////////
//                           Keops
/////////////////////////////////////////////////////////////////////////////////

using VARSI = typename F::template VARS<0>;    // list variables of type I
using VARSJ = typename F::template VARS<1>;    // list variables of type J
using VARSP = typename F::template VARS<2>;    // list variables of type parameter

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

const int NARGS = Generic<F>::sEval::NMINARGS;
const int DIMOUT = F::DIM;

const std::string f =  PrintFormula<F>();


/////////////////////////////////////////////////////////////////////////////////
//        Operator overloading (py::array = numpy, at::Tensor = pytorch)
/////////////////////////////////////////////////////////////////////////////////

int get_size(py::array_t<__TYPE__, py::array::c_style> obj_ptri, int l){
    return obj_ptri.shape(l);
}

int get_size(at::Tensor obj_ptri, int l){
    return obj_ptri.size(l);
}


__TYPE__* get_data(py::array_t<__TYPE__, py::array::c_style> obj_ptri){
    return (__TYPE__ *) obj_ptri.data();
}

__TYPE__* get_data(at::Tensor obj_ptri){
    return obj_ptri.data<__TYPE__>();
}


/////////////////////////////////////////////////////////////////////////////////
//                    Sanity checks on args
/////////////////////////////////////////////////////////////////////////////////


void check_tag(int tag, std::string msg){
    if ((tag < 0) || (tag > 1))
        throw std::runtime_error("[Keops] tag" + msg + " should be (0 or 1) but is " + std::to_string(tag));
}

template<typename array_t>
std::pair<int,int> check_args(std::vector<array_t> obj_ptr) {
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
    int nx = 0;
    int ny = 0;
    for (size_t i = 0; i < NARGS; i++) {
        if (typeargs[i] == 0) {
            if (nx == 0 ) {
                nx = get_size(obj_ptr[i],0); // get nx
            } else if (nx != get_size(obj_ptr[i],0)) {
                throw std::runtime_error("[Keops] Wrong number of rows for arg number " + std::to_string(i) + " : is "
                        + std::to_string(get_size(obj_ptr[i],0)) + " but should be " + std::to_string(nx));
            }

            // column
            if (get_size(obj_ptr[i],1) != dimargs[i]) {
                throw std::runtime_error("[Keops] Wrong number of column for arg number " + std::to_string(i) + " : is "
                        + std::to_string(get_size(obj_ptr[i],1)) + " but should be " + std::to_string(dimargs[i])) ;
            }
        } else if (typeargs[i] == 1) {
            if (ny == 0 ) {
                ny = get_size(obj_ptr[i],0) ; // get ny
            } else if (ny != get_size(obj_ptr[i],0) ) {
                throw std::runtime_error("[Keops] Wrong number of rows for arg number " + std::to_string(i) + " : is "
                        + std::to_string(get_size(obj_ptr[i],0)) + " but should be " + std::to_string(ny));
            }
            // column
            if (get_size(obj_ptr[i],1) != dimargs[i]) {
                throw std::runtime_error("[Keops] Wrong number of column for arg number " + std::to_string(i) + " : is "
                        + std::to_string(get_size(obj_ptr[i],1)) + " but should be " + std::to_string(dimargs[i])) ;
            }

        } else if (typeargs[i] == 2) {
            if (get_size(obj_ptr[i],0) != dimargs[i]) {
                throw std::runtime_error("[Keops] Wrong number of elements for arg number " + std::to_string(i) + " : is "
                        + std::to_string(get_size(obj_ptr[i],0)) + " but should be " + std::to_string(dimargs[i])) ;
            }
        }
        //bool a = obj_ptr[i].flags().c_contiguous();
        //std::cout << a << std::endl;
        //if (!(obj_ptr[i].flags.c_contiguous()))
        //    throw std::runtime_error("[Keops] Arg number " + std::to_string(i) + " : is not contiguous. Abort.") ;

    }

    delete[] dimargs;
    delete[] typeargs;

    return std::make_pair(nx,ny);
}


/////////////////////////////////////////////////////////////////////////////////
//                    Call Cuda functions
/////////////////////////////////////////////////////////////////////////////////


template < typename array_t >
array_t launch_keops(int tagIJ, int tag1D2D, int tagCpuGpu, int tagHostDevice,
                        int nx, int ny, int nout, int dimout,
                        __TYPE__ ** castedargs);

template <>
py::array_t< __TYPE__, py::array::c_style > launch_keops(int tagIJ, int tag1D2D, int tagCpuGpu, int tagHostDevice,
                        int nx, int ny, int nout, int dimout,
                        __TYPE__ ** castedargs){

    auto result_array = py::array_t<__TYPE__, py::array::c_style>({nout,dimout});
    if (tagCpuGpu == 0) {

        if (tagIJ == 0) {
            CpuConv(nx, ny,  get_data(result_array), castedargs);
        } else if (tagIJ == 1) {
            CpuTransConv(nx, ny, get_data(result_array), castedargs);
        }

    } else if (tagCpuGpu == 1) {

#if USE_CUDA
        if (tagIJ == 0) {
            if (tag1D2D == 0) {
                GpuConv1D( nx, ny, get_data(result_array), castedargs);
            } else if (tag1D2D == 1) {
                GpuConv2D( nx, ny, get_data(result_array), castedargs);
            }
        } else if (tagIJ == 1) {
            if (tag1D2D == 0) {
                GpuTransConv1D( nx, ny, get_data(result_array), castedargs);
            } else if (tag1D2D == 1) {
                GpuTransConv2D( nx, ny, get_data(result_array), castedargs);
            }
        }
#else
        throw std::runtime_error("[KeOps] No cuda device detected... try to set tagCpuGpu to 0.");
#endif

    }
    return result_array;
}

#if USE_DOUBLE
    #define AT_TYPE at::kDouble
#else
    #define AT_TYPE at::kFloat
#endif

template <>
at::Tensor launch_keops(int tagIJ, int tag1D2D, int tagCpuGpu, int tagHostDevice,
                        int nx, int ny, int nout, int dimout,
                        __TYPE__ ** castedargs){

    if(tagHostDevice == 0) {

        at::Tensor result_array = at::empty(torch::CPU(AT_TYPE), {nout,dimout});

        if (tagCpuGpu == 0) {
            if (tagIJ == 0) {
                CpuConv(nx, ny, get_data(result_array), castedargs);
            } else if (tagIJ == 0) {
                CpuTransConv(nx, ny, get_data(result_array), castedargs);
            }
        } else if(tagCpuGpu==1) {
#if USE_CUDA
            if(tagIJ==0) {
                if(tag1D2D==0) {
                    GpuConv1D( nx, ny, get_data(result_array), castedargs);
                } else if(tag1D2D==1) {
                    GpuConv2D( nx, ny, get_data(result_array), castedargs);
                }
            } else if(tagIJ==0) {
                if(tag1D2D==0) {
                    GpuTransConv1D( nx, ny, get_data(result_array), castedargs);
                } else if(tag1D2D==1) {
                    GpuTransConv2D( nx, ny, get_data(result_array), castedargs);
                }
            }
            return result_array;
#else
            throw std::runtime_error("[KeOps] No cuda device detected... try to set tagCpuGpu to 0.");
#endif
        }
    } else if(tagHostDevice == 1) {

#if USE_CUDA
        at::Tensor result_array = at::empty(torch::CUDA(AT_TYPE), {nout,dimout});

        if(tagIJ==0) {
            if(tag1D2D==0) {
                GpuConv1D_FromDevice(nx, ny, get_data(result_array), castedargs);
            } else if(tag1D2D==1) {
                GpuConv2D_FromDevice(nx, ny, get_data(result_array), castedargs);
            }
        } else if(tagIJ==1) {
            if(tag1D2D==0) {
                GpuTransConv1D_FromDevice(nx, ny, get_data(result_array), castedargs);
            } else if(tag1D2D==1){
                GpuTransConv2D_FromDevice(nx, ny, get_data(result_array), castedargs);
            }
        }

        return result_array;
#else
        throw std::runtime_error("[KeOps] No cuda device detected... try to set tagHostDevice to 0.");
#endif
    }
    throw std::runtime_error("[KeOps] Meooooooooooooooooow...");
}


/////////////////////////////////////////////////////////////////////////////////
//                    Main function
/////////////////////////////////////////////////////////////////////////////////

template < typename array_t >
array_t generic_red(int tagIJ,            // tagIJ=0         means sum over j,         tagIJ=1         means sum over j
                    int tag1D2D,          // tag1D2D=0       means 1D Gpu scheme,      tag1D2D=1       means 2D Gpu scheme
                    int tagCpuGpu,        // tagCpuGpu=0     means convolution on Cpu, tagCpuGpu=1     means convolution on Gpu, tagCpuGpu=2 means convolution on Gpu from device data
                    int tagHostDevice,    // tagHostDevice=1 means _fromDevice suffix. tagHostDevice=0 means no suffix
                    py::args py_args) {

    // Checks
    if (py_args.size() < NARGS)
        throw std::runtime_error(
        "[Keops] Wrong number of args : is " + std::to_string(py_args.size())
        + " but should be at least " + std::to_string(NARGS)
        + " in " + f
        );

    check_tag(tagIJ, "IJ");
    check_tag(tag1D2D, "1D2D");
    check_tag(tagCpuGpu, "CpuGpu");
    check_tag(tagHostDevice, "HostDevice");

    // Cast the input variable : It may be a copy here...
    auto obj_ptr = py::cast<std::vector<array_t>>(py_args);

    // get the pointers to data to avoid a copy
    __TYPE__ **castedargs = new __TYPE__ *[NARGS];
    for(auto i=0; i<NARGS; i++)
        castedargs[i] = get_data(obj_ptr[i]);

    // Check all the dimensions
    std::pair<int,int> n = check_args<array_t>(obj_ptr);  // int nx = n.first; int ny = n.second;

    // dimension Output : nout is the nbr of rows of the result
    int nout = (tagIJ == 0)? n.first : n.second;

    // Call Cuda codes
    array_t result = launch_keops<array_t>(tagIJ, tag1D2D, tagCpuGpu, tagHostDevice,
                            n.first, n.second, // nx, ny
                            nout, F::DIM,      // dimout, nout
                            castedargs);

    delete[] castedargs;

    return result;

}


/////////////////////////////////////////////////////////////////////////////////
//                    PyBind11 entry point
/////////////////////////////////////////////////////////////////////////////////


// the following macro force the compilator to change MODULE_NAME to its value
#define VALUE_OF(x) x

PYBIND11_MODULE(VALUE_OF(MODULE_NAME), m) {
    m.doc() = "keops io through pybind11"; // optional module docstring

    // <__TYPE__, py::array::c_style>  ensures 2 things whatever is the arguments:
    //  1) the precision used is __TYPE__ (float or double typically) on the device,
    //  2) everything is convert as contiguous before being loaded in memory
    // this is maybe not the best in term of performance... but at least it is safe.
    m.def("genred_numpy",
          &generic_red<py::array_t<__TYPE__, py::array::c_style>>,
          "Entry point to keops - numpy version.");

    m.def("genred_pytorch",
          &generic_red<at::Tensor>,
          "Entry point to keops - pytorch version.");

    m.attr("nargs") = NARGS;
    m.attr("dimout") = DIMOUT;
    m.attr("formula") = f;
}

}