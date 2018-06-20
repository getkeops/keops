#include <vector>
#include <string>
#include "formula.h"

#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
//#include <include/pybind11/pybind11.h>
//#include <include/pybind11/numpy.h>

extern "C" int CpuConv(int, int, __TYPE__*, __TYPE__**);
extern "C" int CpuTransConv(int, int, __TYPE__*, __TYPE__**);

#ifdef USE_CUDA
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

    // ------ get the dimension from formula -----//

    using VARSI = typename F::template VARS<0>;    // list variables of type I
    using VARSJ = typename F::template VARS<1>; // list variables of type J
    using VARSP = typename F::template VARS<2>; // list variables of type parameter

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

    const int NARGS = NARGSI + NARGSJ + NARGSP;

    const std::string f =  PrintFormula<F>();

int GET_NARGS() {
    int NARGS2 = 0;
    for(int k=0; k<VARSI::SIZE; k++){
        NARGS2 =( INDSI::VAL(k) > NARGS2) ? INDSI::VAL(k) : NARGS2;
    }
    for(int k=0; k<VARSJ::SIZE; k++){
        NARGS2 =( INDSJ::VAL(k) > NARGS2) ? INDSJ::VAL(k) : NARGS2;
    }
    for(int k=0; k<VARSP::SIZE; k++){
        NARGS2 =( INDSP::VAL(k) > NARGS2) ? INDSP::VAL(k) : NARGS2;
    }

    return NARGS2 +1;
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
                nx = obj_ptr[i].size(0) ; // get nx
            } else if (nx != obj_ptr[i].size(0) ) {
                throw std::runtime_error("[Keops] Wrong number of rows for args number " +  std::to_string(i) + " : is " +
                        std::to_string(obj_ptr[i].size(0)) + " but should be " +std::to_string(nx)) ;
            }

            // column
            if (obj_ptr[i].size(1) != dimargs[i]) {
                throw std::runtime_error("[Keops] Wrong number of column for args number " +  std::to_string(i) + " : is "
                        + std::to_string(obj_ptr[i].size(1)) + " but should be " +std::to_string(dimargs[i])) ;
            }
        } else if (typeargs[i] == 1) {
            if (ny == 0 ) {
                ny = obj_ptr[i].size(0) ; // get ny
            } else if (ny != obj_ptr[i].size(0) ) {
                throw std::runtime_error("[Keops] Wrong number of rows for args number " +  std::to_string(i) + " : is "
                        + std::to_string(obj_ptr[i].size(0)) + " but should be " +std::to_string(ny) );
            }
            // column
            if (obj_ptr[i].size(1) != dimargs[i]) {
                throw std::runtime_error("[Keops] Wrong number of column for args number " +  std::to_string(i) + " : is "
                        + std::to_string(obj_ptr[i].size(1)) + " but should be " +std::to_string(dimargs[i])) ;
            }

        } else if (typeargs[i] == 2) {
            if (obj_ptr[i].size(0) != dimargs[i]) {
                throw std::runtime_error("[Keops] Wrong number of elements for args number " +  std::to_string(i) + " : is "
                        + std::to_string(obj_ptr[i].size(0)) + " but should be " +std::to_string(dimargs[i])) ;
            }
        }
    }

    delete[] dimargs;
    delete[] typeargs;

    return std::make_pair(nx,ny);
}

py::array_t<__TYPE__> generic_red(int tagIJ,
                                  int tagCpuGpu,
                                  int tag1D2D,
                                  py::args py_args) {



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

    if (py_args.size() != NARGS)
        throw std::runtime_error("[Keops] Wrong number of args : is " + std::to_string(py_args.size()) + " but should be " +std::to_string(NARGS)) ;

    std::array< py::array_t<__TYPE__>, NARGS > obj_ptr;
    __TYPE__ **castedargs = new __TYPE__ *[NARGS];
    // Seems to be necessary to store every pointer in an array to keep in memory the right address
    for (size_t i = 0; i < NARGS; i++) {
        obj_ptr[i] = py::cast<py::array_t<__TYPE__>>(py_args[i]);
    }

    int nx = 0;
    int ny = 0;
    for (size_t i = 0; i < NARGS; i++) {
        // check the dimension :
        if (typeargs[i] == 0) {
            if (nx == 0 ) {
                nx = obj_ptr[i].shape(0) ; // get nx
            } else if (nx != obj_ptr[i].shape(0) ) {
                throw std::runtime_error("Wrong number of rows for args number " +  std::to_string(i) + " : is " +
                        std::to_string(obj_ptr[i].shape(0)) + " but should be " +std::to_string(nx)) ;
            }

            // column
            if (obj_ptr[i].shape(1) != dimargs[i]) {
                throw std::runtime_error("Wrong number of column for args number " +  std::to_string(i) + " : is "
                        + std::to_string(obj_ptr[i].shape(1)) + " but should be " +std::to_string(dimargs[i])) ;
            }
        } else if (typeargs[i] == 1) {
            if (ny == 0 ) {
                ny = obj_ptr[i].shape(0) ; // get ny
            } else if (ny != obj_ptr[i].shape(0) ) {
                throw std::runtime_error("Wrong number of rows for args number " +  std::to_string(i) + " : is "
                        + std::to_string(obj_ptr[i].shape(0)) + " but should be " +std::to_string(ny) );
            }
            // column
            if (obj_ptr[i].shape(1) != dimargs[i]) {
                throw std::runtime_error("Wrong number of column for args number " +  std::to_string(i) + " : is " 
                        + std::to_string(obj_ptr[i].shape(1)) + " but should be " +std::to_string(dimargs[i])) ;
            }

        } else if (typeargs[i] == 2) {

            if (obj_ptr[i].shape(0) != dimargs[i]) {
                throw std::runtime_error("Wrong number of elements for args number " +  std::to_string(i) + " : is " 
                        + std::to_string(obj_ptr[i].shape(0)) + " but should be " +std::to_string(dimargs[i])) ;
            }
        }

        // get memory address
        castedargs[i] = (__TYPE__ *) obj_ptr[i].data();
    }

    // dimension Output
    int dimout = F::DIM;
    int nout;
    if (tagIJ == 0) {
        nout = nx;
    }
    else {
        nout = ny;
    }
    auto result_array = py::array_t<__TYPE__>({nout,dimout});
    __TYPE__ *result = (__TYPE__ *) result_array.data();

    //auto buf3 = result_array.request();
    //__TYPE__ *result = (__TYPE__ *) buf3.ptr;


    //////////////////////////////////////////////////////////////
    // Call Cuda codes
    //////////////////////////////////////////////////////////////

    // tagIJ=0 means sum over j, tagIJ=1 means sum over j
    // tagCpuGpu=0 means convolution on Cpu, tagCpuGpu=1 means convolution on Gpu, tagCpuGpu=2 means convolution on Gpu from device data
    // tag1D2D=0 means 1D Gpu scheme, tag1D2D=1 means 2D Gpu scheme

    if (tagCpuGpu == 0) {
        if (tagIJ == 0) {
            CpuConv(nx, ny, result, castedargs);
        } else {
            CpuTransConv(nx, ny, result, castedargs);
        }
    }
#if USE_CUDA
    else if(tagCpuGpu==1) {
        if(tagIJ==0) {
            if(tag1D2D==0) {
                GpuConv1D( nx, ny, result, castedargs);
            } else {
                GpuConv2D( nx, ny, result, castedargs);
            }
        } else {
            if(tag1D2D==0) {
                GpuTransConv1D( nx, ny, result, castedargs);
            } else {
                GpuTransConv2D( nx, ny, result, castedargs);
            }
        }
    }
#endif

    delete[] castedargs;
    delete[] dimargs;
    delete[] typeargs;

    return result_array;
}


at::Tensor generic_red_from_device(int tagIJ,
                                  int tag1D2D,
                                  int tagHostDevice,
                                  py::args py_args) {

    // Check number of variables
    int NARGS2 = GET_NARGS();
    if (py_args.size() != NARGS2)
        throw std::runtime_error("[Keops] Wrong number of args : is "
        + std::to_string(py_args.size()) + " but should be " +std::to_string(NARGS2)
        + " in "  + f) ;

    // Cast the input variable : It may be a copy here...
    auto obj_ptr = py::cast<std::vector<at::Tensor>>(py_args);

    // get the pointers to data to avoid a copy
    __TYPE__ **castedargs = new __TYPE__ *[NARGS2];
    int i=-1;
    for(auto argi : obj_ptr) {
        i += 1;
        castedargs[i] = argi.data<__TYPE__>();
    }

    // Check all the dimensions
    std::pair<int,int> n = check_args<at::Tensor>(obj_ptr);
    int nx = n.first; int ny = n.second;

    // dimension Output
    int dimout = F::DIM;
    int nout = (tagIJ == 0)? nx : ny;
    at::Tensor result_array = at::empty(torch::CUDA(at::kFloat), {nout,dimout});

    //////////////////////////////////////////////////////////////
    // Call Cuda codes
    //////////////////////////////////////////////////////////////

    // tagIJ=0 means sum over j, 
    // tag1D2D=0 means 1D Gpu scheme,

    if(tagIJ==0) {
        if(tag1D2D==0) {
            //    __TYPE__ *result = (__TYPE__ *) result_array.data_ptr();  == result_array.data<__TYPE__>()
            GpuConv1D_FromDevice(nx, ny, result_array.data<__TYPE__>(), castedargs);
        } else {
            GpuConv2D_FromDevice(nx, ny, result_array.data<__TYPE__>(), castedargs);
        }
    } else {
        if(tag1D2D==0) {
            GpuTransConv1D_FromDevice(nx, ny, result_array.data<__TYPE__>(), castedargs);
        } else {
            GpuTransConv2D_FromDevice(nx, ny, result_array.data<__TYPE__>(), castedargs);
        }
    }

    delete[] castedargs;

    return result_array;
}


// the following macro force the compilator to change MODULE_NAME to its value
#define VALUE_OF(x) x

PYBIND11_MODULE(VALUE_OF(MODULE_NAME), m) {
    m.doc() = "keops io through pybind11"; // optional module docstring

    m.def("gen_red", &generic_red, "A function...");
    m.def("gen_red_from_device", &generic_red_from_device, "A function...");

    int NARGS2 = GET_NARGS();
    m.attr("nargs") = NARGS2;
    m.attr("dimout") = F::DIM;
//    m.def("print_formula", &PrintFormula<F>, "Print formula");
    m.attr("formula") = f ;
    
    }

}
