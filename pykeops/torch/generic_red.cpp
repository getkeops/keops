#include <torch/torch.h>
#include <pybind11/pybind11.h>
//#include <pybind11/numpy.h>

#include "common/keops_io.h"

namespace pykeops {

using namespace keops;
namespace py = pybind11;

/////////////////////////////////////////////////////////////////////////////////
//                             Utils
/////////////////////////////////////////////////////////////////////////////////

template <>
int get_size(at::Tensor obj_ptri, int l){
    return obj_ptri.size(l);
}

template <>
__TYPE__* get_data(at::Tensor obj_ptri){
    return obj_ptri.data<__TYPE__>();
}


/////////////////////////////////////////////////////////////////////////////////
//                    Call Cuda functions
/////////////////////////////////////////////////////////////////////////////////


#if USE_DOUBLE
    #define AT_TYPE at::kDouble
#else
    #define AT_TYPE at::kFloat
#endif

template <>
at::Tensor launch_keops(int tag1D2D, int tagCpuGpu, int tagHostDevice,
                        int nx, int ny, int nout, int dimout,
                        __TYPE__ ** castedargs){

    if(tagHostDevice == 0) {

        at::Tensor result_array = at::empty(torch::CPU(AT_TYPE), {nout,dimout});

        if (tagCpuGpu == 0) {
            CpuReduc(nx, ny, get_data(result_array), castedargs);
            return result_array;
        } else if(tagCpuGpu==1) {
#if USE_CUDA
            if(tag1D2D==0) 
                GpuReduc1D_FromHost( nx, ny, get_data(result_array), castedargs);
            else if(tag1D2D==1)
                GpuReduc2D_FromHost( nx, ny, get_data(result_array), castedargs);
            return result_array;
#else
            throw std::runtime_error("[KeOps] No cuda device detected... try to set tagCpuGpu to 0.");
#endif
        }
    } else if(tagHostDevice == 1) {
#if USE_CUDA
        at::Tensor result_array = at::empty(torch::CUDA(AT_TYPE), {nout,dimout});
        if(tag1D2D==0)
            GpuReduc1D_FromDevice(nx, ny, get_data(result_array), castedargs);
        else if(tag1D2D==1)
            GpuReduc2D_FromDevice(nx, ny, get_data(result_array), castedargs);
        return result_array;
#else
        throw std::runtime_error("[KeOps] No cuda device detected... try to set tagHostDevice to 0.");
#endif
    }
    throw std::runtime_error("[KeOps] Meooooooooooooooooow...");
}


/////////////////////////////////////////////////////////////////////////////////
//                    PyBind11 entry point
/////////////////////////////////////////////////////////////////////////////////


// the following macro force the compilator to change MODULE_NAME to its value
#define VALUE_OF(x) x

PYBIND11_MODULE(VALUE_OF(MODULE_NAME), m) {
    m.doc() = "keops for pytorch through pybind11"; // optional module docstring

    m.def("genred_pytorch",
          &generic_red<at::Tensor>,
          "Entry point to keops - pytorch version.");

    m.attr("nargs") = NARGS;
    m.attr("dimout") = DIMOUT;
    m.attr("formula") = f;
}

}
