#include <torch/torch.h>
#include <pybind11/pybind11.h>

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

template <>
bool is_contiguous(at::Tensor obj_ptri){
    return obj_ptri.is_contiguous();
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
at::Tensor launch_keops(int tag1D2D, int tagCpuGpu, int tagHostDevice, int Device_Id,
                        int nx, int ny, int nout, int dimout,
                        __TYPE__ ** castedargs){
    
    if(tagHostDevice == 0) {

        at::Tensor result_array = at::empty({nout,dimout},torch::CPU(AT_TYPE));
        torch::set_requires_grad(result_array, true);

        if (tagCpuGpu == 0) {
            CpuReduc(nx, ny, get_data(result_array), castedargs);
            return result_array;
        } else if(tagCpuGpu==1) {
#if USE_CUDA
            if(tag1D2D==0) 
                GpuReduc1D_FromHost( nx, ny, get_data(result_array), castedargs, Device_Id);
            else if(tag1D2D==1)
                GpuReduc2D_FromHost( nx, ny, get_data(result_array), castedargs, Device_Id);
            return result_array;
#else
            throw std::runtime_error("[KeOps] No cuda device detected... try to set tagCpuGpu to 0.");
#endif
        }
    } else if(tagHostDevice == 1) {
#if USE_CUDA       
       at::Tensor result_array = at::empty({nout,dimout}, {torch::CUDA(AT_TYPE),Device_Id});
       torch::set_requires_grad(result_array, true);
        if(tag1D2D==0)
            GpuReduc1D_FromDevice(nx, ny, get_data(result_array), castedargs, Device_Id);
        else if(tag1D2D==1)
            GpuReduc2D_FromDevice(nx, ny, get_data(result_array), castedargs, Device_Id);
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


// the following macro force the compiler to change MODULE_NAME to its value
#define VALUE_OF(x) x

#define xstr(s) str(s)
#define str(s) #s

PYBIND11_MODULE(VALUE_OF(MODULE_NAME), m) {
    m.doc() = "keops for pytorch through pybind11"; // optional module docstring

    m.def("genred_pytorch",
          &generic_red<at::Tensor>,
          "Entry point to keops - pytorch version.");

    m.attr("tagIJ") = TAGIJ;
    m.attr("dimout") = DIMOUT;
    m.attr("formula") = f;
    m.attr("compiled_formula") = xstr(FORMULA_OBJ_STR);
    m.attr("compiled_aliases") = xstr(VAR_ALIASES_STR);
}

}
