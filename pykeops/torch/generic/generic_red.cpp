// Import done by Cmake
// #include <torch/extension.h>
// #include <pybind11/pybind11.h>

#include "common/keops_io.h"

namespace pykeops {

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

        auto result_array = torch::empty({nout, dimout}, at::device(at::kCPU).dtype(AT_TYPE).requires_grad(true));

        if (tagCpuGpu == 0) {
            CpuReduc(nx, ny, get_data(result_array), castedargs);
            return result_array;
        } else if(tagCpuGpu==1) {
#if USE_CUDA
            if(tag1D2D==0) 
                GpuReduc1D_FromHost(nx, ny, get_data(result_array), castedargs, Device_Id);
            else if(tag1D2D==1)
                GpuReduc2D_FromHost(nx, ny, get_data(result_array), castedargs, Device_Id);
            return result_array;
#else
            throw std::runtime_error("[KeOps] No cuda device detected... try to set tagCpuGpu to 0.");
#endif
        }
    } else if(tagHostDevice == 1) {
#if USE_CUDA       

        assert(Device_Id < std::numeric_limits<c10::DeviceIndex>::max());  // check that int will fit in a c10::DeviceIndex type
        auto result_array = torch::empty({nout, dimout}, at::device({at::kCUDA, static_cast<c10::DeviceIndex>(Device_Id)}).dtype(AT_TYPE).requires_grad(true));
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
