#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "common/keops_io.h"

namespace pykeops {

using namespace keops;
namespace py = pybind11;


/////////////////////////////////////////////////////////////////////////////////
//                             Utils
/////////////////////////////////////////////////////////////////////////////////

template <>
int get_size(py::array_t<__TYPE__, py::array::c_style> obj_ptri, int l){
    return obj_ptri.shape(l);
}

template <>
__TYPE__* get_data(py::array_t<__TYPE__, py::array::c_style> obj_ptri){
    return (__TYPE__ *) obj_ptri.data();
}


/////////////////////////////////////////////////////////////////////////////////
//                    Call Cuda functions
/////////////////////////////////////////////////////////////////////////////////


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

    m.attr("nargs") = NARGS;
    m.attr("dimout") = DIMOUT;
    m.attr("formula") = f;
}

}