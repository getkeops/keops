#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include "pybind11_numpy_scalar.h"
#if USE_HALF
//#include <half.h>
#endif

#include "common/keops_io.h"



namespace pykeops {

using namespace keops;
namespace py = pybind11;

#if USE_HALF
// adapted from https://github.com/eacousineau/repro/blob/43407e3/python/pybind11/custom_tests/test_numpy_issue1776.cc#L78-L79
using float16 = half;
static_assert(sizeof(float16) == 2, "Bad size");

namespace pybind11 { namespace detail {

// Similar to enums in `pybind11/numpy.h`. Determined by doing:
// python3 -c 'import numpy as np; print(np.dtype(np.float16).num)'
constexpr int NPY_FLOAT16 = 23;

// Kinda following: https://github.com/pybind/pybind11/blob/9bb3313162c0b856125e481ceece9d8faa567716/include/pybind11/numpy.h#L1000
template <>
struct npy_format_descriptor<float16> {
  static constexpr auto name = _("float16");
  static pybind11::dtype dtype() {
    handle ptr = npy_api::get().PyArray_DescrFromType_(NPY_FLOAT16);
    return reinterpret_borrow<pybind11::dtype>(ptr);
  }
};

template <>
struct type_caster<float16> : npy_scalar_caster<float16> {
  static constexpr auto name = _("float16");
};

}}  // namespace pybind11::detail
#endif


// <__TYPE__, py::array::c_style>  ensures 2 things whatever is the arguments:
//  1) the precision used is __TYPE__ (float or double typically) on the device,
//  2) everything is convert as contiguous before being loaded in memory
// this is maybe not the best in term of performance... but at least it is safe.
using __NUMPYARRAY__ = py::array_t<__TYPE__, py::array::c_style>;
using __RANGEARRAY__ = py::array_t<__INDEX__, py::array::c_style>;

/////////////////////////////////////////////////////////////////////////////////
//                             Utils
/////////////////////////////////////////////////////////////////////////////////

template <>
int get_ndim(__NUMPYARRAY__ obj_ptri){
    return obj_ptri.ndim();
}

template <>
int get_size(__NUMPYARRAY__ obj_ptri, int l){
    return obj_ptri.shape(l);
}

template <>
int get_size(__RANGEARRAY__ obj_ptri, int l){
    return obj_ptri.shape(l);
}

template <>
__TYPE__* get_data(__NUMPYARRAY__ obj_ptri){
    return (__TYPE__ *) obj_ptri.data();
}

template <>
__INDEX__* get_rangedata(__RANGEARRAY__ obj_ptri){
    return (__INDEX__ *) obj_ptri.data();
}

template <>
bool is_contiguous(__NUMPYARRAY__ obj_ptri){
    return obj_ptri.c_style;  // always true because of py::array::c_style
}

/////////////////////////////////////////////////////////////////////////////////
//                    Call Cuda functions
/////////////////////////////////////////////////////////////////////////////////


template <>
__NUMPYARRAY__ launch_keops(int tag1D2D, int tagCpuGpu, int tagHostDevice, short int Device_Id,
                        int nx, int ny, int nbatchdims, int *shapes, int *shape_out,
                        int tagRanges, int nranges_x, int nranges_y, int nredranges_x, int nredranges_y, __INDEX__ **castedranges,
                        __TYPE__ ** castedargs){

    // Create a new result array of shape [A, .., B, M, D] or [A, .., B, N, D]:
    std::vector<int> shape_vector(shape_out, shape_out+nbatchdims+2);
    auto result_array = __NUMPYARRAY__(shape_vector);

    if (tagCpuGpu == 0) {
        if (tagRanges == 0) { // Full M-by-N computation
            CpuReduc(nx, ny, get_data(result_array), castedargs);
        } else if( tagRanges == 1) { // Block sparsity
            CpuReduc_ranges(nx, ny, nbatchdims, shapes, nranges_x, nranges_y, castedranges, get_data(result_array), castedargs);
        }
    }
    else if (tagCpuGpu == 1) {
#if USE_CUDA
        if (tagHostDevice == 0) {
            if (tagRanges == 0) { // Full M-by-N computation
                if (tag1D2D == 0)
                    GpuReduc1D_FromHost( nx, ny, get_data(result_array), castedargs, Device_Id);
                else if (tag1D2D == 1)
                    GpuReduc2D_FromHost( nx, ny, get_data(result_array), castedargs, Device_Id);
            } else if( tagRanges == 1) { // Block sparsity
                GpuReduc1D_ranges_FromHost(nx, ny, nbatchdims, shapes, nranges_x, nranges_y, nredranges_x, nredranges_y,
                    castedranges, get_data(result_array), castedargs, Device_Id);
            }
        } else if (tagHostDevice==1)
            throw std::runtime_error("[KeOps] Gpu computations with Numpy are performed from host data... try to set tagHostDevice to 0.");
#else
        throw std::runtime_error(Error_msg_no_cuda);
#endif
    }

    return result_array;
}



/////////////////////////////////////////////////////////////////////////////////
//                    PyBind11 entry point
/////////////////////////////////////////////////////////////////////////////////

// the following macro force the compiler to change MODULE_NAME to its value
#define VALUE_OF(x) x

#define xstr(s) str(s)
#define str(s) #s

PYBIND11_MODULE(VALUE_OF(MODULE_NAME), m) {
    m.doc() = "This module has been generated by pyKeOps."; // optional module docstring

    m.def("genred_numpy",
          &generic_red<__NUMPYARRAY__,__RANGEARRAY__>,
          "Entry point to keops - numpy version.");

    m.attr("tagIJ") = TAGIJ;
    m.attr("dimout") = DIMOUT;
    m.attr("formula") = f;
    m.attr("compiled_formula") = xstr(FORMULA_OBJ_STR);
    m.attr("compiled_aliases") = xstr(VAR_ALIASES_STR);
}

}
