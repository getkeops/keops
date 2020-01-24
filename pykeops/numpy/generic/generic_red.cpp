#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// keops_binders import
#include "keops/binders/include.h"

// pykeops import
#include "common/keops_io.h"


using __NUMPYARRAY__ = py::array_t< __TYPE__, py::array::c_style >;
using __RANGEARRAY__ = py::array_t< __INDEX__, py::array::c_style >;


namespace py = pybind11;

namespace keops_binders {
/////////////////////////////////////////////////////////////////////////////////
//                  Template specialization (NumPy Arrays)                     //
/////////////////////////////////////////////////////////////////////////////////

// <__TYPE__, py::array::c_style>  ensures 2 things whatever is the arguments:
//  1) the precision used is __TYPE__ (float or double typically) on the device,
//  2) everything is convert as contiguous before being loaded in memory
// this is maybe not the best in term of performance... but at least it is safe.

template<>
int get_ndim(__NUMPYARRAY__ obj_ptri) {
  return obj_ptri.ndim();
}

template<>
int get_size(__NUMPYARRAY__ obj_ptri, int l) {
  return obj_ptri.shape(l);
}

template<>
__TYPE__* get_data(__NUMPYARRAY__ obj_ptri) {
  return const_cast< __TYPE__* >(obj_ptri.data());
}

template<>
bool is_contiguous(__NUMPYARRAY__ obj_ptri) {
  return obj_ptri.c_style;  // always true because of py::array::c_style
}

template<>
__NUMPYARRAY__ allocate_result_array< __NUMPYARRAY__, __TYPE__ >(int* shape_out, int nbatchdims) {
  // Create a new result array of shape [A, .., B, M, D] or [A, .., B, N, D]:
  std::vector< int > shape_vector(shape_out, shape_out + nbatchdims + 2);
  return __NUMPYARRAY__(shape_vector);
}

template <>
__NUMPYARRAY__ allocate_result_array_gpu< __NUMPYARRAY__, __TYPE__ >(int* shape_out, int nbatchdims,
                                                                     short int Device_Id) {
  throw std::runtime_error("[KeOps] numpy does not yet support nd array on GPU.");
}


template<>
int get_size(__RANGEARRAY__ obj_ptri, int l) {
  return obj_ptri.shape(l);
}

template<>
__INDEX__ *get_rangedata(__RANGEARRAY__ obj_ptri) {
  return const_cast< __INDEX__ * >(obj_ptri.data());
}

void keops_error(std::basic_string< char > msg) {
  throw std::runtime_error(msg);
}

}

using namespace keops;

/////////////////////////////////////////////////////////////////////////////////
//                    PyBind11 entry point                                     //
/////////////////////////////////////////////////////////////////////////////////


PYBIND11_MODULE(VALUE_OF(MODULE_NAME), m) {
m.doc() = "pyKeOps: KeOps for numpy through pybind11.";

m.def("genred_numpy", &generic_red <__NUMPYARRAY__, __RANGEARRAY__>, "Entry point to keops - numpy version.");

m.attr("tagIJ") = keops::TAGIJ;
m.attr("dimout") = keops::DIMOUT;
m.attr("formula") = keops::f;
m.attr("compiled_formula") = xstr(keops::FORMULA_OBJ_STR);
m.attr("compiled_aliases") = xstr(keops::VAR_ALIASES_STR);
}

