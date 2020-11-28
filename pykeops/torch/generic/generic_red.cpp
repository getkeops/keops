// Keops and torch import done by Cmake
// #include <torch/extension.h>

#include <pybind11/pybind11.h>

// keops_binders import
#include "keops/binders/include.h"

// pykeops import
#include "common/keops_io.h"

namespace keops_binders {

/////////////////////////////////////////////////////////////////////////////////
//                  Template specialization (aTen Tensors)                     //
/////////////////////////////////////////////////////////////////////////////////

//Specialization of functions in keops/binders/checks.h

template <>
int get_ndim(at::Tensor obj_ptri) {
  return obj_ptri.dim();
}

template <>
int get_size(at::Tensor obj_ptri, int l) {
  return obj_ptri.size(l);
}

template <>
bool is_contiguous(at::Tensor obj_ptri) {
  return obj_ptri.is_contiguous();
}

#if USE_DOUBLE
  #define AT_kTYPE at::kDouble
  #define AT_TYPE double
#elif USE_HALF
  #define AT_kTYPE at::kHalf
  #define AT_TYPE at::Half
#else
  #define AT_kTYPE at::kFloat
  #define AT_TYPE float
#endif

template <>
__TYPE__* get_data< at::Tensor, __TYPE__ >(at::Tensor obj_ptri) {
  return (__TYPE__*)obj_ptri.data_ptr< AT_TYPE >();
}

template <>
at::Tensor allocate_result_array< at::Tensor, __TYPE__ >(int* shape_out, int nbatchdims) {
  // ATen only accepts "long int arrays" to specify the shape of a new tensor:
  int64_t shape_out_long[nbatchdims + 2];
  std::copy(shape_out, shape_out + nbatchdims + 2, shape_out_long);
  c10::ArrayRef < int64_t > shape_out_array(shape_out_long, (int64_t) nbatchdims + 2);

  return torch::empty(shape_out_array, at::device(at::kCPU).dtype(AT_kTYPE).requires_grad(true));

}


template <>
at::Tensor allocate_result_array_gpu< at::Tensor, __TYPE__ >(int* shape_out, int nbatchdims, short int Device_Id) {
#if USE_CUDA
  // ATen only accepts "long int arrays" to specify the shape of a new tensor:
  int64_t shape_out_long[nbatchdims + 2];
  std::copy(shape_out, shape_out + nbatchdims + 2, shape_out_long);
  c10::ArrayRef < int64_t > shape_out_array(shape_out_long, (int64_t) nbatchdims + 2);

  // Create a new result array of shape [A, .., B, M, D] or [A, .., B, N, D]:
  return torch::empty(shape_out_array, at::device({at::kCUDA, Device_Id}).dtype(AT_kTYPE).requires_grad(true));
#else
  keops_error(Error_msg_no_cuda);
  throw std::runtime_error("Simply here to avoid a warning at compilation.");
#endif
}

template <>
__INDEX__ *get_rangedata(at::Tensor obj_ptri) {
  return obj_ptri.data_ptr< __INDEX__ >();
}

void keops_error(std::basic_string< char > msg) {
  throw std::runtime_error(msg);
}
}

/////////////////////////////////////////////////////////////////////////////////
//                    PyBind11 entry point                                     //
/////////////////////////////////////////////////////////////////////////////////


PYBIND11_MODULE(VALUE_OF(MODULE_NAME), m) {
m.doc() = "pyKeOps: KeOps for pytorch through pybind11 (pytorch flavour).";

m.def("genred_pytorch", &generic_red <at::Tensor, at::Tensor>, "Entry point to keops - pytorch version.");

m.attr("tagIJ") = keops_binders::keops_tagIJ;
m.attr("dimout") = keops_binders::keops_dimout;
m.attr("formula") = keops_binders::keops_formula_string;

}


