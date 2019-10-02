#pragma once

namespace keops_binders {

// the following macro force the compiler to change MODULE_NAME to its value
#define VALUE_OF(x) x

#define xstr(s) str(s)
#define str(s) #s

template < typename _T >
short int cast_Device_Id(_T Device_Id) {
  static_assert(std::is_integral< _T >::value, "Device_Id must be of integral type.");
  if (Device_Id < std::numeric_limits< short int >::max()) {
    return (static_cast<short int>(Device_Id));
  } else {
    throw std::runtime_error("[KeOps] Device_Id exceeded short int limit");
  }
}

const auto Error_msg_no_cuda =
    "[KeOps] This KeOps shared object has been compiled without cuda support: \n 1) to perform computations on CPU, simply set tagHostDevice to 0\n 2) to perform computations on GPU, please recompile the formula with a working version of cuda.";

}