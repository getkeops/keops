#pragma once

namespace keops_binders {

/////////////////////////////////////////////////////////////////////////////////
//                             Utils                                           //
/////////////////////////////////////////////////////////////////////////////////

// This function has to be specialized in the various binders:

template< typename array_t >
int get_ndim(array_t obj_ptri);  // len( a.shape )

template< typename array_t >
int get_size(array_t obj_ptri, int l);  // a.shape[l]

template< typename array_t, typename _T >
_T* get_data(array_t obj_ptri);   // raw pointer to "a.data"

template< typename array_t >
__INDEX__* get_rangedata(array_t obj_ptri);  // raw pointer to "a.data", casted as integer
template< typename array_t >
bool is_contiguous(array_t obj_ptri);  // is "a" ordered properly? KeOps does *not* support strides!


template< typename array_t, typename _T >
array_t allocate_result_array(int* shape_out, int nbatchdims = 0);

template< typename array_t, typename _T >
array_t allocate_result_array_gpu(int* shape_out, int nbatchdims = 0, short int Device_Id = 0);


void keops_error(std::string);

void keops_error(std::basic_string< char >);



// the following macro force the compiler to change MODULE_NAME to its value
#define VALUE_OF(x) x

#define xstr(s) str(s)
#define str(s) #s

template < typename _T >
short int cast_Device_Id(_T Device_Id) {
  // static_assert(std::is_integral< _T >::value, "Device_Id must be of integral type.");
  if (Device_Id < std::numeric_limits< short int >::max()) {
    return (static_cast<short int>(Device_Id));
  } else {
    throw std::runtime_error("[KeOps] Device_Id exceeded short int limit");
  }
}

const auto Error_msg_no_cuda =
    "[KeOps] This KeOps shared object has been compiled without cuda support: \n 1) to perform computations on CPU, simply set tagHostDevice to 0\n 2) to perform computations on GPU, please recompile the formula with a working version of cuda.";

void check_tag(int tag, std::string msg);

void check_nargs(int nargs);


}