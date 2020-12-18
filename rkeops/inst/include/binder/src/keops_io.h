#ifndef RKEOPS_IO_H
#define RKEOPS_IO_H

#include <string>
#include <stdexcept>
#include <vector>

#include "keops/binders/include.h"

#include "rkeops_data_type.h"
#include "rkeops_matrix.h"

// rkeops matrix type
using rkeops_base_matrix_t = rkeops::base_matrix< rkeops::type_t >;
using rkeops_matrix_t = rkeops::matrix< rkeops::type_t >;
// list of raw input data
using rkeops_list_t = std::vector< rkeops_base_matrix_t >;

namespace keops_binders {

void keops_error(std::basic_string< char > msg) {
    throw std::runtime_error(msg);
}

////////////////////////////////////////////////////////////////////////////////
//                  Template specialization (shell Matrix)                    //
////////////////////////////////////////////////////////////////////////////////

using __TYPE_BASE_ARRAY__ = rkeops::base_matrix< __TYPE__ >;
using __TYPEARRAY__ = rkeops::matrix< __TYPE__ >;

template <>
int get_ndim< __TYPE_BASE_ARRAY__ >(__TYPE_BASE_ARRAY__ obj_ptri) {
    return(obj_ptri.get_ndim());
}

template <>
int get_size< __TYPE_BASE_ARRAY__ >(__TYPE_BASE_ARRAY__ obj_ptri, int l) {
    return(obj_ptri.get_size(l));
}

template <>
__TYPE__* get_data< __TYPE_BASE_ARRAY__, __TYPE__ >(__TYPE_BASE_ARRAY__ obj_ptri) {
    return( (__TYPE__*) obj_ptri.get_data());
}

template <>
bool is_contiguous< __TYPE_BASE_ARRAY__ >(__TYPE_BASE_ARRAY__ obj_ptri) {
    return(obj_ptri.is_contiguous());
}

template <>
__TYPE_BASE_ARRAY__ allocate_result_array< __TYPE_BASE_ARRAY__, __TYPE__ >(
        int* shape_out, int nbatchdims) {
    // Create a new result array of shape [A, .., B, M, D] or [A, .., B, N, D]:
    std::vector< int > shape_vector(shape_out, shape_out + nbatchdims + 2);
    // assume 2d array = matrix
    return __TYPE_BASE_ARRAY__(shape_vector[1], shape_vector[0]);
}

template<>
__INDEX__* get_rangedata< __TYPE_BASE_ARRAY__ >(__TYPE_BASE_ARRAY__ obj_ptri) {
    return NULL;
}

template <>
__TYPE_BASE_ARRAY__ allocate_result_array_gpu< __TYPE_BASE_ARRAY__, __TYPE__ >(
        int* shape_out, int nbatchdims, short int device_id) {
    throw std::runtime_error("[rkeops] does not yet support array on GPU.");
}

template <>
int get_ndim< __TYPEARRAY__ >(__TYPEARRAY__ obj_ptri) {
    return(obj_ptri.get_ndim());
}

template <>
int get_size< __TYPEARRAY__ >(__TYPEARRAY__ obj_ptri, int l) {
    return(obj_ptri.get_size(l));
}

template <>
__TYPE__* get_data< __TYPEARRAY__, __TYPE__ >(__TYPEARRAY__ obj_ptri) {
    return( (__TYPE__*) obj_ptri.get_data());
}

template <>
bool is_contiguous< __TYPEARRAY__ >(__TYPEARRAY__ obj_ptri) {
    return(obj_ptri.is_contiguous());
}

template <>
__TYPEARRAY__ allocate_result_array< __TYPEARRAY__, __TYPE__ >(
        int* shape_out, int nbatchdims) {
    // assume 2d array = matrix
    __TYPEARRAY__ out(shape_out[0], shape_out[1]);
    out.update_data();
    return out;
}

template<>
__INDEX__* get_rangedata< __TYPEARRAY__ >(__TYPEARRAY__ obj_ptri) {
    return NULL;
}

template <>
__TYPEARRAY__ allocate_result_array_gpu< __TYPEARRAY__, __TYPE__ >(
        int* shape_out, int nbatchdims, short int device_id) {
    throw std::runtime_error("[rkeops] does not yet support array on GPU.");
}

}

////////////////////////////////////////////////////////////////////////////////
//                            Main function                                   //
////////////////////////////////////////////////////////////////////////////////

namespace rkeops {

template< typename array_t_in, typename array_t_out, typename index_t >
array_t_out generic_red(
        int tagCpuGpu,
        int tag1D2D,
        int tagHostDevice,
        int Device_Id,
		int nx,
		int ny,
        rkeops_list_t & input) {
    
    // number of arguments
    size_t nargs = input.size();
    
    // Call Cuda codes =========================================================
    array_t_out result = keops_binders::launch_keops< array_t_in, array_t_out, index_t >(
        tag1D2D,
        tagCpuGpu,
        tagHostDevice,
        Device_Id,
        nx,
        ny,
        nargs,
        input.data(),
        0, {});

    return result;
}

}

#endif // RKEOPS_IO_H
