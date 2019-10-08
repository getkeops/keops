#ifndef RKEOPS_IO_H
#define RKEOPS_IO_H

#include <string>
#include <stdexcept>
#include <vector>

#include "binders/checks.h"
#include "binders/utils.h"
#include "binders/switch.h"
// #include "formula.h" made in cmake

#include "rkeops_data_type.h"
#include "rkeops_matrix.h"

// rkeops matrix type
using rkeops_matrix_t = rkeops::matrix<rkeops::type_t>;
// list of input data
using rkeops_list_t = std::vector<rkeops_matrix_t>;

namespace keops_binders {

void keops_error(std::basic_string< char > msg) {
    throw std::runtime_error(msg);
}

////////////////////////////////////////////////////////////////////////////////
//                  Template specialization (shell Matrix)                    //
////////////////////////////////////////////////////////////////////////////////

using __TYPEARRAY__ = rkeops::matrix<__TYPE__>;

template <>
int get_ndim(__TYPEARRAY__ &obj_ptri) {
    return(obj_ptri.get_ndim());
}

template <>
int get_size(__TYPEARRAY__ &obj_ptri, int l) {
    return(obj_ptri.get_size(l));
}

template <>
__TYPE__* get_data(__TYPEARRAY__ &obj_ptri) {
    return( (__TYPE__*) obj_ptri.get_data());
}

template <>
bool is_contiguous(__TYPEARRAY__ &obj_ptri) {
    return(obj_ptri.is_contiguous());
}

template <>
__TYPEARRAY__ allocate_result_array(int* shape_out, int nbatchdims) {
    // Create a new result array of shape [A, .., B, M, D] or [A, .., B, N, D]:
    std::vector< int > shape_vector(shape_out, shape_out + nbatchdims + 2);
    // assume 2d array = matrix
    return __TYPEARRAY__(shape_vector[1], shape_vector[0]);
}

#if USE_CUDA
template <>
__TYPEARRAY__ allocate_result_array_gpu(int* shape_out, int nbatchdims) {
    throw std::runtime_error("[rkeops] does not yet support array on GPU.");
}
#endif


// using namespace keops;

////////////////////////////////////////////////////////////////////////////////
//                            Main function                                   //
////////////////////////////////////////////////////////////////////////////////

template < typename array_t >
array_t generic_red(
        int nx, int ny,
        int tagCpuGpu,
        int tag1D2D,
        int tagHostDevice,
        int Device_Id,
        rkeops_list_t & input) {
    
    // Check that we have enough arguments:
    size_t nargs = input.size();
    check_narg(nargs);

    check_tag(tag1D2D, "1D2D");
    check_tag(tagCpuGpu, "CpuGpu");
    check_tag(tagHostDevice, "HostDevice");

    short int Device_Id_s = cast_Device_Id(Device_Id);
    
    // get the pointers to data to avoid a copy
    __TYPE__ **castedargs = new __TYPE__ *[keops::NARGS];
    for (size_t i = 0; i < keops::NARGS; i++)
        castedargs[i] = input[i].get_data();
    
    int shape_output[2] = {keops::TAGIJ ? ny : nx, keops::DIMOUT};
    
    // Call Cuda codes =========================================================
    array_t result = launch_keops< array_t >(
                                tag1D2D, tagCpuGpu, tagHostDevice,
                                Device_Id_s, nx, ny, shape_output,
                                castedargs);
    
    delete[] castedargs;
    return result;
}

}

#endif // RKEOPS_IO_H
