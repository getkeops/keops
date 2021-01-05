#include <vector>

#include "rkeops_data_type.h"
#include "rkeops_matrix.h"

// rkeops matrix type
using rkeops_base_matrix_t = rkeops::base_matrix< rkeops::type_t >;
using rkeops_matrix_t = rkeops::matrix< rkeops::type_t >;
// list of raw input data
using rkeops_list_t = std::vector< rkeops_base_matrix_t >;

// generic reduction
rkeops_matrix_t genred(
        int tagCpuGpu, int tag1D2D, int tagHostDevice, 
        int Device_Id, int nx, int ny,
        rkeops_list_t & input);
