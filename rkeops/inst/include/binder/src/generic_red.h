#include <vector>

#include "rkeops_data_type.h"
#include "rkeops_matrix.h"

// rkeops matrix type
using rkeops_matrix_t = rkeops::matrix<rkeops::type_t>;
// list of input data
using rkeops_list_t = std::vector< rkeops_matrix_t >;

// generic reduction
rkeops_matrix_t genred(
        int tagCpuGpu, int tag1D2D, int tagHostDevice, 
        int Device_Id, int nx, int ny,
        rkeops_list_t & input);
