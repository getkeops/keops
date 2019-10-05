#include <vector>

#include "keops_io.h"
#include "rkeops_data_type.h"
#include "rkeops_matrix.h"

// rkeops matrix type
using rkeops_matrix_t = rkeops::matrix<rkeops::type_t>;
// list of input data
using rkeops_list_t = std::vector<rkeops_matrix_t>;

// generic reduction
rkeops_matrix_t genred(
        int tagCpuGpu, int tag1D2D, int tagHostDevice, 
        int Device_Id, int nx, int ny,
        rkeops_list_t & input) {
    
    rkeops_matrix_t output = keops_binders::generic_red<rkeops_matrix_t> (
        nx,
        ny,
        tagCpuGpu,
        tag1D2D,
        tagHostDevice,
        Device_Id,
        input);
    
    return(output);
}
