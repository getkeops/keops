#include <vector>
#include <string>

#include <torch/torch.h>
#include <pybind11/pybind11.h>

#include "common/gridcluster_io.h"

namespace pykeops {

//using namespace keops;
namespace py = pybind11;

/////////////////////////////////////////////////////////////////////////////////
//                             Utils
/////////////////////////////////////////////////////////////////////////////////

template <>
int get_size(at::Tensor obj_ptri, int l){
    return obj_ptri.size(l);
}

template <>
__TYPE__* get_data(at::Tensor obj_ptri){
    return obj_ptri.data<__TYPE__>();
}

template <>
__INDEX__* get_rangedata(at::Tensor obj_ptri){
    return obj_ptri.data<__INDEX__>();
}

template <>
bool is_contiguous(at::Tensor obj_ptri){
    return obj_ptri.is_contiguous();
}


#define AT_INDEX at::kInt // == Int32

/////////////////////////////////////////////////////////////////////////////////
//                          Main functions
/////////////////////////////////////////////////////////////////////////////////

template <>
at::Tensor launch_grid_label(int tagCpuGpu, int tagHostDevice, int Device_Id, 
                      int npoints, int dimpoints,
                      __TYPE__ *voxelsize,
                      __TYPE__ *points){

    if(tagHostDevice == 0) { // Data is located on Host
        throw std::runtime_error("[KeOps] CPU clustering has not been implemented yet.");
    } else if(tagHostDevice == 1) { // Data is on the device
    #if USE_CUDA       
        // The output will be on the device too
        at::Tensor result_array = at::empty({npoints,1}, {torch::CUDA(AT_INDEX),Device_Id});
        torch::set_requires_grad(result_array, false);

        GpuGridLabel_FromDevice(npoints, dimpoints, 
            get_rangedata(result_array), voxelsize, points, Device_Id);
        return result_array;
    #else
            throw std::runtime_error("[KeOps] No cuda device detected... try to set tagHostDevice to 0.");
    #endif
    }
    throw std::runtime_error("[KeOps] tagHostDevice should be equal to 0 (CPU) or 1 (GPU).");
}


/////////////////////////////////////////////////////////////////////////////////
//                    PyBind11 entry point
/////////////////////////////////////////////////////////////////////////////////
#define VALUE_OF(x) x

PYBIND11_MODULE(VALUE_OF(GCMODULE_NAME), m) {
    m.doc() = "VoxelGrid clustering through pybind11."; // optional module docstring

    m.def("grid_label",
          &grid_label<at::Tensor>,
          "Entry point to grid_label - pytorch version.");

}

}
