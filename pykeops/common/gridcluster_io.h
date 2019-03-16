#include <vector>
#include <string>

#define __INDEX__ int32_t // use int instead of double

#if USE_CUDA
extern "C" {
    int GpuGridLabel_FromDevice(int, int, __INDEX__*, __TYPE__*, __TYPE__*, int);
};
#endif


namespace pykeops {

//using namespace keops;
namespace py = pybind11;

/////////////////////////////////////////////////////////////////////////////////
//                             Utils
/////////////////////////////////////////////////////////////////////////////////

template< typename array_t >
int get_size(array_t obj_ptri, int l);

template< typename array_t >
__TYPE__* get_data(array_t obj_ptri);

template< typename array_t >
__INDEX__* get_rangedata(array_t obj_ptri);

template< typename array_t >
bool is_contiguous(array_t obj_ptri);


/////////////////////////////////////////////////////////////////////////////////
//                    Sanity checks on args
/////////////////////////////////////////////////////////////////////////////////


void check_tag(int tag, std::string msg){
    if ((tag < 0) || (tag > 1)) {
        throw std::runtime_error("[Keops] tag" + msg + " should be (0 or 1) but is " + std::to_string(tag));
    }
}


/////////////////////////////////////////////////////////////////////////////////
//                    Call Cuda functions
/////////////////////////////////////////////////////////////////////////////////

// Implemented by pykeops/torch/cluster/grid_cluster.cpp
//             or pykeops/numpy/cluster/grid_cluster.cpp
template < typename array_t >
array_t launch_grid_label(int tagCpuGpu, int tagHostDevice, int Device_Id, 
                      int npoints, int dimpoints,
                      __TYPE__ *voxelsize,
                      __TYPE__ *points);


/////////////////////////////////////////////////////////////////////////////////
//                    Main function
/////////////////////////////////////////////////////////////////////////////////

template < typename array_t >
array_t grid_label(int tagCpuGpu,        // tagCpuGpu=0     means Reduction on Cpu, tagCpuGpu=1
                   int tagHostDevice,    // tagHostDevice=1 means _fromDevice suffix. tagHostDevice=0 means _fromHost suffix
                   int Device_Id,        // id of GPU device
                   py::args py_args) {   // == (&voxelsize, &points)

    check_tag(tagCpuGpu, "CpuGpu");
    check_tag(tagHostDevice, "HostDevice");

    // Cast the input variable : It may be a copy here...
    std::vector<array_t> obj_ptr(py_args.size());
    for (size_t i = 0; i < py_args.size(); i++)
        obj_ptr[i] = py::cast<array_t> (py_args[i]);
    // If torch.h is included, the last 3 lines could be replaced by : auto obj_ptr = py::cast<std::vector<array_t>>(py_args);

    // get the pointers to data to avoid a copy
    __TYPE__ *voxelsize = get_data(obj_ptr[0]);
    __TYPE__ *points    = get_data(obj_ptr[1]);

    int npoints   = get_size(obj_ptr[0],0);
    int dimpoints = get_size(obj_ptr[1],1) ;

    // Sanity checks
    /*if (!is_contiguous(voxelsize)) {
        throw std::runtime_error("[Keops] The array of voxelsizes is not contiguous. Please provide 'contiguous' dara array, as KeOps does not support strides. ") ;
    }
    if (!is_contiguous(points)) {
        throw std::runtime_error("[Keops] The points array is not contiguous. Please provide 'contiguous' dara array, as KeOps does not support strides. ") ;
    }*/

    // Call Cuda codes
    array_t result = launch_grid_label<array_t>(tagCpuGpu, tagHostDevice, Device_Id,
                            npoints, dimpoints,
                            voxelsize, points );

    return result;

}


}
