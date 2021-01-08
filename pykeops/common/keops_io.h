#include <vector>

#include "keops/binders/include.h"

namespace py = pybind11;

/////////////////////////////////////////////////////////////////////////////////
//                    Main function
/////////////////////////////////////////////////////////////////////////////////

template< typename array_t, typename index_t >
array_t generic_red(
        int tagCpuGpu,        // tagCpuGpu=0     means Reduction on Cpu, tagCpuGpu=1       means Reduction on Gpu, tagCpuGpu=2 means Reduction on Gpu from device data
        int tag1D2D,          // tag1D2D=0       means 1D Gpu scheme,      tag1D2D=1       means 2D Gpu scheme
        int tagHostDevice,    // tagHostDevice=1 means _fromDevice suffix. tagHostDevice=0 means _fromHost suffix
        int Device_Id,        // id of GPU device
        py::tuple py_ranges,  // () if no "sparsity" ranges are given (default behavior)
                              // Otherwise, ranges is a 6-uple of (integer) array_t
                              // ranges = (ranges_i, slices_i, redranges_j, ranges_j, slices_j, redranges_i)
                              // as documented in the doc on sparstiy and clustering.
		int nx,				  // number of samples / data points of the "i" indexed variables
		int ny,				  // number of samples / data points of the "j" indexed variables
        py::args py_args) {
  
//////////////////////////////////////////////////////////////
// Input arguments                                          //
//////////////////////////////////////////////////////////////
  
  // get the number of args
  int nargs = py_args.size();
  
  // Cast the input variable : It may be a copy here...
  // If torch.h is included, the next 3 lines could be replaced by :
  // auto args = py::cast<std::vector<array_t>>(py_args);
  std::vector< array_t > args(nargs);
  for (int i = 0; i < nargs; i++)
    args[i] = py::cast< array_t >(py_args[i]);

  // get the number of ranges
  int nranges = py_ranges.size();

  // Cast the ranges arrays
  std::vector< index_t > ranges(nranges);
  for (int i = 0; i < nranges; i++)
    ranges[i] = py::cast< index_t >(py_ranges[i]);

//////////////////////////////////////////////////////////////
// Call Cuda codes                                          //
//////////////////////////////////////////////////////////////
  py::gil_scoped_release release;
  array_t result = keops_binders::launch_keops< array_t, array_t, index_t >
          (tag1D2D,
           tagCpuGpu,
           tagHostDevice,
           Device_Id,
		   nx,
		   ny,
           nargs,
           &args[0],
           nranges,
           &ranges[0]);
  py::gil_scoped_acquire acquire;
  return result;
}

