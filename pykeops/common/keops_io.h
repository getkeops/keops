#include <vector>
#include <string>
#include <type_traits>
#include <limits>
#include <stdexcept>
#include <tuple>

// #include "formula.h" done by cmake
#include "binders/checks.h"
#include "binders/utils.h"
#include "binders/switch.h"


using namespace keops;
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
        py::tuple ranges,  // () if no "sparsity" ranges are given (default behavior)
        // Otherwise, ranges is a 6-uple of (integer) array_t
        // ranges = (ranges_i, slices_i, redranges_j, ranges_j, slices_j, redranges_i)
        // as documented in the doc on sparstiy and clustering.
        py::args py_args) {
  
  // Check that we have enough arguments:
  int nargs = py_args.size();
  
  // Cast the input variable : It may be a copy here...
  std::vector< array_t > obj_ptr(py_args.size());
  for (size_t i = 0; i < py_args.size(); i++)
    obj_ptr[i] = py::cast< array_t >(py_args[i]);
  // If torch.h is included, the last 3 lines could be replaced by : auto obj_ptr = py::cast<std::vector<array_t>>(py_args);
  
  // Cast the six integer arrays
  std::vector< index_t > ranges_ptr(ranges.size());
  for (size_t i = 0; i < ranges.size(); i++)
    ranges_ptr[i] = py::cast< index_t >(ranges[i]);
  
  // get the pointers to data to avoid a copy
  __INDEX__* castedranges[ranges.size()];
  for (size_t i = 0; i < ranges.size(); i++)
    castedranges[i] = keops_binders::get_rangedata(ranges_ptr[i]);
  
  
  // Call Cuda codes =========================================================
  /*if (tagRanges == 1) {
    keops_binders::launch_keops_ranges(tag1D2D, tagCpuGpu, tagHostDevice,
                                            Device_Id_s,
                                            nx, ny,
                                         nbatchdims, shapes,
                                            nranges_x, nranges_y,
                                            nredranges_x, nredranges_y,
                                         keops_binders::get_data< array_t, __TYPE__ >(result),
                                            castedranges,
                                         
                                         castedargs);
  } else {*/
  array_t result = keops_binders::launch_keops< array_t >(tag1D2D,
                                                          tagCpuGpu,
                                                          tagHostDevice,
                                                          Device_Id,
                                                          nargs,
                                                          &obj_ptr[0]);
  
  /*} */
  
  return result;
}

