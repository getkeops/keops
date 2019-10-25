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

namespace keops_binders {



using namespace keops;
namespace py = pybind11;

/////////////////////////////////////////////////////////////////////////////////
//                    Main function
/////////////////////////////////////////////////////////////////////////////////

template < typename array_t, typename index_t >
array_t generic_red(int tagCpuGpu,        // tagCpuGpu=0     means Reduction on Cpu, tagCpuGpu=1       means Reduction on Gpu, tagCpuGpu=2 means Reduction on Gpu from device data
                    int tag1D2D,          // tag1D2D=0       means 1D Gpu scheme,      tag1D2D=1       means 2D Gpu scheme
                    int tagHostDevice,    // tagHostDevice=1 means _fromDevice suffix. tagHostDevice=0 means _fromHost suffix
                    int Device_Id,        // id of GPU device
                    py::tuple ranges={},  // () if no "sparsity" ranges are given (default behavior)
                                          // Otherwise, ranges is a 6-uple of (integer) array_t
                                          // ranges = (ranges_i, slices_i, redranges_j, ranges_j, slices_j, redranges_i)
                                          // as documented in the doc on sparstiy and clustering.
                    py::tuple categories={},
                    py::tuple dimensions={},
                    py::args py_args={}) {

  // Check that we have enough arguments:
  size_t nargs = py_args.size();
  check_narg(nargs);

  check_tag(tag1D2D, "1D2D");
  check_tag(tagCpuGpu, "CpuGpu");
  check_tag(tagHostDevice, "HostDevice");

  short int Device_Id_s = cast_Device_Id(Device_Id);

  // Cast the input variable : It may be a copy here...
  std::vector <array_t> obj_ptr(py_args.size());
  for (size_t i = 0; i < py_args.size(); i++)
    obj_ptr[i] = py::cast< array_t >(py_args[i]);
  // If torch.h is included, the last 3 lines could be replaced by : auto obj_ptr = py::cast<std::vector<array_t>>(py_args);

  // get the pointers to data to avoid a copy
  __TYPE__ **castedargs = new __TYPE__ *[NARGS];
  for (size_t i = 0; i < NARGS; i++)
    castedargs[i] = get_data(obj_ptr[i]);

  // Cast the input signature:
  std::vector< int > cats(categories.size());
  for (size_t i = 0; i < categories.size(); i++)
    cats[i] = py::cast< int >(categories[i]);

  std::vector< int > dims(dimensions.size());
  for (size_t i = 0; i < dimensions.size(); i++)
    dims[i] = py::cast< int >(dimensions[i]);

  if ((categories.size() != nargs) or (dimensions.size() != nargs)) {
    throw std::runtime_error(
        "[KeOps] The lengths of the 'categories', 'dimensions' and '*args' tuples mismatch."
    );
  }

  // Check the aguments' dimensions, and retrieve all the shape information:
  std::tuple< int, int, int, int* > nx_ny_nbatch_shapes = check_ranges< array_t >(nargs, cats, dims, obj_ptr);
  int nx = std::get< 0 >(nx_ny_nbatch_shapes), ny = std::get< 1 >(nx_ny_nbatch_shapes);
  int nbatchdims = std::get< 2 >(nx_ny_nbatch_shapes);
  int *shapes = std::get< 3 >(nx_ny_nbatch_shapes);

  int tagRanges, nranges_x, nranges_y, nredranges_x, nredranges_y;
  __INDEX__ **castedranges;
  // N.B.: This vector is only used if ranges.size() == 6,
  //       but should *absolutely* be declared in all cases.
  //       Otherwise, a silent error (not detected by the compiler) provokes
  //       a silent buffer re-allocation, with a random corruption (in some edge cases)
  //       of the "ranges" arrays and, eventually, a segmentation fault.
  std::vector< index_t > ranges_ptr(ranges.size());

  // Sparsity: should we handle ranges? ======================================

  if (nbatchdims == 0) {  // Standard M-by-N computation
    if (ranges.size() == 0) {
      tagRanges = 0;

    } else if (ranges.size() == 6) {
      // Cast the six integer arrays
      for (size_t i = 0; i < ranges.size(); i++)
        ranges_ptr[i] = py::cast< index_t >(ranges[i]);

      // get the pointers to data to avoid a copy
      castedranges = new __INDEX__ *[ranges.size()];
      for (size_t i = 0; i < ranges.size(); i++)
        castedranges[i] = get_rangedata(ranges_ptr[i]);

      tagRanges = 1;
      nranges_x = get_size(ranges_ptr[0], 0);
      nranges_y = get_size(ranges_ptr[3], 0);

      nredranges_x = get_size(ranges_ptr[5], 0);
      nredranges_y = get_size(ranges_ptr[2], 0);
    } else {
      throw std::runtime_error(
          "[KeOps] the 'ranges' argument should be a tuple of size 0 or 6, "
          "but is of size " + std::to_string(ranges.size()) + "."
      );
    }



  } else if (ranges.size() == 0) {
    // Batch processing: we'll have to generate a custom, block-diagonal sparsity pattern
    tagRanges = 1;  // Batch processing is emulated through the block-sparse mode

    // We compute/read the number and size of our diagonal blocks ----------
    int nbatches = 1;
    for (int b = 0; b < nbatchdims; b++) {
      nbatches *= shapes[b];  // Compute the product of all "batch dimensions"
    }
    int M = shapes[nbatchdims], N = shapes[nbatchdims + 1];

    // Create new "castedranges" from scratch ------------------------------
    // With pythonic notations, we'll have:
    //   castedranges = (ranges_i, slices_i, redranges_j,   ranges_j, slices_j, redranges_i)
    // with:
    // - ranges_i    = redranges_i = [ [0,M], [M,2M], ..., [(nbatches-1)M, nbatches*M] ]
    // - slices_i    = slices_j    = [    1,     2,   ...,   nbatches-1,   nbatches    ]
    // - redranges_j = ranges_j    = [ [0,N], [N,2N], ..., [(nbatches-1)N, nbatches*N] ]

    castedranges = new __INDEX__ *[6];
    castedranges[0] = new __INDEX__[2 * nbatches];  // ranges_i
    castedranges[1] = new __INDEX__[nbatches];    // slices_i
    castedranges[2] = new __INDEX__[2 * nbatches];  // redranges_j
    castedranges[3] = castedranges[2];            // ranges_j
    castedranges[4] = castedranges[1];            // slices_j
    castedranges[5] = castedranges[0];            // redranges_i

    for (int b = 0; b < nbatches; b++) {
      castedranges[0][2 * b] = b * M;
      castedranges[0][2 * b + 1] = (b + 1) * M;
      castedranges[1][b] = b + 1;
      castedranges[2][2 * b] = b * N;
      castedranges[2][2 * b + 1] = (b + 1) * N;
    }

    nranges_x = nbatches;
    nredranges_x = nbatches;
    nranges_y = nbatches;
    nredranges_y = nbatches;

  } else {
    throw std::runtime_error(
        "[KeOps] The 'ranges' argument (block-sparse mode) is not supported with batch processing, "
        "but we detected " + std::to_string(nbatchdims) + " > 0 batch dimensions."
    );
  }

/*
  // Store, in a raw int array, the shape of the output: =====================
  // [A, .., B, M, D]  if TAGIJ==0
  //  or
  // [A, .., B, N, D]  if TAGIJ==1

  int *shape_output = new int[nbatchdims + 2];
  for (int b = 0; b < nbatchdims; b++) {
    shape_output[b] = shapes[b];                               // Copy the "batch dimensions"
  }
  shape_output[nbatchdims] = shapes[nbatchdims + TAGIJ];      // M or N
  shape_output[nbatchdims + 1] = shapes[nbatchdims + 2];      // D
*/

  int* shape_output = get_output_shape(shapes, nbatchdims);

  // Call Cuda codes =========================================================
  array_t result = allocate_result_array< array_t >(shape_output, tagCpuGpu);

  if (tagRanges == 1) {
    /*result = launch_keops_ranges< array_t >(tag1D2D, tagCpuGpu, tagHostDevice,
                                            Device_Id_s,
                                            nx, ny,
                                            nbatchdims, shapes, shape_output,
                                            nranges_x, nranges_y,
                                            nredranges_x, nredranges_y,
                                            castedranges,
                                            castedargs);*/
  } else {
    launch_keops < array_t > (tag1D2D, tagCpuGpu, tagHostDevice, Device_Id_s,
                                     nx, ny,
        , get_data(result),
                                     castedargs);
  }

  // Free the allocated memory, return our output array ======================
  if (nbatchdims != 0) {
    delete[] castedranges[0];  // ranges_i = redranges_i
    delete[] castedranges[1];  // slices_i = slices_j
    delete[] castedranges[2];  // redranges_j = ranges_j
  }

  delete[] castedargs;
  if (tagRanges)
     delete[] castedranges;
  delete[] shapes;
  delete[] shape_output;

  return result;

}

}
