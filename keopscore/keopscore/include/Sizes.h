#pragma once

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) < (b)) ? (b) : (a))
#define MAX3(a, b, c) (MAX(MAX(a, b), c))

#define do_checks 0
#if do_checks
void error(std::string message) { throw std::runtime_error(message); }
#endif

#if C_CONTIGUOUS

int get_val_batch(std::vector<signed long int> _shape, int nbatch, int b) {
  return _shape[b];
}

#else

int get_val_batch(std::vector<signed long int> _shape, int nbatch, int b) {
  return _shape[nbatch - b];
}

#endif

template <typename TYPE> class Sizes {
public:
  // attributs
  int nargs;
  signed long int nx, ny;
  signed long int M, N;
  int nbatchdims;
  int nbatches;

  std::vector<signed long int> _shapes;
  signed long int *shapes;
  std::vector<signed long int> _shape_out;
  signed long int *shape_out;

  int tagIJ;
  int use_half;
  std::vector<int> indsI;
  std::vector<int> indsJ;
  std::vector<int> indsP;
  int pos_first_argI;
  int pos_first_argJ;
  signed long int dimout;
  int nminargs;
  int nvarsI;
  int nvarsJ;
  int nvarsP;
  std::vector<signed long int> dimsX;
  std::vector<signed long int> dimsY;
  std::vector<signed long int> dimsP;

  // constructors
  Sizes(int _nargs, TYPE **args,
        const std::vector<std::vector<signed long int>> &argshapes, signed long int _nx,
        signed long int _ny, int tagIJ_, int use_half_, signed long int dimout_,
        const std::vector<int> &indsI_, std::vector<int> indsJ_,
        const std::vector<int> &indsP_, const std::vector<signed long int> &dimsX_,
        std::vector<signed long int> dimsY_, const std::vector<signed long int> &dimsP_) {

    tagIJ = tagIJ_;
    use_half = use_half_;
    indsI = indsI_;
    indsJ = indsJ_;
    indsP = indsP_;
    dimout = dimout_;

    nvarsI = indsI.size();
    nvarsJ = indsJ.size();
    nvarsP = indsP.size();

    pos_first_argI =
        (nvarsI > 0) ? *std::min_element(indsI.begin(), indsI.end()) : -1;
    pos_first_argJ =
        (nvarsJ > 0) ? *std::min_element(indsJ.begin(), indsJ.end()) : -1;

    int max_i =
        (nvarsI > 0) ? *std::max_element(indsI.begin(), indsI.end()) : -1;
    int max_j =
        (nvarsJ > 0) ? *std::max_element(indsJ.begin(), indsJ.end()) : -1;
    int max_p =
        (nvarsP > 0) ? *std::max_element(indsP.begin(), indsP.end()) : -1;

    nminargs = 1 + MAX3(max_i, max_j, max_p);
    dimsX = dimsX_;
    dimsY = dimsY_;
    dimsP = dimsP_;
    nargs = _nargs;
    nx = _nx;
    ny = _ny;

    // fill _shapes with "batch dimensions" [A, .., B], the table will look like:
    //
    // [ A, .., B, M, N, D_out]  -> output
    // [ A, .., B, M, 1, D_1  ]  -> "i" variable
    // [ A, .., B, 1, N, D_2  ]  -> "j" variable
    // [ A, .., B, 1, 1, D_3  ]  -> "parameter"
    // [ A, .., 1, M, 1, D_4  ]  -> N.B.: we support broadcasting on the batch dimensions
    // [ 1, .., 1, M, 1, D_5  ]  -> (we'll just ask users to fill in the shapes 
    //                               with *explicit* ones)
    fill_shape(nargs, argshapes);

    check_ranges(argshapes);

    // fill shape_out
    _shape_out.resize(nbatchdims + 3);

#if C_CONTIGUOUS
    // Copy the first row "[A, .., B, M, N, D_out]" of _shapes into _shape_out
    std::copy(_shapes.begin(), _shapes.begin() + nbatchdims + 3,
              _shape_out.begin());
    // If this is a reduction over the "j" variables, tagIJ = 0,
    // we need to remove the "N" dimension from the output shape:
    // [A, .., B, M, N, D_out] -> [A, .., B, M, D_out]
    //
    // Otherwise, if this is a reduction over the "i" variables, tagIJ = 1,
    // we drop the "M" dimension.
    _shape_out.erase(_shape_out.begin() + nbatchdims + (1 - tagIJ));

#else
    std::reverse_copy(_shapes.begin(), _shapes.begin() + nbatchdims + 3,
                      _shape_out.begin()); // Copy the "batch dimensions"
    _shape_out.erase(_shape_out.begin() + 1 tagIJ);

#endif

    // fill nx and ny
    M = _shapes[nbatchdims];     // = M
    N = _shapes[nbatchdims + 1]; // = N

    // Compute the product of all "batch dimensions"
    nbatches = std::accumulate(_shapes.begin(), _shapes.begin() + nbatchdims, 1,
                               std::multiplies<signed long int>());

    nx = nbatches * M; // = A * ... * B * M
    ny = nbatches * N; // = A * ... * B * N

    // The KeOps numerical routines expect pointers to raw integer arrays, 
    // not the clean std::vector that we've been using so far.
    shapes = &_shapes[0];
    shape_out = &_shape_out[0];
  }

  // methods

  void switch_to_half2_indexing();

private:
  void fill_shape(const int nargs,
                  const std::vector<std::vector<signed long int>> &argshapes);

  void check_broadcasting_batch_dimensions(
    const std::vector<std::vector<signed long int>> &argshapes,
    int off_i,
    int i);

  void check_ranges(const std::vector<std::vector<signed long int>> &argshapes);

  int MN_pos, D_pos;
};

template <typename TYPE>
void Sizes<TYPE>::fill_shape(
    const int nargs, const std::vector<std::vector<signed long int>> &argshapes) {
  // This function starts filling the _shapes array with the shapes of the output and all
  // the input variables, following the conventions described in the comments a few lines
  // above.

  int pos = std::max(pos_first_argI, pos_first_argJ);

  if (pos > -1) {
    // Are we working in batch mode? Infer the answer from the first arg
    // =============
    nbatchdims =
        argshapes[pos].size() - 2; // number of batch dimensions = Number of
                                   // dims of the first tensor - 2

    if (nbatchdims < 0) {
#if do_checks
      error("[KeOps] Wrong number of dimensions for arg at position 0: is " +
            std::to_string(argshapes[0].size()) + " but should be at least 2.");
#endif
    }
  } else {
    nbatchdims = 0;
  }

#if C_CONTIGUOUS
  // The arguments' shapes read something like [A, .., B, "M or N", D]
  MN_pos = nbatchdims;  // Position of the "M" or "N" dimension
  D_pos = nbatchdims + 1;  // Position of the "D" dimension
#else
  D_pos = 0;
  MN_pos = 1;
#endif

  // Now, we'll keep track of the output + all arguments' shapes in a large
  // array:
  // _shapes = ones(1 + nargs, nbatchdims + 3), with the first row describing the output.
  _shapes.resize((nargs + 1) * (nbatchdims + 3), 1);

  // Put "M, N, D_out" at the end of the first row of _shapes
  if (use_half) {
    if (tagIJ == 0) {
      _shapes[nbatchdims] = nx % 2 ? nx + 1 : nx;
      _shapes[nbatchdims + 1] = 2 * ny;
    } else {
      _shapes[nbatchdims] = 2 * nx;
      _shapes[nbatchdims + 1] = ny % 2 ? ny + 1 : ny;
    }
  } else {
    _shapes[nbatchdims] = nx;
    _shapes[nbatchdims + 1] = ny;
  }

  _shapes[nbatchdims + 2] = dimout; // Top right corner: dimension of the output

  // See check_ranges below for the rest of the filling process
}


template <typename TYPE>
void Sizes<TYPE>::check_broadcasting_batch_dimensions(
  const std::vector<std::vector<signed long int>> &argshapes,
  int off_i,
  int i) {
  // First, the batch dimensions:
  for (int b = 0; b < nbatchdims; b++) {
    // if C_CONTIGUOUS, we read the batch dimensions from the left:
    // this is just argshapes[i][b]
    _shapes[off_i + b] = get_val_batch(argshapes[i], nbatchdims + 2, b);

    // Check that the current value is compatible with what
    // we've encountered so far, as stored in the first line of "shapes"
    if (_shapes[off_i + b] != 1) { // This dimension is not "broadcasted"
      if (_shapes[b] == 1) {
        _shapes[b] = _shapes[off_i + b]; // -> it becomes the new standard
      }
#if do_checks
      else if (_shapes[b] != _shapes[off_i + b]) {
        error("[KeOps] Wrong value of the batch dimension " +
              std::to_string(b) + " for argument number " +
              std::to_string(i) + " : is " +
              std::to_string(_shapes[off_i + b]) + " but was " +
              std::to_string(_shapes[b]) + " or 1 in previous arguments.");
      }
#endif
    }
  }
}


template <typename TYPE>
void Sizes<TYPE>::check_ranges(
    const std::vector<std::vector<signed long int>> &argshapes) {

  // Check the compatibility of all tensor shapes
  // ==================================
  if (nminargs > 0) {

    // Checks args in all the positions that correspond to "i" variables:
    for (int k = 0; k < nvarsI; k++) {
      int i = indsI[k];

      // Fill in the (i+1)-th line of the "shapes" array
      // ---------------------------
      int off_i = (i + 1) * (nbatchdims + 3);

      // Check the number of dimensions
      // --------------------------------------------
      int ndims = argshapes[i].size(); // Number of dims of the i-th tensor
      // We expect argshapes[i] to be a vector that reads something like
      // [A, ..., B, M, D_in]
      // with batch dimensions that are compatible with broadcast
      // and a D_in which is compatible with its usage in the required KeOps formula.

#if do_checks
      if (ndims != nbatchdims + 2) {
        error(
            "[KeOps] Wrong number of dimensions for arg at position " +
            std::to_string(i) + " (i type): KeOps detected " +
            std::to_string(nbatchdims) +
            " batch dimensions from the first argument 0, and thus expected " +
            std::to_string(nbatchdims + 2) +
            " dimensions here, but only received " + std::to_string(ndims) +
            ". Note that KeOps supports broadcasting on batch dimensions, " +
            "but still expects 'dummy' unit dimensions in the input shapes, " +
            "for the sake of clarity.");
      }
#endif

      check_broadcasting_batch_dimensions(argshapes, off_i, i);

      _shapes[off_i + nbatchdims] = argshapes[i][MN_pos];    // = "M"
      // _shapes[off_i + nbatchdims + 1] remains equal to 1
      _shapes[off_i + nbatchdims + 2] = argshapes[i][D_pos]; // = "D"

#if do_checks
      // Check the number of "lines", the index "M":
      if (_shapes[nbatchdims] != _shapes[off_i + nbatchdims]) {
        error("[KeOps] Wrong value of the 'i' dimension " +
              std::to_string(nbatchdims) + "for arg at position " +
              std::to_string(i) + " : is " +
              std::to_string(_shapes[off_i + nbatchdims]) + " but was " +
              std::to_string(_shapes[nbatchdims]) +
              " in previous 'i' arguments.");
      }

      // And the number of "columns", the dimension "D":
      if (_shapes[off_i + nbatchdims + 2] != static_cast<int>(dimsX[k])) {
        error("[KeOps] Wrong value of the 'vector size' dimension " +
              std::to_string(nbatchdims + 1) + " for arg at position " +
              std::to_string(i) + " : is " +
              std::to_string(_shapes[off_i + nbatchdims + 2]) +
              " but should be " + std::to_string(dimsX[k]));
      }
#endif
    }

    // Checks args in all the positions that correspond to "j" variables:
    for (int k = 0; k < nvarsJ; k++) {
      int i = indsJ[k];

      // Check the number of dimensions
      // --------------------------------------------
      int ndims = argshapes[i].size(); // Number of dims of the i-th tensor

#if do_checks
      if (ndims != nbatchdims + 2) {
        error(
            "[KeOps] Wrong number of dimensions for arg at position " +
            std::to_string(i) + " (j type): KeOps detected " +
            std::to_string(nbatchdims) +
            " batch dimensions from the first argument 0, and thus expected " +
            std::to_string(nbatchdims + 2) +
            " dimensions here, but only received " + std::to_string(ndims) +
            ". Note that KeOps supports broadcasting on batch dimensions, " +
            "but still expects 'dummy' unit dimensions in the input shapes, " +
            "for the sake of clarity.");
      }
#endif

      // Fill in the (i+1)-th line of the "shapes" array
      // ---------------------------
      int off_i = (i + 1) * (nbatchdims + 3);

      check_broadcasting_batch_dimensions(argshapes, off_i, i);

      // _shapes[off_i + nbatchdims] remains equal to 1
      _shapes[off_i + nbatchdims + 1] = argshapes[i][MN_pos]; // = "N"
      _shapes[off_i + nbatchdims + 2] = argshapes[i][D_pos];  // = "D"

#if do_checks
      // Check the number of "lines", the index "N":
      if (_shapes[nbatchdims + 1] != _shapes[off_i + nbatchdims + 1]) {
        error("[KeOps] Wrong value of the 'j' dimension " +
              std::to_string(nbatchdims) + " for arg at position " +
              std::to_string(i) + " : is " +
              std::to_string(_shapes[off_i + nbatchdims + 1]) + " but was " +
              std::to_string(_shapes[nbatchdims + 1]) +
              " in previous 'j' arguments.");
      }

      // And the number of "columns", the dimension "D":
      if (_shapes[off_i + nbatchdims + 2] != static_cast<int>(dimsY[k])) {
        error("[KeOps] Wrong value of the 'vector size' dimension " +
              std::to_string(nbatchdims + 1) + " for arg at position " +
              std::to_string(i) + " : is " +
              std::to_string(_shapes[off_i + nbatchdims + 2]) +
              " but should be " + std::to_string(dimsY[k]));
      }
#endif
    }

    // Finally, we check the parameters, i.e. variables that may vary with
    // batch dimensions, but not with the "i" or "j" indices:
    for (int k = 0; k < nvarsP; k++) {
      int i = indsP[k];
      // Fill in the (i+1)-th line of the "shapes" array
      // ---------------------------
      int off_i = (i + 1) * (nbatchdims + 3);
      
      check_broadcasting_batch_dimensions(argshapes, off_i, i);

      // _shapes[off_i + nbatchdims] remains equal to 1
      // _shapes[off_i + nbatchdims + 1] remains equal to 1
      _shapes[off_i + nbatchdims + 2] = argshapes[i][nbatchdims]; // = "D"
#if do_checks
      signed long int dim_param;
      if (use_half)
        dim_param = _shapes[off_i + nbatchdims + 2] / 2;
      else
        dim_param = _shapes[off_i + nbatchdims + 2];
      if (dim_param != static_cast<int>(dimsP[k])) {
        error("[KeOps] Wrong value of the 'vector size' dimension " +
              std::to_string(nbatchdims) + " for arg at position " +
              std::to_string(i) + " : is " + std::to_string(dim_param) +
              " but should be " + std::to_string(dimsP[k]));
      }
#endif
    }
  }
}

template <typename TYPE> void Sizes<TYPE>::switch_to_half2_indexing() {
  // special case of float16 inputs : because we use half2 type in Cuda codes,
  // we need to divide by two nx, ny, and M, N, or D values inside the shapes
  // vector.
  nx = nx / 2;
  ny = ny / 2;
  M = M / 2;
  N = N / 2;
  _shapes[nbatchdims] = _shapes[nbatchdims] / 2;
  _shapes[nbatchdims + 1] = _shapes[nbatchdims + 1] / 2;
  for (int i = 0; i < nargs; i++) {
    int off_i = (i + 1) * (nbatchdims + 3);
    // we don't have anymore the category information...
    // the last three dimensions are either of the form (M,1,D), (1,N,D), or
    // (1,1,D) where M or N are even in the 2 first cases, or D is even in the
    // third case.
    if (_shapes[off_i + nbatchdims] > 1)
      _shapes[off_i + nbatchdims] = _shapes[off_i + nbatchdims] / 2;
    else if (_shapes[off_i + nbatchdims + 1] > 1)
      _shapes[off_i + nbatchdims + 1] = _shapes[off_i + nbatchdims + 1] / 2;
    else
      _shapes[off_i + nbatchdims + 2] = _shapes[off_i + nbatchdims + 2] / 2;
  }
}
