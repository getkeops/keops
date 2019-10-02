#pragma once

#include <string>
#include <tuple>

namespace keops {
/////////////////////////////////////////////////////////////////////////////////
//                           Keops
/////////////////////////////////////////////////////////////////////////////////
using FF = F::F; // F::F is formula inside reduction (ex if F is SumReduction<Form> then F::F is Form)

using VARSI = typename FF::template VARS<0>;    // list variables of type I used in formula F
using VARSJ = typename FF::template VARS<1>;    // list variables of type J used in formula F
using VARSP = typename FF::template VARS<2>;    // list variables of type parameter used in formula F

using DIMSX = GetDims<VARSI>;
using DIMSY = GetDims<VARSJ>;
using DIMSP = GetDims<VARSP>;

using INDSI = GetInds<VARSI>;
using INDSJ = GetInds<VARSJ>;
using INDSP = GetInds<VARSP>;

using INDS = ConcatPacks<ConcatPacks<INDSI, INDSJ>, INDSP>;

const int NARGSI = VARSI::SIZE; // number of I variables used in formula F
const int NARGSJ = VARSJ::SIZE; // number of J variables used in formula F
const int NARGSP = VARSP::SIZE; // number of parameters variables used in formula F

const int NARGS = F::NMINARGS;
const int DIMOUT = F::DIM;

const int TAGIJ = F::tagI;

const std::string f =  PrintReduction<F>();


}


namespace keops_binders {

using namespace keops;

/////////////////////////////////////////////////////////////////////////////////
//                             Utils                                           //
/////////////////////////////////////////////////////////////////////////////////

// This function has to be specialized in the various binders:

template < typename array_t >
int get_ndim(array_t obj_ptri);  // len( a.shape )

template < typename array_t >
int get_size(array_t obj_ptri, int l);  // a.shape[l]

template < typename array_t >
__TYPE__ *get_data(array_t obj_ptri);   // raw pointer to "a.data"

template < typename array_t >
__INDEX__ *get_rangedata(array_t obj_ptri);  // raw pointer to "a.data", casted as integer

template < typename array_t >
bool is_contiguous(array_t obj_ptri);  // is "a" ordered properly? KeOps does *not* support strides!

void keops_error(std::string);


/////////////////////////////////////////////////////////////////////////////////
//                    Sanity checks on args
/////////////////////////////////////////////////////////////////////////////////

void check_narg(int narg) {
  if (narg < NARGS)
    keops_error("[KeOps] Wrong number of args : is " + std::to_string(narg)
              + " but should be at least " + std::to_string(NARGS)
              + " in " + f);
}

void check_tag(int tag, std::string msg) {
  if ((tag < 0) || (tag > 1)) {
    keops_error("[KeOps] tag" + msg + " should be (0 or 1) but is " + std::to_string(tag));
  }
}

template < typename array_t >
void check_args(size_t nargs,
                std::vector< int > categories,
                std::vector< int > dimensions,
                std::vector< array_t > obj_ptr) {

  check_narg(nargs);

  int *typeargs, *dimargs;

  if (NARGS > 0) {

    // Expected categories and dimensions, from the formula's signature =============

    typeargs = new int[NARGS];  // Expected categories
    dimargs = new int[NARGS];   // Expected dimenstions

    // Fill typeargs and dimargs with -1's:
    for (int k = 0; k < NARGS; k++) {
      typeargs[k] = -1;
      dimargs[k] = -1;
    }
    // Fill in all the positions that correspond to "i" variables:
    for (int k = 0; k < NARGSI; k++) {
      typeargs[INDSI::VAL(k)] = 0;
      dimargs[INDSI::VAL(k)] = DIMSX::VAL(k);
    }
    // Fill in all the positions that correspond to "j" variables:
    for (int k = 0; k < NARGSJ; k++) {
      typeargs[INDSJ::VAL(k)] = 1;
      dimargs[INDSJ::VAL(k)] = DIMSY::VAL(k);
    }
    // Fill in all the positions that correspond to "parameters":
    for (int k = 0; k < NARGSP; k++) {
      typeargs[INDSP::VAL(k)] = 2;
      dimargs[INDSP::VAL(k)] = DIMSP::VAL(k);
    }

    // Check vs. the user-given categories and dimensions ==================
    for (int k = 0; k < NARGS; k++) {
      if (typeargs[k] != categories[k]) {
        keops_error(
            "[KeOps] Wrong variable category (0 = Vi, 1 = Vj, 2 = Pm) at position " + std::to_string(k)
                + ": received " + std::to_string(categories[k]) +
                +" but expected " + std::to_string(typeargs[k]) + ".");
      }

      if (dimargs[k] != dimensions[k]) {
        keops_error("[KeOps] Wrong dimension for variable at position " + std::to_string(k)
                                     + ": received " + std::to_string(dimargs[k]) +
            +" but expected " + std::to_string(dimensions[k]) + ".");
      }
    }

    // Free the allocated memory (but *not* shapes) ========================
    delete[] dimargs;
    delete[] typeargs;
  }
}

template < typename array_t >
std::tuple< int, int, int, int * > check_ranges(std::vector < array_t > obj_ptr,
                                                std::vector< int > categories={},
                                                std::vector< int > dimensions={}) {

  size_t nargs = obj_ptr.size();

  // Are we working in batch mode? Infer the answer from the first arg =============
  int nbatchdims = get_ndim(obj_ptr[0]);  // Number of dims of the first tensor
  // Remove the "trailing" dim (.., D) if the first arg is a parameter,
  // or the last two (.., M/N, D) if it is an "i" or "j" variable:
  nbatchdims -= (categories[0] == 2) ? 1 : 2;

  if (nbatchdims < 0) {
    keops_error("[KeOps] Wrong number of dimensions for the first arg: is "
                                 + std::to_string(get_ndim(obj_ptr[0])) + " but should be at least "
                                 + std::to_string((categories[0] == 2) ? 1 : 2));
  }

  // Now, we'll keep track of the output + all arguments' shapes in a large array:
  int *shapes = new int[(nargs + 1) * (nbatchdims + 3)];
  // N.B.: shapes will be destroyed at the very end of generic_red
  // Eventually, with "batch dimensions" [A, .., B], the table will look like:
  //
  // [ A, .., B, M, N, D_out]  -> output
  // [ A, .., B, M, 1, D_1  ]  -> "i" variable
  // [ A, .., B, 1, N, D_2  ]  -> "j" variable
  // [ A, .., B, 1, 1, D_3  ]  -> "parameter"
  // [ A, .., 1, M, 1, D_4  ]  -> N.B.: we support broadcasting on the batch dimensions!
  // [ 1, .., 1, M, 1, D_5  ]  ->      (we'll just ask users to fill in the shapes with *explicit* ones)

  // Fill in the first line with
  // [ 1, ..., 1, -1, -1, D_out]
  for (int b = 0; b < nbatchdims; b++) {
    shapes[b] = 1;  // 1 = default option
  }
  shapes[nbatchdims] = -1;           // M is still unknown
  shapes[nbatchdims + 1] = -1;       // N is still unknown
  shapes[nbatchdims + 2] = DIMOUT;   // Top right corner: dimension of the output


  // Check the compatibility of all tensor shapes ==================================

  for (size_t i = 0; i < nargs; i++) {

    // Check the number of dimensions --------------------------------------------
    int ndims = get_ndim(obj_ptr[i]);  // Number of dims of the i-th tensor

    // N.B.: CAT=2 -> "Parameter" -> 1 extra dim ; otherwise, CAT=0 or 1 -> 2 extra dims
    if (ndims != nbatchdims + ((categories[i] == 2) ? 1 : 2)) {
      keops_error("[KeOps] Wrong number of dimensions for arg number " + std::to_string(i)
                                   + " : KeOps detected " + std::to_string(nbatchdims)
                                   + " batch dimensions from the first argument 0, and thus expected "
                                   + std::to_string(nbatchdims + ((categories[i] == 2) ? 1 : 2))
                                   + " dimenstions here, but only received "
                                   + std::to_string(ndims)
                                   + ". Note that KeOps supports broadcasting on batch dimensions, "
                                   + "but still expects 'dummy' unit dimensions in the input shapes, "
                                   + "for the sake of clarity.");
    }

    // Fill in the (i+1)-th line of the "shapes" array ---------------------------
    int off_i = (i + 1) * (nbatchdims + 3);

    // First, the batch dimensions:
    for (int b = 0; b < nbatchdims; b++) {
      shapes[off_i + b] = get_size(obj_ptr[i], b);

      // Check that the current value is compatible with what
      // we've encountered so far, as stored in the first line of "shapes"
      if (shapes[off_i + b] != 1) {  // This dimension is not "broadcasted"
        if (shapes[b] == 1) {
          shapes[b] = shapes[off_i + b];  // -> it becomes the new standard
        } else if (shapes[b] != shapes[off_i + b]) {
          keops_error("[KeOps] Wrong value of the batch dimension "
                                       + std::to_string(b) + " for argument number " + std::to_string(i)
                                       + " : is " + std::to_string(shapes[off_i + b])
                                       + " but was " + std::to_string(shapes[b])
                                       + " or 1 in previous arguments.");
        }
      }
    }

    // Then, the numbers "M", "N" and "D":
    if (categories[i] == 0) {  // "i" variable --------------------------------------
      shapes[off_i + nbatchdims] = get_size(obj_ptr[i], nbatchdims);  // = "M"
      if (shapes[nbatchdims] == -1) {  // This is the first "i" variable that we encounter
        shapes[nbatchdims] = shapes[off_i + nbatchdims];  // -> Fill in the "M" coefficient in the first line
      }

      shapes[off_i + nbatchdims + 1] = 1;
      shapes[off_i + nbatchdims + 2] = get_size(obj_ptr[i], nbatchdims + 1);  // = "D"

      // Check the number of "lines":
      if (shapes[nbatchdims] != shapes[off_i + nbatchdims]) {
        keops_error("[KeOps] Wrong value of the 'i' dimension "
                                     + std::to_string(nbatchdims) + "for arg number " + std::to_string(i)
                                     + " : is " + std::to_string(shapes[off_i + nbatchdims])
                                     + " but was " + std::to_string(shapes[nbatchdims])
                                     + " in previous 'i' arguments.");
      }

      // And the number of "columns":
      if (shapes[off_i + nbatchdims + 2] != dimensions[i]) {
        keops_error("[KeOps] Wrong value of the 'vector size' dimension "
                                     + std::to_string(nbatchdims + 1) + "for arg number " + std::to_string(i)
                                     + " : is " + std::to_string(shapes[off_i + nbatchdims + 2])
                                     + " but should be " + std::to_string(dimensions[i]));
      }
    } else if (categories[i] == 1) {  // "j" variable ----------------------------------
      shapes[off_i + nbatchdims] = 1;
      shapes[off_i + nbatchdims + 1] = get_size(obj_ptr[i], nbatchdims);  // = "N"
      if (shapes[nbatchdims + 1] == -1) {  // This is the first "j" variable that we encounter
        shapes[nbatchdims + 1] = shapes[off_i + nbatchdims + 1];  // -> Fill in the "N" coefficient in the first line
      }

      shapes[off_i + nbatchdims + 2] = get_size(obj_ptr[i], nbatchdims + 1);  // = "D"

      // Check the number of "lines":
      if (shapes[nbatchdims + 1] != shapes[off_i + nbatchdims + 1]) {
        keops_error("[KeOps] Wrong value of the 'j' dimension "
                                     + std::to_string(nbatchdims) + "for arg number " + std::to_string(i)
                                     + " : is " + std::to_string(shapes[off_i + nbatchdims + 1])
                                     + " but was " + std::to_string(shapes[nbatchdims + 1])
                                     + " in previous 'j' arguments.");
      }

      // And the number of "columns":
      if (shapes[off_i + nbatchdims + 2] != dimensions[i]) {
        keops_error("[KeOps] Wrong value of the 'vector size' dimension "
                                     + std::to_string(nbatchdims + 1) + "for arg number " + std::to_string(i)
                                     + " : is " + std::to_string(shapes[off_i + nbatchdims + 2])
                                     + " but should be " + std::to_string(dimensions[i]));
      }

    } else if (categories[i] == 2) {  // "parameters" -------------------------------
      shapes[off_i + nbatchdims] = 1;
      shapes[off_i + nbatchdims + 1] = 1;
      shapes[off_i + nbatchdims + 2] = get_size(obj_ptr[i], nbatchdims);  // = "D"

      if (shapes[off_i + nbatchdims + 2] != dimensions[i]) {
        keops_error("[KeOps] Wrong value of the 'vector size' dimension "
                                     + std::to_string(nbatchdims) + "for arg number " + std::to_string(i)
                                     + " : is " + std::to_string(shapes[off_i + nbatchdims + 2])
                                     + " but should be " + std::to_string(dimensions[i]));
      }
    }

    if (!is_contiguous(obj_ptr[i])) {
      keops_error("[KeOps] Arg number " + std::to_string(i) + " : is not contiguous. "
                                   + "Please provide 'contiguous' data array, as KeOps does not support strides. "
                                   + "If you're getting this error in the 'backward' pass of a code using torch.sum() "
                                   + "on the output of a KeOps routine, you should consider replacing 'a.sum()' with "
                                   + "'(1. * a).sum()' or 'torch.dot(a.view(-1), torch.ones_like(a).view(-1))'. ");
    }
  }

  // Compute the total numbers nx and ny of "i" and "j" indices ==========
  // Remember that the first line of "shapes" is given by:
  //
  // [ A, .., B, M, N, D_out]  -> output

  if (shapes[nbatchdims] == -1) {  // If the formula does not contain any "x" variable
    shapes[nbatchdims] = 1;        // Let's say that M = 1.
  }
  if (shapes[nbatchdims + 1] == -1) {  // If the formula does not contain any "y" variable
    shapes[nbatchdims + 1] = 1;        // Let's say that N = 1.
  }

  int nx = shapes[nbatchdims];      // = M
  int ny = shapes[nbatchdims + 1];  // = N


  int nbatches = 1;
  for (int b = 0; b < nbatchdims; b++) {
    nbatches *= shapes[b];  // Compute the product of all "batch dimensions"
  }
  nx *= nbatches;  // = A * ... * B * M
  ny *= nbatches;  // = A * ... * B * N

  return std::make_tuple(nx, ny, nbatchdims, shapes);

}

}