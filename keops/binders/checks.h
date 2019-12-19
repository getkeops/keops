#pragma once

#include <string>
#include <tuple>
#include <functional>

#include "binders/keops_cst.h"
#include "binders/switch.h"



namespace keops_binders {

using namespace keops;

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

template< typename array_t >
void check_contiguity(array_t &obj_ptr, int i) {
  if (!is_contiguous(obj_ptr)) {
    keops_error("[Keops] Arg at position " + std::to_string(i) + ": is not contiguous. "
                + "Please provide 'contiguous' dara array, as KeOps does not support strides. "
                + "If you're getting this error in the 'backward' pass of a code using torch.sum() "
                + "on the output of a KeOps routine, you should consider replacing 'a.sum()' with "
                + "'torch.dot(a.view(-1), torch.ones_like(a).view(-1))'. ");
  }
}


template< typename array_t >
void Sizes< array_t >::check_ranges(int nargs, array_t* args) {
//void check_ranges(int nargs, array_t* args) {

  // Check the compatibility of all tensor shapes ==================================
  
  
  if (NMINARGS > 0) {
    
    // Checks args in all the positions that correspond to "i" variables:
    for (int k = 0; k < NARGSI; k++) {
      int i = INDSI::VAL(k);
      // Fill in the (i+1)-th line of the "shapes" array ---------------------------
      int off_i = (i + 1) * (nbatchdims + 3);
      
      // Check the number of dimensions --------------------------------------------
      int ndims = get_ndim(args[i]);  // Number of dims of the i-th tensor
      
      if (ndims != nbatchdims + 2) {
        keops_error("[KeOps] Wrong number of dimensions for arg at position " + std::to_string(i)
                    + " (i type): KeOps detected " + std::to_string(nbatchdims)
                    + " batch dimensions from the first argument 0, and thus expected "
                    + std::to_string(nbatchdims + 2)
                    + " dimensions here, but only received "
                    + std::to_string(ndims)
                    + ". Note that KeOps supports broadcasting on batch dimensions, "
                    + "but still expects 'dummy' unit dimensions in the input shapes, "
                    + "for the sake of clarity.");
      }
  
  
  
  
      // First, the batch dimensions:
      for (int b = 0; b < nbatchdims; b++) {
        _shapes[off_i + b] = get_size_batch(args[i], nbatchdims + 2, b);
        
        // Check that the current value is compatible with what
        // we've encountered so far, as stored in the first line of "shapes"
        if (_shapes[off_i + b] != 1) {  // This dimension is not "broadcasted"
          if (_shapes[b] == 1) {
            _shapes[b] = _shapes[off_i + b];  // -> it becomes the new standard
          } else if (_shapes[b] != _shapes[off_i + b]) {
            keops_error("[KeOps] Wrong value of the batch dimension "
                        + std::to_string(b) + " for argument number " + std::to_string(i)
                        + " : is " + std::to_string(_shapes[off_i + b])
                        + " but was " + std::to_string(_shapes[b])
                        + " or 1 in previous arguments.");
          }
        }
      }
  
      _shapes[off_i + nbatchdims] = get_size(args[i], MN_pos);  // = "M"
      _shapes[off_i + nbatchdims + 2] = get_size(args[i], D_pos);  // = "D"
      
      // Check the number of "lines":
      if (_shapes[nbatchdims] != _shapes[off_i + nbatchdims]) {
        keops_error("[KeOps] Wrong value of the 'i' dimension "
                    + std::to_string(nbatchdims) + "for arg at position " + std::to_string(i)
                    + " : is " + std::to_string(_shapes[off_i + nbatchdims])
                    + " but was " + std::to_string(_shapes[nbatchdims])
                    + " in previous 'i' arguments.");
      }
      
      // And the number of "columns":
      if (_shapes[off_i + nbatchdims + 2] != static_cast< int >(DIMSX::VAL(k))) {
        keops_error("[KeOps] Wrong value of the 'vector size' dimension "
                    + std::to_string(nbatchdims + 1) + " for arg at position " + std::to_string(i)
                    + " : is " + std::to_string(_shapes[off_i + nbatchdims + 2])
                    + " but should be " + std::to_string(DIMSX::VAL(k)));
      }
  
      check_contiguity(args[i], i);
    }
    
    
    // Checks args in all the positions that correspond to "j" variables:
    for (int k = 0; k < NARGSJ; k++) {
      int i = INDSJ::VAL(k);
      
      // Check the number of dimensions --------------------------------------------
      int ndims = get_ndim(args[i]);  // Number of dims of the i-th tensor
      
      if (ndims != nbatchdims + 2) {
        keops_error("[KeOps] Wrong number of dimensions for arg at position " + std::to_string(i)
                    + " (j type): KeOps detected " + std::to_string(nbatchdims)
                    + " batch dimensions from the first argument 0, and thus expected "
                    + std::to_string(nbatchdims + 2)
                    + " dimensions here, but only received "
                    + std::to_string(ndims)
                    + ". Note that KeOps supports broadcasting on batch dimensions, "
                    + "but still expects 'dummy' unit dimensions in the input shapes, "
                    + "for the sake of clarity.");
      }
      
      // Fill in the (i+1)-th line of the "shapes" array ---------------------------
      int off_i = (i + 1) * (nbatchdims + 3);
      
      // First, the batch dimensions:
      for (int b = 0; b < nbatchdims; b++) {
        _shapes[off_i + b] = get_size_batch(args[i], nbatchdims + 2, b);
        
        // Check that the current value is compatible with what
        // we've encountered so far, as stored in the first line of "shapes"
        if (_shapes[off_i + b] != 1) {  // This dimension is not "broadcasted"
          if (_shapes[b] == 1) {
            _shapes[b] = _shapes[off_i + b];  // -> it becomes the new standard
          } else if (_shapes[b] != _shapes[off_i + b]) {
            keops_error("[KeOps] Wrong value of the batch dimension "
                        + std::to_string(b) + " for argument number " + std::to_string(i)
                        + " : is " + std::to_string(_shapes[off_i + b])
                        + " but was " + std::to_string(_shapes[b])
                        + " or 1 in previous arguments.");
          }
        }
      }
  
      _shapes[off_i + nbatchdims + 1] = get_size(args[i], MN_pos);  // = "N"
      _shapes[off_i + nbatchdims + 2] = get_size(args[i], D_pos);  // = "D"
      
      // Check the number of "lines":
      if (_shapes[nbatchdims + 1] != _shapes[off_i + nbatchdims + 1]) {
        keops_error("[KeOps] Wrong value of the 'j' dimension "
                    + std::to_string(nbatchdims) + " for arg at position " + std::to_string(i)
                    + " : is " + std::to_string(shapes[off_i + nbatchdims + 1])
                    + " but was " + std::to_string(shapes[nbatchdims + 1])
                    + " in previous 'j' arguments.");
      }
      
      // And the number of "columns":
      if (_shapes[off_i + nbatchdims + 2] != static_cast< int >(DIMSY::VAL(k))) {
        keops_error("[KeOps] Wrong value of the 'vector size' dimension "
                    + std::to_string(nbatchdims + 1) + " for arg at position " + std::to_string(i)
                    + " : is " + std::to_string(_shapes[off_i + nbatchdims + 2])
                    + " but should be " + std::to_string(DIMSY::VAL(k)));
      }
  
      check_contiguity(args[i], i);
    }
    
    
    for (int k = 0; k < NARGSP; k++) {
      int i = INDSP::VAL(k);
      // Fill in the (i+1)-th line of the "shapes" array ---------------------------
      int off_i = (i + 1) * (nbatchdims + 3);
  
      // First, the batch dimensions:
      for (int b = 0; b < nbatchdims; b++) {
        _shapes[off_i + b] = get_size_batch(args[i], nbatchdims + 2, b);
      }
        
      _shapes[off_i + nbatchdims + 2] = get_size(args[i], nbatchdims);  // = "D"
  
      if (_shapes[off_i + nbatchdims + 2] != static_cast< int >(DIMSP::VAL(k))) {
        keops_error("[KeOps] Wrong value of the 'vector size' dimension "
                    + std::to_string(nbatchdims) + " for arg at position " + std::to_string(i)
                    + " : is " + std::to_string(_shapes[off_i + nbatchdims + 2])
                    + " but should be " + std::to_string(DIMSP::VAL(k)));
      }
  
      check_contiguity(args[i], i);
    }
  }
  
  
}

}
