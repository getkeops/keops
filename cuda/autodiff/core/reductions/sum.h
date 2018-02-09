#pragma once

#include "core/Pack.h"

#include "core/autodiff.h"


// Implements the default reduction operation: the summation.

// Sum is the generic template, with arbitrary formula F.
// These functions can be overloaded if, for instance,
//                 F = LogSumExp<G>.

template <typename TYPE, int DIM, class F>
struct InitializeOutput {
    HOST_DEVICE INLINE void operator()(TYPE *tmp) {
        for(int k=0; k<DIM; k++)
            tmp[k] = 0.0f; // initialize output
    }
};

// equivalent of the += operation
template <typename TYPE, int DIM, class F>
struct ReducePair {
    HOST_DEVICE INLINE void operator()(TYPE *tmp, TYPE *xi) {
        for(int k=0; k<DIM; k++) {
            tmp[k] += xi[k];
        }
    }
};

