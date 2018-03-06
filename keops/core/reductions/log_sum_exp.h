#pragma once

#include <iostream>

#include "core/Pack.h"

#include "core/autodiff.h"

#ifdef __CUDACC__
	#include <npp.h>
#endif

// Pads a "1" in front of a scalar F.
// This command in *only* meant to be used internally,
// when computing a Kernel Product as a log-sum-exp operation.
// It allows us to turn a real number F into a pair
// (m,s) = (F,1) which encodes the number s * exp(F).
// Declared using the   LogSumExp<F>   syntax.
//
// Giving a "LogSumExp" to a Conv1D/2D routine will automatically
// result in it using a numerically stable reduce operation.
template < class F >
struct LogSumExp {
    static const int DIM = 1 + F::DIM;
    static_assert(1==F::DIM,"LogSumExp is only meant to be used with scalars.");

    static void PrintId() {
        cout << "LogSumExp<";
        F::PrintId();
        cout << ">";
    }

    template<class A, class B>
    using Replace = CondType< B , LogSumExp<typename F::template Replace<A,B>> , IsSameType<A,LogSumExp<F>>::val >;

    using AllTypes = MergePacks<univpack<LogSumExp<F>>,typename F::AllTypes>;

    template < int CAT >
    using VARS = typename F::template VARS<CAT>;

    // Evaluation is easy : simply fill-up out[-1] with 1, then eval F on out.
    template < class INDS, typename... ARGS >
    static HOST_DEVICE INLINE void Eval(__TYPE__* out, ARGS... args) {
        // The evaluated cell is (m,1) which represents exp(m) * 1
        out[F::DIM] = 1.0f;
        F::template Eval<INDS>(out, args...);
    }

    // LogSumExp is a utility function related to a numerically stable reduction operation.
    // It should not be used a a standard formula by the user, and thus,
    // should never be differentiated !
    // Calling LogSumExp<F>::DiffT should result in an error, so we may as well
    // note define it.

    // template < class V, class GRADIN >
    // using DiffT = ...;
};

template <typename TYPE>
struct NEG_INFINITY;

template <>
struct NEG_INFINITY<float> {
    static constexpr float value = - INFINITY_FLOAT;
};

template <>
struct NEG_INFINITY<double> {
    static constexpr double value = - INFINITY_DOUBLE;
};

// Overloads the reduce operations when F is a LogSumExp<G>
template <typename TYPE, int DIM, class G>
struct InitializeOutput<TYPE,DIM,LogSumExp<G>> {
    HOST_DEVICE INLINE void operator()(TYPE *tmp) {
        // We fill empty cells with the neutral element of the reduction operation,
        //                   (-inf,0) = e^{-inf} * 0 = 0
        static_assert(2==DIM,"LogSumExp is only meant to be used with scalars -> pairs (m,s).");
        // We should use 0xfff0000000000000 for doubles
        //-340282346638528859811704183484516925440.0f;//__int_as_float(0xff800000); // -infty, as +infty = 0x7f800000
        tmp[0] = NEG_INFINITY<TYPE>::value;
        tmp[1] = 0.0f;
    }
};

// equivalent of the += operation
template <typename TYPE, int DIM, class G>
struct ReducePair<TYPE,DIM,LogSumExp<G>> {
    HOST_DEVICE INLINE void operator()(TYPE *tmp, TYPE *xi) {
        static_assert(2==DIM,"LogSumExp is only meant to be used with scalars -> pairs (m,s).");
        // (m,s) + (m',s'), i.e. exp(m)*s + *exp(m')*s'
        if(tmp[0] > xi[0]) { // =  exp(m)  * (s + exp(m'-m)*s')   if m > m'
            tmp[1] += exp( xi[0]-tmp[0] ) * xi[1] ;
        } else {             // =  exp(m') * (s' + exp(m-m')*s)   if m <= m'
            tmp[1] = xi[1] + exp( tmp[0]-xi[0] ) * tmp[1] ;
            tmp[0] = xi[0] ;
        }
    }
};

