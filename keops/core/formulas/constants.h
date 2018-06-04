#pragma once

#include <iostream>

#include "core/Pack.h"
#include "core/autodiff.h"

/*
 * The file where the elementary constants are defined.
 * Available constants are :
 *
 *      Zero<DIM>					: zero-valued vector of dimension DIM
 *      IntConstant<N>				: constant integer function with value N
 *
 */

// A "zero" vector of size _DIM
// Declared using the   Zero<DIM>   syntax.
namespace keops {

template < int _DIM >
struct Zero {
    static const int DIM = _DIM;

    // in c++17 the IdString() function could be replaced by:
    //constexpr std::string_view id ="0";
    static const std::string& IdString(){
        static const std::string str = "0";
        return str;
    }
    static void PrintId() { std::cout << IdString(); }

    template<class A, class B>
    using Replace = Zero<DIM>;
    
    using AllTypes = univpack<Zero<DIM>>;

    template < int CAT >      // Whatever CAT...
    using VARS = univpack<>;  // there's no variable used in there.

    // Evaluation is easy : simply fill-up *out with zeros.
    template < class INDS, typename... ARGS >
    static HOST_DEVICE INLINE void Eval(__TYPE__* out, ARGS... args) {
        for(int k=0; k<DIM; k++)
            out[k] = 0;
    }

    // There is no gradient to accumulate on V, whatever V.
    template < class V, class GRADIN >
    using DiffT = Zero<V::DIM>;
};



// A constant integer value, defined using the IntConstant<N> syntax.

template < int N > struct IntConstantImpl;
template < int N > struct IntConstantAlias;
template < int N > 
using IntConstant = typename IntConstantAlias<N>::type; 

template < int N >
struct IntConstantImpl {
    static const int DIM = 1;

    // in c++17 the IdString() function could be replaced by:
    //constexpr std::string_view id = std::to_string(N);
    static const std::string& IdString(){
        static const std::string str = std::to_string(N);
        return str;
    }

    static void PrintId() { std::cout << IdString(); }

    template<class A, class B>
    using Replace = IntConstant<N>;
    
    using AllTypes = univpack<IntConstant<N>>;

    template < int CAT >      // Whatever CAT...
    using VARS = univpack<>;  // there's no variable used in there.

    // Evaluation is easy : simply fill *out = out[0] with N.
    template < class INDS, typename... ARGS >
    static HOST_DEVICE INLINE void Eval(__TYPE__* out, ARGS... args) {
        *out = N;
    }

    // There is no gradient to accumulate on V, whatever V.
    template < class V, class GRADIN >
    using DiffT = Zero<V::DIM>;
};

// Simplification rule

// base class, redirects to implementation
template < int N > 
struct IntConstantAlias { 
    using type = IntConstantImpl<N>; 
}; 
 
// 0 = 0
template<>
struct IntConstantAlias<0> { 
    using type = Zero<1>; 
}; 

}
