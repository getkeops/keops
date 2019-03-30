/*
 * To differentiate automatically our code at compilation time, we'll have to use some
 * very advanced C++ syntax. Indeed, as we use the compiler as a "formula/graph-processing engine",
 * we need it to process tree structures and lists of variables.
 * This is achieved using the recursive ("variadic") templating of C++11;
 * we define the following "container/symbolic" templates:
 * - univpack,    which acts as a list of symbolic types
 * - pack,        which acts more specifically as a list of vectors of known sizes
 * - CondType,    which acts as a symbolic conditional statement
 * - ConcatPacks, which is a concatenation operator for packs
 *
 */

#pragma once

#include <sstream>
#include <assert.h>

#ifdef __CUDACC__
	#include <npp.h>
	#define HOST_DEVICE __host__ __device__
	#define INLINE __forceinline__
	#define INFINITY_FLOAT NPP_MAXABS_32F
	#define INFINITY_DOUBLE NPP_MAXABS_64F
#else
        #define HOST_DEVICE 
	#include <limits>
	#define INLINE inline
	#define INFINITY_FLOAT std::numeric_limits<float>::infinity()
	#define INFINITY_DOUBLE std::numeric_limits<double>::infinity()
#endif

#define __INDEX__ int32_t // use int instead of double

namespace keops {

// templated plus and minus infinity
template <typename TYPE>
struct PLUS_INFINITY;

template <>
struct PLUS_INFINITY<float> {
    static constexpr float value = INFINITY_FLOAT;
};

template <>
struct PLUS_INFINITY<double> {
    static constexpr double value = INFINITY_DOUBLE;
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

// At compilation time, detect the maximum between two values (typically, dimensions)
template <typename T>
static constexpr T static_max(T a, T b) {
    return a < b ? b : a;
}

// At compilation time, detect the minimum between two values (typically, dimensions)
template <typename T>
static constexpr T static_min(T a, T b) {
    return a < b ? a : b;
}

// same but for integers only and defining classes to avoid constexpr (because nvcc does not allow it in device functions)
template <int M, int N>
struct INTMAX {
    static const int value = M < N ? N : M;
};

template <int M, int N>
struct INTMIN {
    static const int value = M < N ? M : N;
};

// custom implementation of tuple "get" function, to avoid the use of thrust/tuple
// which is limited to 10 arguments. This function works only when all arguments
// have same type

template < int N, typename TYPE >
struct GetHelper {
    template < typename... ARGS >
    static HOST_DEVICE INLINE TYPE Eval(TYPE arg0, ARGS... args) {
        return GetHelper<N-1,TYPE>::Eval(args...);
    }
};

template < typename TYPE >
struct GetHelper<0,TYPE> {
    template < typename... ARGS >
    static HOST_DEVICE INLINE TYPE Eval(TYPE arg0, ARGS... args) {
        return arg0;
    }
};

template < int N, typename TYPE, typename... ARGS >
static HOST_DEVICE INLINE TYPE Get(TYPE arg0, ARGS... args) {
    return GetHelper<N,TYPE>::Eval(arg0, args...);
}

// Conditional type, a templating emulator of a conditional statement. --------------------------
// This convoluted syntax allows us to write
// CondType<A,B,1> = A,  CondType<A,B,0> = B
template < class A, class B, bool TEST >
struct CondType_Alias;

template < class A, class B >
struct CondType_Alias<A,B,true> {
    using type = A;
};

template < class A, class B >
struct CondType_Alias<A,B,false> {
    using type = B;
};

template < class A, class B, bool TEST >
using CondType = typename CondType_Alias<A,B,TEST>::type;


// IsSameType<A,B> = false and IsSameType<A,A> = true
template < class A, class B >
struct IsSameType_Alias {
    static const bool val = false;
};

template < class A >
struct IsSameType_Alias<A,A> {
    static const bool val = true;
};

template < class A, class B >
struct IsSameType {
    static const bool val = IsSameType_Alias<A,B>::val;
};


// "univpack" is a minimal "templating list", defined recursively. ------------------------------
// It allows us to work with "lists" of variables in a formula, at compilation time.

template < int... NS > struct pack; // "pack" is the m

// The empty "univpack", an empty list "< > = []"
template < typename... Args >
struct univpack {
    using FIRST = void;         // [].head() = void
    static const int SIZE = 0;  // len([])   = 0

    // helpers to print the univpack to the standard output
    static void PrintAll(std::stringstream& str) { }
    static void PrintComma(std::stringstream& str) { }
    static void PrintId(std::stringstream& str) {
        str << "univpack< >";
    }

    template < class D >        // [].append_first(D) = [D]
    using PUTLEFT = univpack<D>;

    using NEXT = void;          // [].tail() = void
};

// A non-empty univpack, defined recursively as [C] + univpack( Args )
template < class C, typename... Args >
struct univpack<C,Args...> {
    using FIRST = C;             // [C, ...].head() = C
    static const int SIZE = 1+sizeof...(Args); // len([C, ...]) = 1 + len([...])

    // helpers to print the univpack to the standard output
    static void PrintComma(std::stringstream& str) {
        str << " ," << std::endl;
    }

    static void PrintAll(std::stringstream& str) {
        FIRST::PrintId(str);
        NEXT::PrintComma(str);
        NEXT::PrintAll(str);
    }

    static void PrintId(std::stringstream& str) {
        str << "univpack< " << std::endl;
        PrintAll(str);
        str << " >";
    }

    template < class D >         // [C, ...].append_first(D) = [D, C, ...]
    using PUTLEFT = univpack<D, C, Args...>;

    using NEXT = univpack<Args...>; // [C, ...].tail() = [...]

};




// OPERATIONS ON PACKS AND UNIVPACKS ============================================================
// Once again, a convoluted syntax to write the "concatenation" of two lists. -------------------
// ConcatPacks<[...],[...]> = [..., ...]  (for packs or univpacks)
template < class PACK1, class PACK2 >
struct ConcatPacks_Alias {
    using type = int;
}; // default dummy type

template < int... IS, int... JS >
struct ConcatPacks_Alias<pack<IS...>,pack<JS...>> {
    using type = pack<IS...,JS...>;
};

template < typename... Args1, typename... Args2 >
struct ConcatPacks_Alias<univpack<Args1...>,univpack<Args2...>> {
    using type = univpack<Args1...,Args2...>;
};

template < class PACK1, class PACK2 >
using ConcatPacks = typename ConcatPacks_Alias<PACK1,PACK2>::type;


// count number of occurrences of a type in a univpack

template < class C, class PACK >
struct CountInPack_Alias {
    static const int N = 0;
};

template < class C, typename... Args >
struct CountInPack_Alias<C,univpack<C,Args...>> { // CountIn( C, [C, ...] )
    static const int N = 1+CountInPack_Alias<C,univpack<Args...>>::N;
};

template < class C, class D, typename... Args >
struct CountInPack_Alias<C,univpack<D,Args...>> { // CountIn( C, [D, ...] )
    static const int N = CountInPack_Alias<C,univpack<Args...>>::N;
};

template < class C >
struct CountInPack_Alias<C,univpack<>> {        // CountIn( C, [] )
    static const int N = 0;
};

//template < class C, class PACK >
//static const int CountInPack() { return CountInPack_Alias<C,PACK>::N; }


// Remove an element from a univpack

template < class C, class PACK >
struct RemoveFromPack_Alias;

template < class C >
struct RemoveFromPack_Alias<C,univpack<>> { // RemoveFrom( C, [] )
    using type = univpack<>;    // []
};

template < class C, class D, typename... Args >
struct RemoveFromPack_Alias<C,univpack<D,Args...>> { // RemoveFrom( C, [D, ...] )
    using tmp = typename RemoveFromPack_Alias<C,univpack<Args...>>::type;
    using type = typename tmp::template PUTLEFT<D>;     // = [D] + RemoveFrom( C, [...] )
};

template < class C, typename... Args >
struct RemoveFromPack_Alias<C,univpack<C,Args...>> { // RemoveFrom( C, [C, ...] )
    using type = typename RemoveFromPack_Alias<C,univpack<Args...>>::type;
};

//template < class C, class PACK >
//using RemoveFromPack = typename RemoveFromPack_Alias<C,PACK>::type;

// Merge operation for univpacks. ---------------------------------------------------------------
// MergePacks<[...],[...]> = {...} : a "merged" list, without preservation of ordering
//                                   and uniqueness of elements
// Basically, this operator concatenates two LISTS and sees the result as a SET.
// (Jean :) the syntax becomes *really* convoluted here. I may have made a mistake when commenting.

template < class PACK1, class PACK2 >
struct MergePacks_Alias;

template < class C, typename... Args1, typename... Args2 >
struct MergePacks_Alias<univpack<Args1...>,univpack<C,Args2...>> {         // Merge([...], [C,...])
    using tmp = typename RemoveFromPack_Alias<C,univpack<Args1...>>::type;
    using type = typename MergePacks_Alias<ConcatPacks<tmp,univpack<C>>,univpack<Args2...>>::type;
};

template < typename... Args1 >
struct MergePacks_Alias<univpack<Args1...>,univpack<>> {                   // Merge( [], [...])
    using type = univpack<Args1...>;
};                                   // = [...]

template < class PACK1, class PACK2 >
using MergePacks = typename MergePacks_Alias<PACK1,PACK2>::type;

// Get the list of dimensions. ------------------------------------------------------------------
// GetDims([a, b, c]) = [dim_a, dim_b, dim_c]                (works for univpacks)
template < class UPACK >
struct GetDims_Alias {
    using a = typename UPACK::NEXT;
    using c = typename GetDims_Alias<a>::type;
    using type = typename c::template PUTLEFT<UPACK::FIRST::DIM>;
};

template <>
struct GetDims_Alias< univpack<> > {
    using type = pack<>;
};

template < class UPACK >
using GetDims = typename GetDims_Alias<UPACK>::type;


// Get the list of indices (useful for univpacks of abstract Variables) -------------------------
// GetInds( [X1, X3, Y2] ) = [1, 3, 2]
template < class UPACK >
struct GetInds_Alias {                                  // GetInds( [Xi, ...] )
    using a = typename UPACK::NEXT;
    using c = typename GetInds_Alias<a>::type;
    using type = typename c::template PUTLEFT<UPACK::FIRST::N>; // = [i] + GetInds( [...] )
};

template <>
struct GetInds_Alias< univpack<> > { // GetInds( [] )
    using type = pack<>;
};        // = []

template < class UPACK >
using GetInds = typename GetInds_Alias<UPACK>::type;


// Search position in a pack -------------------------------------------------------------------------
// IndVal( [ 1, 4, 2, 0,...], 2 ) = 3
template < class INTPACK, int N >    // IndVal( [C, ...], N)     ( C != N )
struct IndVal_Alias {                 // = 1 + IndVal( [...], N)
    static const int ind = 1+IndVal_Alias<typename INTPACK::NEXT,N>::ind;
};

template < int N, int... NS >
struct IndVal_Alias< pack<N,NS...>, N > { // IndVal( [N, ...], N)
    static const int ind = 0;
};        // = 0

template < int N >
struct IndVal_Alias< pack<>, N > {       // IndVal( [], N )
    static const int ind = 0;
};        // = 0

template < class INTPACK, int N >
struct IndVal {				// Use as IndVal<Intpack, N>::value
    static const int value = IndVal_Alias<INTPACK,N>::ind;
};

// Access the n-th element of an univpack -------------------------------------------------------
// Val( [ x0, x1, x2, ...], i ) = xi
template < class UPACK, int N >                  // Val([C, ...], N)  (N > 0)
struct Val_Alias {
    using a = typename UPACK::NEXT;
    using type = typename Val_Alias<a,N-1>::type; // = Val([...], N-1)
};

template < class UPACK >
struct Val_Alias< UPACK, 0 > {                    // Val([C, ...], 0)
    using type = typename UPACK::FIRST;
};       // = C

template < class UPACK, int N >
using Val = typename Val_Alias<UPACK,N>::type;

// PACKS OF VECTORS =================================================================================
// Define recursively how a "package" of variables should behave.
// Packages are handled as list of vectors, stored in a contiguous memory space.
// They are meant to represent (location,), (location, normal), (signal, location)
// or any other kind of "pointwise" feature that is looped over by convolution operations.
//
// A package is instantiated using a
//        "typedef pack<1,DIM,DIM> DIMSX;"
// syntax (for example), which means that DIMSX is the dimension
// of a "3-uple" of (TYPE) variables, one scalar then two vectors
// of size DIM.

// The EMPTY package : ==============================================================================
template < int... NS > struct pack {

    // DIMSX::VAL(2) is the size of its 3rd vector (starts at 0) (in the example above, DIM).
    // Therefore, EMPTY::VAL(n) should never be called : we return -1 as an error signal.
    static int VAL(int m) {
        return -1;
    }

    // helpers to print the pack to the standard output
    static void PrintAll(std::stringstream& str) { }
    static void PrintComma(std::stringstream& str) { }
    static void PrintId(std::stringstream& str) {
        str << "pack< >";
    }

    // [].append(M) = [M]
    template < int M >
    using PUTLEFT = pack<M>;

    // Furthermore, the empty package :
    static const int SIZE = 0; // Has zero size (number of vectors) ...
    static const int MAX = -1; // max is set to -1 (we assume packs of non negative integers...)
    static const int SUM = 0;  // ... zero sum  (total memory footprint) ...

    // ... is loaded trivially ...
    template < typename TYPE >
    HOST_DEVICE static void load(int i, TYPE* xi, TYPE** px) { }

    // ... counts for nothing in the evaluation of a function ...
    template < typename TYPE, class FUN, typename... Args  >
    HOST_DEVICE static void call(FUN fun, TYPE* x, Args... args) {
        fun(args...);
    }

    // ... idem ...
    template < class DIMS, typename TYPE, class FUN, typename... Args  >
    HOST_DEVICE static void call2(FUN fun, TYPE* x, Args... args) {
        DIMS::call(fun,args...);
    }

    // ... idem ...
    template < class DIMS1, class DIMS2, typename TYPE, class FUN, typename... Args  >
    HOST_DEVICE static void call3(FUN fun, TYPE* x, Args... args) {
        DIMS1::template call2<DIMS2>(fun,args...);
    }

    // ... does not have anything to give to a list of variables.
    template < typename TYPE, typename... Args  >
    static void getlist(TYPE** px, Args... args) { }

};

// A non-EMPTY package, recursively defined as : ====================================================
// "The concatenation of a vector of size N, and a (possibly empty) package."
template < int N, int... NS > struct pack<N,NS...> {
    static const int FIRST = N;    // Size of its first element.

    // DIMSX::VAL(2) = size of its 3rd vector (we start counting at 0).
    static int VAL(int m) {
        if(m)
            return NEXT::VAL(m-1);
        else
            return FIRST;
    }

    // helpers to print the pack to the standard output
    static void PrintComma(std::stringstream& str) {
        str << ",";
    }

    static void PrintAll(std::stringstream& str) {
        str << FIRST;
        NEXT::PrintComma(str);
        NEXT::PrintAll(str);
    }

    static void PrintId(std::stringstream& str) {
        str << "pack<" ;
        PrintAll(str);
        str << ">";
    }

    // Operation to append "M" at the head of the list
    template < int M >
    using PUTLEFT = pack<M, N, NS...>;

    static const int SIZE = 1+sizeof...(NS); // The number of vectors in pack<N,NS...>
    typedef pack<NS...> NEXT;                // "NEXT" is the tail of our list of vectors.
    static const int MAX = static_max(N,NEXT::MAX);  // get the max of values
    static const int SUM = N + NEXT::SUM;    // The total "memory footprint" of pack<N,NS...> is computed recursively.

    // Loads the i-th element of the (global device memory pointer) px
    // to the "array" xi.
    template < typename TYPE >
    HOST_DEVICE static void load(int i, TYPE* xi, TYPE** px) {
        assert(xi != nullptr);
        assert(px != nullptr);
        /*
         * px is an "array" of pointers to data arrays of appropriate sizes.
         * That is, px[0] = *px     is a pointer to a TYPE array of size Ni * FIRST
         * Then,    px[1] = *(px+1) is a pointer to a TYPE array of size Ni * NEXT::FIRST; etc.
         *
         * (where Ni is the max value of "i" you should expect)
         * Obviously, we do not make any sanity check... so beware of illicit memory accesses !
         */
        // Using pythonic syntax, we can describe our loading procedure as follows :
        for(int k=0; k<FIRST; k++){
            assert(&((*px)[i*FIRST+k]) != nullptr);
            xi[k] = (*px)[i*FIRST+k];// First, load the i-th line of px[0]  -> xi[ 0 : FIRST ].
        }
        NEXT::load(i,xi+FIRST,px+1);   // Then,  load the i-th line of px[1:] -> xi[ FIRST : ] (recursively)
    }

    // call(fun, [x1, x2, x3], arg1, arg2 ) will end up executing fun( arg1, arg2, x1, x2, x3 ).
    template < typename TYPE, class FUN, typename... Args  >
    HOST_DEVICE static void call(FUN fun, TYPE* x, Args... args) {
        NEXT::call(fun,x+FIRST,args...,x);  // Append [x[0:FIRST]] to the list of arguments, then iterate.
    }

    // Idem, with a template on DIMS. This allows you to call fun with
    // two "packed" variables (x_i and y_j) as first inputs.
    // call2(fun, [x1, x2], [y1, y2], arg1 ) will end up executing fun(arg1, x1, x2, y1, y2).
    template < class DIMS, typename TYPE, class FUN, typename... Args  >
    HOST_DEVICE static void call2(FUN fun, TYPE* x, Args... args) {
        NEXT::template call2<DIMS>(fun,x+FIRST,args...,x);
    }

    // Idem, with a double template on DIMS. This allows you to call fun with
    // three "packed" variables
    template < class DIMS1, class DIMS2, typename TYPE, class FUN, typename... Args  >
    HOST_DEVICE static void call3(FUN fun, TYPE* x, Args... args) {
        NEXT::template call3<DIMS1,DIMS2>(fun,x+FIRST,args...,x);
    }

    // Out of a long  list of pointers, extract the ones which "belong" to the current pack
    // and put them into a pointer array px.
    template < typename TYPE, typename... Args  >
    static void getlist(TYPE** px, Args... args) {
        *px = Get<FIRST>(args...);
        NEXT::getlist(px+1,args...);
    }

};



// create pack of arbitrary size filled with zero value

template < int N >
struct ZeroPack_Alias {
    using type = typename ZeroPack_Alias<N-1>::type::template PUTLEFT<0>;
};

template < >
struct ZeroPack_Alias<0> {
    using type = pack<>;
};

template < int N >
using ZeroPack = typename ZeroPack_Alias<N>::type;



// Replace a value at position N in pack


template < class P, int V, int N > 
struct ReplaceInPack_Alias {
    using NEXTPACK = typename P::NEXT;
    using type = typename ReplaceInPack_Alias<NEXTPACK,V,N-1>::type::template PUTLEFT<P::FIRST>;
};


template < class P, int V >
struct ReplaceInPack_Alias<P,V,0> {
    using type = typename P::NEXT::template PUTLEFT<V>;
};

template < class P, int V, int N >
using ReplaceInPack = typename ReplaceInPack_Alias<P,V,N>::type;


// get the value at position N from a pack of ints

template < class P, int N > 
struct PackVal {
    using type = typename PackVal<typename P::NEXT,N-1>::type;
};

template < class P >
struct PackVal<P,0> {
    using type = PackVal<P,0>;
    static const int Val = P::FIRST;
};

// Check that all values in a pack of ints are unique
// here we count the number of times each value appears, then
// test if the sum is > 1 (which is not an optimal algorithm, it could be improved...)
template < class P, class TAB = ZeroPack<P::MAX+1> > 
struct CheckAllDistinct_BuildTab {
    static const int VAL = PackVal<TAB,P::FIRST>::type::Val;
    using NEWTAB = ReplaceInPack<TAB,VAL+1,P::FIRST>;
    using type = typename CheckAllDistinct_BuildTab<typename P::NEXT,NEWTAB>::type;
};

template < class TAB > 
struct CheckAllDistinct_BuildTab<pack<>,TAB> {
    using type = TAB;
};

template < class P > 
struct CheckAllDistinct {
    using TAB = typename CheckAllDistinct_BuildTab<P>::type;
    static const bool val = TAB::MAX<2;
};



// USEFUL METHODS ===================================================================================

// Templated call
template < class DIMSX, class DIMSY, class DIMSP, typename TYPE, class FUN, typename... Args  >
HOST_DEVICE void call(FUN fun, TYPE* x, Args... args) {
    DIMSX:: template call3<DIMSY,DIMSP>(fun,x,args...);
}

template < class INDS, typename TYPE, typename... Args >
void getlist(TYPE** px, Args... args) {
    INDS::getlist(px,args...);
}

// Loads the i-th "line" of px to xi.
template < class DIMS, typename TYPE >
HOST_DEVICE void load(int i, TYPE* xi, TYPE** px) {
    DIMS::load(i,xi,px);
}

// global variables maxThreadsPerBlock and sharedMemPerBlock may depend on the device, so we will set them at each call using
// predefined MAXTHREADSPERBLOCK0, SHAREDMEMPERBLOCK0, MAXTHREADSPERBLOCK1, SHAREDMEMPERBLOCK1, etc.
// through the function SetGpuProps
int maxThreadsPerBlock, sharedMemPerBlock;
#define SET_GPU_PROPS_MACRO(n) \
    if(device==n) { \
		maxThreadsPerBlock = MAXTHREADSPERBLOCK ## n; \
		sharedMemPerBlock = SHAREDMEMPERBLOCK ## n; \
                return; \
	}

// I have not managed to use a "recursive macro" hack, it was not compiling on all systems.
// This assumes the number of Gpus is <= 10 ; feel free to add more lines if needed !
void SetGpuProps(int device) {
    #if defined(__CUDACC__) && MAXIDGPU >= 0 
        SET_GPU_PROPS_MACRO(0)
    #endif
    #if defined(__CUDACC__) && MAXIDGPU >= 1 
        SET_GPU_PROPS_MACRO(1)
    #endif
    #if defined(__CUDACC__) && MAXIDGPU >= 2 
        SET_GPU_PROPS_MACRO(2)
    #endif
    #if defined(__CUDACC__) && MAXIDGPU >= 3 
        SET_GPU_PROPS_MACRO(3)
    #endif
    #if defined(__CUDACC__) && MAXIDGPU >= 4 
        SET_GPU_PROPS_MACRO(4)
    #endif
    #if defined(__CUDACC__) && MAXIDGPU >= 5 
        SET_GPU_PROPS_MACRO(5)
    #endif
    #if defined(__CUDACC__) && MAXIDGPU >= 6 
        SET_GPU_PROPS_MACRO(6)
    #endif
    #if defined(__CUDACC__) && MAXIDGPU >= 7 
        SET_GPU_PROPS_MACRO(7)
    #endif
    #if defined(__CUDACC__) && MAXIDGPU >= 8 
        SET_GPU_PROPS_MACRO(8)
    #endif
    #if defined(__CUDACC__) && MAXIDGPU >= 9 
        SET_GPU_PROPS_MACRO(9)
    #endif
    fprintf( stderr, "invalid Gpu device number. If the number of available Gpus is >= 10, add required lines at the end of function SetGpuProps and recompile.\n");
    exit( -1 );

}

}


