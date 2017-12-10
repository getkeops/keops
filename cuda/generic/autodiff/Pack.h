#ifndef PACK
#define PACK

#include <tuple>

using namespace std;

template < class A, class B, bool TEST >
struct CondTypeAlias;

template < class A, class B >
struct CondTypeAlias<A,B,true>
{
	using type = A;
};

template < class A, class B >
struct CondTypeAlias<A,B,false>
{
	using type = B;
};

template < class A, class B, bool TEST >
using CondType = typename CondTypeAlias<A,B,TEST>::type;




template < int... NS > struct pack;

template < typename... Args >
struct univpack
{
    using FIRST = void;
    
	static const int SIZE = 0;
	
    template < class D >
    using PUTLEFT = univpack<D>;
		
	using NEXT = void;

};


template < class C, typename... Args >
struct univpack<C,Args...>
{
    using FIRST = C;
    
	static const int SIZE = 1+sizeof...(Args);
	
    template < class D >
    using PUTLEFT = univpack<D, C, Args...>;
		
	using NEXT = univpack<Args...>;

};

template < class PACK1, class PACK2 >
struct ConcatPacksAlias
{
	using type = int;
};

template < int... IS, int... JS >
struct ConcatPacksAlias<pack<IS...>,pack<JS...>>
{
	using type = pack<IS...,JS...>;
};

template < class PACK1, class PACK2 >
using ConcatPacks = typename ConcatPacksAlias<PACK1,PACK2>::type;



template < class C, class PACK >
struct MergeInPackAlias
{
	using type = int;
};

template < class C, class D, typename... Args >
struct MergeInPackAlias<C,univpack<D,Args...>>
{
	using tmp = typename MergeInPackAlias<C,univpack<Args...>>::type;
	using type = typename tmp::PUTLEFT<D>;
};

template < class C, typename... Args >
struct MergeInPackAlias<C,univpack<C,Args...>>
{
	using type = univpack<C,Args...>;
};

template < class C >
struct MergeInPackAlias<C,univpack<>>
{
	using type = univpack<C>;
};

//template < class C, class PACK >
//using MergeInPack = typename MergeInPackAlias<C,PACK>::type;

	

template < class PACK1, class PACK2 >
struct MergePacksAlias;

template < typename... Args1, class C, typename... Args2 >
struct MergePacksAlias<univpack<C,Args1...>,univpack<Args2...>>
{
	using tmp = typename MergeInPackAlias<C,univpack<Args2...>>::type;
	using type = typename MergePacksAlias<univpack<Args1...>,tmp>::type;
};

template < typename... Args2 >
struct MergePacksAlias<univpack<>,univpack<Args2...>>
{
	using type = univpack<Args2...>;
};

template < class PACK1, class PACK2 >
using MergePacks = typename MergePacksAlias<PACK1,PACK2>::type;

	

template < class UPACK >
struct GetDimsAlias
{
	using a = typename UPACK::NEXT;
	using c = typename GetDimsAlias<a>::type;
	using type = typename c::PUTLEFT<UPACK::FIRST::DIM>;
};

template <>
struct GetDimsAlias< univpack<> >
{
	using type = pack<>;
};

template < class UPACK >
using GetDims = typename GetDimsAlias<UPACK>::type;




template < class UPACK >
struct GetIndsAlias
{
	using a = typename UPACK::NEXT;
	using c = typename GetIndsAlias<a>::type;
	using type = typename c::PUTLEFT<UPACK::FIRST::N>;
};

template <>
struct GetIndsAlias< univpack<> >
{
	using type = pack<>;
};

template < class UPACK >
using GetInds = typename GetIndsAlias<UPACK>::type;


template < class INTPACK, int N >
struct IndValAlias
{	
	static const int ind = 1+IndValAlias<typename INTPACK::NEXT,N>::ind;
};

template < int N, int... NS >
struct IndValAlias< pack<N,NS...>, N >
{	
	static const int ind = 0;
};

template < int N >
struct IndValAlias< pack<>, N >
{	
	static const int ind = 0;
};

template < class INTPACK, int N >
static int IndVal() 
{ 
	return IndValAlias<INTPACK,N>::ind;
}




template < class UPACK, int N >
struct ValAlias
{	
	using a = typename UPACK::NEXT;
	using type = typename ValAlias<a,N-1>::type;
};

template < class UPACK >
struct ValAlias< UPACK, 0 >
{
	using type = typename UPACK::FIRST;
};

template < class UPACK, int N >
using Val = typename ValAlias<UPACK,N>::type;

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
template < int... NS > struct pack
{ 
    
    // DIMSX::VAL(2) is the size of its 3rd vector (starts at 0) (in the example above, DIM). 
    // Therefore, EMPTY::VAL(n) should never be called : we return -1 as an error signal. 
    static int VAL(int m) { return -1; }
    
    // [].append(M) = [M]
    template < int M >
    using PUTLEFT = pack<M>;

    // Furthermore, the empty package :
    static const int SIZE = 0; // Has zero size (number of vectors) ...
    static const int SUM = 0;  // ... zero sum  (total memory footprint) ...
    
    // ... is loaded trivially ...
    template < typename TYPE >
    __host__ __device__ static void load(int i, TYPE* xi, TYPE** px) { }
    
    // ... counts for nothing in the evaluation of a function ...
    template < typename TYPE, class FUN, typename... Args  >
    __host__ __device__ static void call(FUN fun, TYPE* x, Args... args) { fun(args...); }
    
    // ... idem ...
    template < class DIMS, typename TYPE, class FUN, typename... Args  >
    __host__ __device__ static void call2(FUN fun, TYPE* x, Args... args) { DIMS::call(fun,args...); }
    
    // ... does not have anything to give to a list of variables.
    template < typename TYPE, typename... Args  >
    static void getlist(TYPE** px, Args... args) { }
    
};

// A non-EMPTY package, recursively defined as : ====================================================
// "The concatenation of a vector of size N, and a (possibly empty) package."
template < int N, int... NS > struct pack<N,NS...>
{
    static const int FIRST = N;    // Size of its first element.
    
    // DIMSX::VAL(2) = size of its 3rd vector (we start counting at 0).
    static int VAL(int m)
    {
        if(m)
            return NEXT::VAL(m-1);
        else
            return FIRST;
    }
    
    // Operation to append "M" at the head of the list
    template < int M >
    using PUTLEFT = pack<M, N, NS...>;
               
    static const int SIZE = 1+sizeof...(NS); // The number of vectors in pack<N,NS...>
    typedef pack<NS...> NEXT;                // "NEXT" is the tail of our list of vectors.
    static const int SUM = N + NEXT::SUM;    // The total "memory footprint" of pack<N,NS...> is computed recursively.

    // Loads the i-th element of the (global device memory pointer) px
    // to the "array" xi.
    template < typename TYPE >
    __host__ __device__ static void load(int i, TYPE* xi, TYPE** px)
    {
        /*
         * px is an "array" of pointers to data arrays of appropriate sizes.
         * That is, px[0] = *px     is a pointer to a TYPE array of size Ni * FIRST
         * Then,    px[1] = *(px+1) is a pointer to a TYPE array of size Ni * NEXT::FIRST; etc.
         * 
         * (where Ni is the max value of "i" you should expect)
         * Obviously, we do not make any sanity check... so beware of illicit memory accesses !
         */
        // Using pythonic syntax, we can describe our loading procedure as follows :
		for(int k=0; k<FIRST; k++)
			xi[k] = (*px)[i*FIRST+k];  // First, load the i-th line of px[0]  -> xi[ 0 : FIRST ].
		NEXT::load(i,xi+FIRST,px+1);   // Then,  load the i-th line of px[1:] -> xi[ FIRST : ] (recursively)
    }
    
    // call(fun, [x1, x2, x3], arg1, arg2 ) will end up executing fun( arg1, arg2, x1, x2, x3 ).
    template < typename TYPE, class FUN, typename... Args  >
    __host__ __device__ static void call(FUN fun, TYPE* x, Args... args)
    {
        NEXT::call(fun,x+FIRST,args...,x);  // Append [x[0:FIRST]] to the list of arguments, then iterate.
    }
    
    // Idem, with a template on DIMS. This allows you to call fun with
    // two "packed" variables (x_i and y_j) as first inputs.
    // call2(fun, [x1, x2], [y1, y2], arg1 ) will end up executing fun(arg1, x1, x2, y1, y2).
    template < class DIMS, typename TYPE, class FUN, typename... Args  >
    __host__ __device__ static void call2(FUN fun, TYPE* x, Args... args)
    {
        NEXT::template call2<DIMS>(fun,x+FIRST,args...,x);
    }
    
    // Out of a long  list of pointers, extract the ones which "belong" to the current pack
    // and put them into a pointer array px.
    // Since we use the std container "tuple", treatment of the list of arguments args is straightforward
    template < typename TYPE, typename... Args  >
    static void getlist(TYPE** px, Args... args)
    {
    	auto t = make_tuple(args...);
        *px = get<FIRST>(t);
        NEXT::getlist(px+1,args...);
    }
     
};

// USEFUL METHODS ===================================================================================

// Templated call
template < class DIMSX, class DIMSY, typename TYPE, class FUN, typename... Args  >
__host__ __device__ void call(FUN fun, TYPE* x, Args... args)
{
    DIMSX:: template call2<DIMSY>(fun,x,args...);
}

template < class INDS, typename TYPE, typename... Args >
void getlist(TYPE** px, Args... args)
{
    INDS::getlist(px,args...);
}

// Loads the i-th "line" of px to xi.
template < class DIMS, typename TYPE >
__host__ __device__ void load(int i, TYPE* xi, TYPE** px)
{
	DIMS::load(i,xi,px);
}



#endif
