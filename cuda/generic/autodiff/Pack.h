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



template < int... NS > struct pack
{ 
    static int VAL(int m)
    {
        return -1;
    }
    
    template < int M >
    using PUTLEFT = pack<M>;

    static const int SIZE = 0;
    
    static const int SUM = 0;
    
    template < typename TYPE >
    __host__ __device__ static void load(int i, TYPE* xi, TYPE** px) { }
    
    template < typename TYPE, class FUN, typename... Args  >
    __host__ __device__ static void call(FUN fun, TYPE* x, Args... args) { fun(args...); }

    template < class DIMS, typename TYPE, class FUN, typename... Args  >
    __host__ __device__ static void call2(FUN fun, TYPE* x, Args... args) { DIMS::call(fun,args...); }
    
    template < typename TYPE, typename... Args  >
    static void getlist(TYPE** px, Args... args) { }
    
};

template < int N, int... NS > struct pack<N,NS...>
{
    static const int FIRST = N;
    
    static int VAL(int m)
    {
        if(m)
            return NEXT::VAL(m-1);
        else
            return FIRST;
    }
                
    static const int SIZE = 1+sizeof...(NS);
    
    template < int M >
    using PUTLEFT = pack<M, N, NS...>;

    typedef pack<NS...> NEXT;
    
    static const int SUM = N + NEXT::SUM;
    
    template < typename TYPE >
    __host__ __device__ static void load(int i, TYPE* xi, TYPE** px)
    {
		for(int k=0; k<FIRST; k++)
			xi[k] = (*px)[i*FIRST+k];
		NEXT::load(i,xi+FIRST,px+1);
    }
    
    template < typename TYPE, class FUN, typename... Args  >
    __host__ __device__ static void call(FUN fun, TYPE* x, Args... args)
    {
        NEXT::call(fun,x+FIRST,args...,x);
    }
    
    template < class DIMS, typename TYPE, class FUN, typename... Args  >
    __host__ __device__ static void call2(FUN fun, TYPE* x, Args... args)
    {
        NEXT::template call2<DIMS>(fun,x+FIRST,args...,x);
    }
    
    template < typename TYPE, typename... Args  >
    static void getlist(TYPE** px, Args... args)
    {
    	auto t = make_tuple(args...);
        *px = get<FIRST>(t);
        NEXT::getlist(px+1,args...);
    }
     
};

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

template < class DIMS, typename TYPE >
__host__ __device__ void load(int i, TYPE* xi, TYPE** px)
{
	DIMS::load(i,xi,px);
}



#endif
