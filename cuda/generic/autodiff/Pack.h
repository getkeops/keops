#ifndef PACK
#define PACK

#include <tuple>

using namespace std;

template < int... NS > struct pack;

template < typename... Args >
struct univpack
{
    using FIRST = void;
    
	static const int SIZE = 0;
	
    template < class D >
    using PUTLEFT = D;
	
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
    
    template < typename TYPE, typename... Args  >
    static void getlist_new(TYPE** px, Args... args) { }
    
    template < typename TYPE, typename... Args  >
    static void getlist_dispatch(TYPE** px, TYPE** py, Args... args) { }
    
    template < class DIMS, typename TYPE, typename... Args  >
    static void getlist_delayed(TYPE** px, Args... args) { DIMS::getlist(px,args...); }
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
    static void getlist(TYPE** px, TYPE* x, Args... args)
    {
        *px = x;
        NEXT::getlist(px+1,args...);
    }
     
    template < typename TYPE, typename... Args  >
    static void getlist_new(TYPE** px, Args... args)
    {
    	auto t = make_tuple(args...);
        *px = get<FIRST>(t);
        NEXT::getlist_new(px+1,args...);
    }
     
    template < typename TYPE, typename... Args  >
    static void getlist_dispatch(TYPE** px, TYPE** py, TYPE* x, Args... args)
    {
    	if(N>0)
    	{
	        *px = x;
	        NEXT::getlist_dispatch(px+1,py,args...);
	    }
	    else
	    {
	    	*py = x;
	    	NEXT::getlist_dispatch(px,py+1,args...);
	    }       
    }
     
    template < class DIMS, typename TYPE, typename... Args  >
    static void getlist_delayed(TYPE** px, TYPE* y, Args... args)
    {
        NEXT::template getlist_delayed<DIMS>(px,args...);
    }
};

template < class DIMSX, class DIMSY, typename TYPE, class FUN, typename... Args  >
__host__ __device__ void call(FUN fun, TYPE* x, Args... args)
{
    DIMSX:: template call2<DIMSY>(fun,x,args...);
}

template < class DIMS, typename TYPE, typename... Args >
void getlist(TYPE** px, Args... args)
{
    DIMS::getlist(px,args...);
}

template < class INDS, typename TYPE, typename... Args >
void getlist_new(TYPE** px, Args... args)
{
    INDS::getlist_new(px,args...);
}

template < class DIMS, typename TYPE, typename... Args >
void getlist_dispatch(TYPE** px, TYPE** py, Args... args)
{
    DIMS::getlist_dispatch(px,py,args...);
}

template < class DIMSX, class DIMSY, typename TYPE, typename... Args  >
static void getlist_delayed(TYPE** px, Args... args)
{
    DIMSX::template getlist_delayed<DIMSY>(px,args...);
}

template < class DIMS, typename TYPE >
__host__ __device__ void load(int i, TYPE* xi, TYPE** px)
{
	DIMS::load(i,xi,px);
}



#endif
