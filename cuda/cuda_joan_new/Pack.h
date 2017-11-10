#ifndef PACK
#define PACK

using namespace std;

template < int... NS > struct pack
{ 
    static int VAL(int m)
    {
        return -1;
    }

    static const int SIZE = 0;
    
    static const int SUM = 0;
    
    template < typename TYPE >
    __host__ __device__ static void load(int i, TYPE* xi, TYPE** px) { }
    
    template < typename TYPE, class FUN, typename... Args  >
    __host__ __device__ static void call(FUN fun, TYPE* x, Args... args) { fun(args...); }
    
    template < class DIMS, typename TYPE, class FUN, typename... Args  >
    __host__ __device__ static void call(FUN fun, TYPE* x, Args... args) { DIMS::call(fun,args...); }
    
    template < typename TYPE, typename... Args  >
    static void getlist(TYPE** px, Args... args) { }
    
    template < class DIMS, typename TYPE, typename... Args  >
    static void getlist(TYPE** px, Args... args) { DIMS::getlist(px,args...); }
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
    __host__ __device__ static void call(FUN fun, TYPE* x, Args... args)
    {
        NEXT::template call<DIMS>(fun,x+FIRST,args...,x);
    }
    
    template < typename TYPE, typename... Args  >
    static void getlist(TYPE** px, TYPE* x, Args... args)
    {
        *px = x;
        NEXT::getlist(px+1,args...);
    }
    
    template < class DIMS, typename TYPE, typename... Args  >
    static void getlist(TYPE** px, TYPE* x, Args... args)
    {
        NEXT::template getlist<DIMS>(px,x,args...);
    }
};

#endif
