
#define INLINE static __host__ __device__ __forceinline__ 
//#define INLINE static inline

#include <tuple>
#include <cmath>
#include <thrust/tuple.h>

using namespace std;


template <typename T>
static constexpr T static_max(T a, T b) 
{
    return a < b ? b : a;
}

template < int DIM > struct Zero;

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

template < class Vref, class V, class FUN >
struct IdOrZeroAlias
{
	using type = Zero<V::DIM>;
};

template < class V, class FUN >
struct IdOrZeroAlias<V,V,FUN>
{
	using type = FUN;
};

template < class Vref, class V, class FUN >
using IdOrZero = typename IdOrZeroAlias<Vref,V,FUN>::type;

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

// class for base variable
template < int _N, int _DIM, int CAT=0 >
struct Var
{
	static const int N = _N;
	static const int DIM = _DIM;
	
	template < int CAT_ >
	using VARS = CondType<univpack<Var<N,DIM>>,univpack<>,CAT==CAT_>;

	template < class INDS, typename ...ARGS >
	INLINE void Eval(float* params, float* out, ARGS... args)
	{
		auto t = thrust::make_tuple(args...);
		float* xi = thrust::get<IndValAlias<INDS,N>::ind>(t);
		for(int k=0; k<DIM; k++)
			out[k] = xi[k];
	}

	template < class V, class GRADIN >
	using DiffT = IdOrZero<Var<N,DIM,CAT>,V,GRADIN>;
			
};

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

template < int _DIM >
struct Zero
{
	
	static const int DIM = _DIM;

	template < int CAT >
	using VARS = univpack<>;

	template < class INDS, typename... ARGS >
	INLINE void Eval(float* params, float* out, ARGS... args)
	{	
		for(int k=0; k<DIM; k++)	
			out[k] = 0;		
	}
		
	template < class V, class GRADIN >
	using DiffT = Zero<V::DIM>;
	
};

template < int N >
struct IntConstant
{	
	static const int DIM = 1;

	template < int CAT >
	using VARS = univpack<>;

	template < class INDS, typename... ARGS >
	INLINE void Eval(float* params, float* out, ARGS... args)
	{	
		*out = N;		
	}
		
	template < class V, class GRADIN >
	using DiffT = Zero<V::DIM>;
};



template < class PRM >
struct Constant
{	
	static const int DIM = 1;

	template < int CAT >
	using VARS = CondType<univpack<PRM>,univpack<>,CAT==2>;

	template < class INDS, typename... ARGS >
	INLINE void Eval(float* params, float* out, ARGS... args)
	{	
		*out = params[PRM::INDEX];		
	}
		
	template < class V, class GRADIN >
	using DiffT = Zero<V::DIM>;
};


//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////


template < class FA, class FB > struct AddAlias;
template < class FA, class FB > struct ScalprodAlias;
template < class FA, class FB > struct ScalAlias;
template < class F > struct Norm2Alias;

template < class FA, class FB >
using Add = typename AddAlias<FA,FB>::type;

template < class FA, class FB >
using Scalprod = typename ScalprodAlias<FA,FB>::type;

template < class FA, class FB >
using Scal = typename ScalAlias<FA,FB>::type;


//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////


template < class FA, class FB >
struct AddImpl
{	
	
	static const int DIM = FA::DIM;
	static_assert(DIM==FB::DIM,"Dimensions must be the same for Add");
	
	template < int CAT >
	using VARS = MergePacks<typename FA::VARS<CAT>,typename FB::VARS<CAT>>;

	template < class INDS, typename... ARGS >
	INLINE void Eval(float* params, float* out, ARGS... args)
	{
		float outA[DIM], outB[DIM];
		FA::template Eval<INDS>(params,outA,args...);
		FB::template Eval<INDS>(params,outB,args...);
		for(int k=0; k<DIM; k++)
			out[k] = outA[k] + outB[k];
	}

	template < class V, class GRADIN >
	using DiffTA = typename FA::template DiffT<V,GRADIN>;
	
	template < class V, class GRADIN >
	using DiffTB = typename FB::template DiffT<V,GRADIN>;
	
	template < class V, class GRADIN >
	using DiffT = Add < DiffTA<V,GRADIN> , DiffTB<V,GRADIN> >;

};

template < class FA, class FB >
struct AddAlias
{
	using type = AddImpl<FA,FB>;
};

template < class FA, int DIM >
struct AddAlias<FA,Zero<DIM>>
{
	static_assert(DIM==FA::DIM,"Dimensions must be the same for Add");
	using type = FA;
};

template < class FB, int DIM >
struct AddAlias<Zero<DIM>,FB>
{
	static_assert(DIM==FB::DIM,"Dimensions must be the same for Add");
	using type = FB;
};

template < int DIM1, int DIM2 >
struct AddAlias<Zero<DIM1>,Zero<DIM2>>
{
	static_assert(DIM1==DIM2,"Dimensions must be the same for Add");
	using type = Zero<DIM1>;
};


//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////




template < class FA, class FB >
struct ScalImpl
{
	
	static const int DIM = FB::DIM;
	static_assert(FA::DIM==1,"Dimension of FA must be 1 for Scal");

	template < int CAT >
	using VARS = MergePacks<typename FA::VARS<CAT>,typename FB::VARS<CAT>>;

	template < class INDS, typename... ARGS >
	INLINE void Eval(float* params, float* out, ARGS... args)
	{
		float outA[1], outB[DIM];
		FA::template Eval<INDS>(params,outA,args...);
		FB::template Eval<INDS>(params,outB,args...);
		for(int k=0; k<DIM; k++)
			out[k] = *outA*outB[k];
	}
	
	template < class V, class GRADIN >
	using DiffTA = typename FA::template DiffT<V,GRADIN>;
	
	template < class V, class GRADIN >
	using DiffTB = typename FB::template DiffT<V,GRADIN>;
	
	template < class V, class GRADIN >
	using DiffT = Add < DiffTA<V,Scalprod<GRADIN,FB>> , Scal < FA, DiffTB<V,GRADIN> > >;

};


template < class FA, class FB >
struct ScalAlias
{
	using type = ScalImpl<FA,FB>;
};

template < class FA, int DIM >
struct ScalAlias<FA,Zero<DIM>>
{
	static_assert(1==FA::DIM,"Dimension of FA must be 1 for Scal");
	using type = Zero<DIM>;
};

template < class FB, int DIM >
struct ScalAlias<Zero<DIM>,FB>
{
	static_assert(DIM==1,"Dimension of FA must be 1 for Scal");
	using type = Zero<FB::DIM>;
};

template < int DIM1, int DIM2 >
struct ScalAlias<Zero<DIM1>,Zero<DIM2>>
{
	static_assert(DIM1==1,"Dimension of FA must be 1 for Scal");
	using type = Zero<DIM2>;
};


//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////



template < class FA, class FB >
struct ScalprodImpl
{
	
	static const int DIMIN = FA::DIM;
	static_assert(DIMIN==FB::DIM,"Dimensions must be the same for Scalprod");
	static const int DIM = 1;

	template < int CAT >
	using VARS = MergePacks<typename FA::VARS<CAT>,typename FB::VARS<CAT>>;

	template < class INDS, typename... ARGS >
	INLINE void Eval(float* params, float* out, ARGS... args)
	{		
		*out = 0;
		float outA[DIMIN], outB[DIMIN];
		FA::template Eval<INDS>(params,outA,args...);		
		FB::template Eval<INDS>(params,outB,args...);	
		for(int k=0; k<DIMIN; k++)
			*out += outA[k]*outB[k];
	}
		
	template < class V, class GRADIN >
	using DiffTA = typename FA::template DiffT<V,GRADIN>;
	
	template < class V, class GRADIN >
	using DiffTB = typename FB::template DiffT<V,GRADIN>;
	
	template < class V, class GRADIN >
	using DiffT = Scal < GRADIN , Add < DiffTA<V,FB> , DiffTB<V,FA> > >;

};


template < class FA, class FB >
struct ScalprodAlias
{
	using type = ScalprodImpl<FA,FB>;
};

template < class FA, int DIM >
struct ScalprodAlias<FA,Zero<DIM>>
{
	static_assert(DIM==FA::DIM,"Dimensions must be the same for Scalprod");
	using type = Zero<1>;
};


//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////


template < class F >
using SqNorm2 = Scalprod<F,F>;

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////



template < class F >
struct Exp
{	
	static const int DIM = 1;
	static_assert(F::DIM==1,"Dimension of input must be one for exp function");

	template < int CAT >
	using VARS = typename F::VARS<CAT>;

	template < class INDS, typename... ARGS >
	INLINE void Eval(float* params, float* out, ARGS... args)
	{	
		float outF[1];	
		F::template Eval<INDS>(params,outF,args...);
		*out = exp(*outF);		
	}

	template < class V, class GRADIN >
	using DiffTF = typename F::template DiffT<V,GRADIN>;
		
	template < class V, class GRADIN >
	using DiffT = Scal<Exp<F>,DiffTF<V,GRADIN>>;

};


template < class F, int M >
struct Pow
{	
	static const int DIM = 1;
	static_assert(F::DIM==1,"Dimension of input must be one for exp function");

	template < int CAT >
	using VARS = typename F::VARS<CAT>;

	template < class INDS, typename... ARGS >
	INLINE void Eval(float* params, float* out, ARGS... args)
	{	
		float outF[1];	
		F::template Eval<INDS>(params,outF,args...);
		*out = pow(*outF,M);		
	}

	template < class V, class GRADIN >
	using DiffTF = typename F::template DiffT<V,GRADIN>;
		
	template < class V, class GRADIN >
	using DiffT = Scal<Scal<IntConstant<M>,Pow<F,M-1>>,DiffTF<V,GRADIN>>;

};

template < class F >
using Square = Pow<F,2>;


template < class F >
using Minus = Scal<IntConstant<-1>,F>;

template < class FA, class FB >
using Subtract = Add<FA,Minus<FB>>;



template < class OOS2, class X, class Y, class Beta >
using GaussKernel = Scal<Exp<Scal<Constant<OOS2>,Minus<SqNorm2<Subtract<X,Y>>>>>,Beta>;




template < int N >
struct Param
{
	static const int INDEX = N;
};


template < class F, class V, class GRADIN >
using Grad = typename F::template DiffT<V,GRADIN>;

