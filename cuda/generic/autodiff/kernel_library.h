// Aliases for custom kernels

// 1. stuff that should go in autodiff.h :

template < class F >
using Inv = Pow<F,-1>;

template < int N >
using IntInv = Inv<IntConstant<N>>;

template < class FA, class FB >
using Divide = Scal<FA,Inv<FB>>;

template < class X, class Y >
using SqDist = SqNorm2<Subtract<X,Y>>;

template < class F >
struct Log
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
		*out = log(*outF);		
	}

	template < class V, class GRADIN >
	using DiffTF = typename F::template DiffT<V,GRADIN>;
		
	template < class V, class GRADIN >
	using DiffT = Scal<Inv<F>,DiffTF<V,GRADIN>>;
};

template < class FA, class FB >
using Powf = Exp<Scal<FB,Log<FA>>>;

template < class F >
using Sqrt = Powf<F,IntInv<2>>;



// 2. Scalar radial kernels

template < template<class,class> class F, int DIMPOINT, int DIMVECT >
using ScalarRadialKernel = Scal<F<Param<0>,SqDist<X<0,DIMPOINT>,Y<1,DIMPOINT>>>,Y<2,DIMVECT>>;

template < class OOS2, class R2 >
using GaussFunction = Exp<Scal<Constant<OOS2>,Minus<R2>>>;

template < class OOS2, class R2 >
using LaplaceFunction = Exp<Minus<Sqrt<Add<Inv<Constant<OOS2>>,R2>>>>;

template < class OOS2, class R2 >
using EnergyFunction = Inv<Powf<Add<Inv<Constant<OOS2>>,R2>,IntInv<4>>>;

template < int DIMPOINT, int DIMVECT >
using GaussKernel = ScalarRadialKernel<GaussFunction,DIMPOINT,DIMVECT>;

template < int DIMPOINT, int DIMVECT >
using LaplaceKernel = ScalarRadialKernel<LaplaceFunction,DIMPOINT,DIMVECT>;

template < int DIMPOINT, int DIMVECT >
using EnergyKernel = ScalarRadialKernel<EnergyFunction,DIMPOINT,DIMVECT>;


// 3. Matrix-valued kernels

//template < template<class,class> class FORTHO, template<class,class> class FTILDE, int DIMPOINT, int DIMVECT >
//using TRI_Kernel = Add<Scal<FORTHO<Param<0>,SqDist<X<0,DIMPOINT>,Y<1,DIMPOINT>>>,X<2,DIMVECT>>,Scal<Scal<FTILDE<Param<1>,SqDist<X<0,DIMPOINT>,Y<1,DIMPOINT>>>,Scalprod<X<2,DIMVECT>,Subtract<X<0,DIMPOINT>,Y<1,DIMPOINT>>>>,Subtract<X<0,DIMPOINT>,Y<1,DIMPOINT>>>;

//DivFreeFunctionOrtho = F<OOS2,R2>::DiffT
