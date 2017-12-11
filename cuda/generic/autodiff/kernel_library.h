// Aliases for custom kernels


// Scalar radial kernels

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


// Matrix-valued kernels

//template < template<class,class> class FORTHO, template<class,class> class FTILDE, int DIMPOINT, int DIMVECT >
//using TRI_Kernel = Add<Scal<FORTHO<Param<0>,SqDist<X<0,DIMPOINT>,Y<1,DIMPOINT>>>,X<2,DIMVECT>>,Scal<Scal<FTILDE<Param<1>,SqDist<X<0,DIMPOINT>,Y<1,DIMPOINT>>>,Scalprod<X<2,DIMVECT>,Subtract<X<0,DIMPOINT>,Y<1,DIMPOINT>>>>,Subtract<X<0,DIMPOINT>,Y<1,DIMPOINT>>>;

//DivFreeFunctionOrtho = F<OOS2,R2>::DiffT
