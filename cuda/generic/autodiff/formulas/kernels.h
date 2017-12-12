//////////////////////////////////////////////////////////////
//// GAUSSIAN KERNEL  : GaussKernel< OOS2, X, Y, Beta >   ////
//////////////////////////////////////////////////////////////

// GaussKernel( c, x, y, b = exp(- c * |x-y|_2^2 )  * b
template < class OOS2, class X, class Y, class Beta >
using GaussKernel = Scal<Exp<Scal<Constant<OOS2>,Minus<SqNorm2<Subtract<X,Y>>>>>,Beta>;

