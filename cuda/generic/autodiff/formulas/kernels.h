/*
 * The file where the most useful kernel-related operators are defined.
 * Available kernel operations are :
 * 
 *      GaussKernel<OOS2,X,Y,Beta>	: Gaussian kernel, OOS2 = 1/s^2
 * 
 */


//////////////////////////////////////////////////////////////
//// GAUSSIAN KERNEL  : GaussKernel< OOS2, X, Y, Beta >   ////
//////////////////////////////////////////////////////////////

// GaussKernel( c, x, y, b = exp(- c * |x-y|_2^2 )  * b
template < class OOS2, class X, class Y, class Beta >
using GaussKernel = Scal<Exp<Scal<Constant<OOS2>,Minus<SqDist<X,Y>>>>,Beta>;

