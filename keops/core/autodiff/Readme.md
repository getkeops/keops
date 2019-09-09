/*
 * The files where the elementary operators are defined.
 *
 * The core operators of our engine are :
 *      Var<N,DIM,CAT>           : the N-th variable, a vector of dimension DIM,
 *                                 with CAT = 0 (i-variable), 1 (j-variable) or 2 (parameter)
 *      Grad<F,V,GRADIN>         : gradient (in fact transpose of diff op) of F with respect to variable V, applied to GRADIN
 *      _P<N>, or Param<N>       : the N-th parameter variable
 *      _X<N,DIM>                : the N-th variable, vector of dimension DIM, CAT = 0
 *      _Y<N,DIM>                : the N-th variable, vector of dimension DIM, CAT = 1
 *      Elem<F,N>                : Extract Nth element of F
 *
 *
 * Available constants are :
 *
 *      Zero<DIM>                : zero-valued vector of dimension DIM
 *      IntConstant<N>           : constant integer function with value N
 *
 * Available math operations are :
 *
 *   +, *, - :
 *      Add<FA,FB>               : adds FA and FB functions
 *      Scal<FA,FB>              : product of FA (scalar valued) with FB
 *      Minus<F>                 : alias for Scal<IntConstant<-1>,F>
 *      Subtract<FA,FB>          : alias for Add<FA,Minus<FB>>
 *
 *   /, ^, ^2, ^-1, ^(1/2) :
 *      Divide<FA,FB>            : alias for Scal<FA,Inv<FB>>
 *      Pow<F,M>                 : Mth power of F (scalar valued) ; M is an integer
 *      Powf<A,B>                : alias for Exp<Scal<FB,Log<FA>>>
 *      Square<F>                : alias for Pow<F,2>
 *      Inv<F>                   : alias for Pow<F,-1>
 *      IntInv<N>                : alias for Inv<IntConstant<N>>
 *      Sqrt<F>                  : alias for Powf<F,IntInv<2>>
 *
 *   exp, log :
 *      Exp<F>                   : exponential of F (scalar valued)
 *      Log<F>                   : logarithm   of F (scalar valued)
 *
 * Available norms and scalar products are :
 *
 *   < .,. >, | . |^2, | .-. |^2 :
 *      Scalprod<FA,FB>          : scalar product between FA and FB
 *      SqNorm2<F>               : alias for Scalprod<F,F>
 *      SqDist<A,B>              : alias for SqNorm2<Subtract<A,B>>
 *
 * Available kernel operations are :
 *
 *      GaussKernel<OOS2,X,Y,Beta>    : Gaussian kernel, OOS2 = 1/s^2
 *
 */
