/*
 * The file where the elementary math operators are defined.
 * Available math operations are :
 *
 *   +, *, - :
 *      Add<FA,FB>                  : adds FA and FB functions
 *      Subtract<FA,FB>             : subtractss FA and FB functions
 *      Scal<FA,FB>                 : product of FA (scalar valued) with FB
 *      Mult<FA,FB>                 : element-wise multiplication of FA and FB
 *      Minus<F>                    : alias for Scal<IntConstant<-1>,F>
 *
 *   /, ^, ^2, ^-1, ^(1/2) :
 *      Divide<FA,FB>               : alias for Scal<FA,Inv<FB>>
 *      Pow<F,M>                    : Mth power of F (vectorized) ; M is an integer
 *      Powf<A,B>                   : alias for Exp<Scal<FB,Log<FA>>>
 *      Square<F>                   : Pointwise square, more efficient than Pow<F,2>
 *      Inv<F>                      : Pointwise inverse, more efficient than Pow<F,-1>
 *      IntInv<N>                   : alias for Inv<IntConstant<N>>
 *      Sqrt<F>                     : alias for Powf<F,IntInv<2>>
 *      Rsqrt<F>                    : inverse square root
 *
 *   standard math functions :
 *      Sum<F>						: sum of values in F
 *      Abs<F>						: absolute value of F (vectorized)
 *      Exp<F>                      : exponential of F (vectorized)
 *      Log<F>                      : logarithm   of F (vectorized)
 *      Sin<F>                      : sine        of F (vectorized)
 *      Cos<F>                      : cosine      of F (vectorized)
 *      Sign<F>                     : sign        of F (vectorized)
 *      Step<F>                     : step        of F (vectorized)
 *      ReLU<F>                     : ReLU        of F (vectorized)
 *      Sign<F>                     : sign        of F (vectorized)
 *
 *   concatenation and matrix-vector products:
 *      Concat<FA,FB>               : concatenation of FB and FB
 *      MatVecMult<FA,FB>           : matrix-vector product (FA::DIM must be a muliple of FB::DIM)
 *      VecMatMult<FA,FB>           : vector-matrix product (FB::DIM must be a muliple of FA::DIM)
 *      TensorProd<FA,FB>           : tensor product (output is of dimension FA::DIM*FB::DIM)
 *      TensorDot<FA,FB, I>         : tensor dot as in numpy (FA::DIM must be a muliple of FB::DIM) I is an IntCst()
 *
 */