The files where the elementary math operators are defined.

Available math operations are :
 *      Add<FA,FB>                     : adds FA and FB functions
 *      Subtract<FA,FB>                : subtracts FA and FB functions
 *      Scal<FA,FB>                    : product of FA (scalar valued) with FB
 *      Mult<FA,FB>                    : element-wise multiplication of FA and FB
 *      Minus<F>                       : alias for Scal<IntConstant<-1>,F>
 *      Divide<FA,FB>                  : alias for Scal<FA,Inv<FB>>
 *      Pow<F,M>                       : Mth power of F (vectorized) ; M is an integer
 *      Powf<A,B>                      : alias for Exp<Scal<FB,Log<FA>>>
 *      Square<F>                      : Pointwise square, more efficient than Pow<F,2>
 *      Inv<F>                         : Pointwise inverse, more efficient than Pow<F,-1>
 *      IntInv<N>                      : alias for Inv<IntConstant<N>>
 *      Sqrt<F>                        : alias for Powf<F,IntInv<2>>
 *      Rsqrt<F>                       : inverse square root

Standard math functions :   
 *      Sum<F>                         : sum of values in F
 *      Max<F>                         : max of values in F
 *      Min<F>                         : min of values in F
 *      ArgMax<F>                      : argmax of values in F
 *      ArgMin<F>                      : argmin of values in F
 *      Abs<F>                         : absolute value of F (vectorized)
 *      Exp<F>                         : exponential of F (vectorized)
 *      Log<F>                         : logarithm of F (vectorized)
 *      XLogX<F>                       : function F*log(F) (vectorized)
 *      Sin<F>                         : sine of F (vectorized)
 *      Cos<F>                         : cosine of F (vectorized)
 *      Acos<F>                        : arc-cosine of F (vectorized)
 *      Asin<F>                        : arc-sine of F (vectorized)
 *      Atan<F>                        : arc-tangent of F (vectorized)
 *      Sign<F>                        : sign of F (vectorized)
 *      Step<F>                        : step of F (vectorized)
 *      ReLU<F>                        : ReLU of F (vectorized)
 *      Sign<F>                        : sign of F (vectorized)
 *      ClampInt<F,A,B>                : clamping of F in [A,B] (vectorized) ; F is function, A and B are integers
 *      Clamp<F,G,H>                   : clamping of F in [G,H] (vectorized) ; F, G, H are functions
 
Concatenation and matrix-vector products:
 *      Concat<FA,FB>                  : concatenation of FB and FB
 *      Elem<F,N>                      : extract Nth element of F
 *      Extract<F,START,LENGTH>        : extract a number LENGTH starting at index START
 *      MatVecMult<FA,FB>              : matrix-vector product (FA::DIM must be a muliple of FB::DIM)
 *      OneHot<F,D>                    : represents a scalar formula (rounded to an integer) as a one-hot vector of dimension D
 *      VecMatMult<FA,FB>              : vector-matrix product (FB::DIM must be a muliple of FA::DIM)
 *      TensorProd<FA,FB>              : tensor product (output is of dimension FA::DIM*FB::DIM)
 *      TensorDot<FA,FB,DA,DB,CA,CB>   : tensor dot as in numpy. FA and FB are formulas and DA, DB, CA and CB are
                                         IndexSequences that store the parameters needed to perform the tensor contraction.
                                         FA::DIM (resp. FB::DIM) must be the product of elements in DA (resp. DB). CA and
                                         CB are the index of the contracted dimensions.
