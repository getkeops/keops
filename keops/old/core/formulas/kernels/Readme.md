The files where the most useful kernel-related operators are defined.

Available kernel-related routines are :
 *   Radial functions :
 *      GaussFunction<R2,C>                     : = exp( - C * R2 )
 *      CauchyFunction<R2,C>                    : = 1/( 1 +  R2 * C )
 *      LaplaceFunction<R2,C>                   : = exp( - sqrt( C * R2 ) )
 *      InverseMultiquadricFunction<R2,C>       : = (1/C + R2)^(-1/2)

Utility functions :
 *      ScalarRadialKernel<F,DIMPOINT,DIMVECT>  : which builds a function
                                                 (x_i,y_j,b_j) -> F_s( |x_i-y_j|^2 ) * b_j from
                                                 a radial function F<S,R2> -> ...,
                                                 a "point" dimension DIMPOINT (x_i and y_j)
                                                 a "vector" dimension DIMVECT (b_j and output)

 Radial Kernel operators : inline expressions w.r.t. x_i = X_0, y_j = Y_1, b_j = Y_2
 *      GaussKernel<DIMPOINT,DIMVECT>                  : uses GaussFunction
 *      CauchyKernel<DIMPOINT,DIMVECT>                 : uses CauchyFunction
 *      LaplaceKernel<DIMPOINT,DIMVECT>                : uses LaplaceFunction
 *      InverseMultiquadricKernel<DIMPOINT,DIMVECT>    : uses InverseMultiquadricFunction
