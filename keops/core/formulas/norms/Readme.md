The files where the elementary norm-related operators are defined.

Available norms and scalar products are :
 *      Scalprod<FA,FB>             : scalar product between FA and FB
 *      SqNorm2<F>                  : alias for Scalprod<F,F>
 *      Norm2<F>                    : alias for Sqrt<SqNorm2<F>>
 *      SqDist<FA,FB>               : alias for SqNorm2<Subtract<FA,FB>>

Non-standard norms :
 *      WeightedSqNorm<A,F>         : squared weighted norm of F, either :
                                        - a * sum_k f_k^2 if A::DIM=1
                                        - sum_k a_k f_k^2 if A::DIM=F::DIM
                                        - sum_kl a_kl f_k f_l if A::DIM=F::DIM^2
 *      WeightedSqDist<A,FA,FB>     : alias for WeightedSqNorm<A,Subtract<FA,FB>>


