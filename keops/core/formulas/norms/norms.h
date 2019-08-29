#pragma once

/*
 * The file where the elementary norm-related operators are defined.
 * Available norms and scalar products are :
 *
 *   (.|.), |.|, |.|^2, |.-.|^2 :
 *      Scalprod<FA,FB>             : scalar product between FA and FB
 *      SqNorm2<F>                  : alias for Scalprod<F,F>
 *      Norm2<F>                    : alias for Sqrt<SqNorm2<F>>
 *      SqDist<FA,FB>               : alias for SqNorm2<Subtract<FA,FB>>
 *   Non-standard norms :
 *      WeightedSqNorm<A,F>         : squared weighted norm of F, either :
 *                                       - a * sum_k f_k^2 if A::DIM=1
 *                                       - sum_k a_k f_k^2 if A::DIM=F::DIM
 *                                       - sum_kl a_kl f_k f_l if A::DIM=F::DIM^2
 *      WeightedSqDist<A,FA,FB>     : alias for WeightedSqNorm<A,Subtract<FA,FB>>
 *
 */

#include "core/formulas/norms/Norm2.h"
#include "core/formulas/norms/Normalize.h"
#include "core/formulas/norms/Scalprod.h"
#include "core/formulas/norms/SqDist.h"
#include "core/formulas/norms/SqNorm2.h"
#include "core/formulas/norms/SqNormDiag.h"
#include "core/formulas/norms/SqNormIso.h"
#include "core/formulas/norms/WeightedSqDist.h"
#include "core/formulas/norms/WeightedSqNorm.h"