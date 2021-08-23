/*
 * This file contains compilers macros (mostly aliases) used to defined
 * high end user friendly formulas.
 *
 */

#pragma once

namespace keops {

////////////////////////////////////////////////////////////////////////////////
//         Deprecated : (old syntax, kept for backward compatibility)         //
////////////////////////////////////////////////////////////////////////////////

#define SumReduction(F,I) KeopsNS<Sum_Reduction<decltype(InvKeopsNS(F)),I>>()
#define LogSumExpReduction(F,I) KeopsNS<Max_SumShiftExp_Reduction<decltype(InvKeopsNS(F)),I>>()
#define LogSumExpVectReduction(F,I,G) KeopsNS<Max_SumShiftExp_Reduction<decltype(InvKeopsNS(F)),I,decltype(InvKeopsNS(G))>>()

#define ArgMinReduction(F,I) KeopsNS<ArgMin_Reduction<decltype(InvKeopsNS(F)),I>>()
#define MinReduction(F,I) KeopsNS<Min_Reduction<decltype(InvKeopsNS(F)),I>>()

#define MaxArgMaxReduction(F,I) KeopsNS<Max_ArgMax_Reduction<decltype(InvKeopsNS(F)),I>>()
#define ArgMaxReduction(F,I) KeopsNS<ArgMax_Reduction<decltype(InvKeopsNS(F)),I>>()
#define MaxReduction(F,I) KeopsNS<Max_Reduction<decltype(InvKeopsNS(F)),I>>()

#define KMinArgKMinReduction(F,K,I) KeopsNS<KMin_ArgKMin_Reduction<decltype(InvKeopsNS(F)),K,I>>()
#define ArgKMinReduction(F,K,I) KeopsNS<ArgKMin_Reduction<decltype(InvKeopsNS(F)),K,I>>()
#define KMinReduction(F,K,I) KeopsNS<KMin_Reduction<decltype(InvKeopsNS(F)),K,I>>()

#define Vx(N,DIM) KeopsNS<_X<N,DIM>>()
#define Vy(N,DIM) KeopsNS<_Y<N,DIM>>()

}
