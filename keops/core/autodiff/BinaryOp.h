#pragma once

#include <sstream>

#include "core/pack/CondType.h"
#include "core/pack/UnivPack.h"
#include "core/pack/MergePacks.h"
#include "core/pack/IndVal.h"
#include "core/autodiff/Var.h"

namespace keops {


// binary operator class : common methods

// binary operators are of the type OP<F,G> : for example Add<F,G>, Mult<F,G>
// There template parameters are two sub-formulas FA and FB and PARAMS... is a pack of index_sequences containing the
// needed parameters. As for now it is used
template < template<class,class,class...> class OP, class FA, class FB, class... PARAMS >
struct BinaryOp_base {

  using THIS = OP<FA,FB,PARAMS...>;

  // recursive function to print the formula as a string
  static void PrintId(::std::stringstream& str) {
    str << "(";                                  // prints "("
    FA::PrintId(str);                            // prints the formula FA
    THIS::PrintIdString(str);                    // prints the id string of the operator : "+", "*", ...
    FB::PrintId(str);                            // prints the formula FB
    univpack<PARAMS...>::PrintAllIndexSequence(str);
    str << ")";                                  // prints ")"
  }

  static void PrintFactorized(::std::stringstream& str) {
    PrintId(str);
  }

  // AllTypes is a tuple of types which gives all sub-formulas in a formula (including the formula itself)
  // for example Add<Var<0,2,0>,Var<1,2,1>>::AllTypes is :
  // univpack< Add<Var<0,2,0>,Var<1,2,1>> , Var<0,2,0> , Var<1,2,2> >
  using AllTypes = MergePacks<univpack<OP<FA,FB,PARAMS...>>,MergePacks<typename FA::AllTypes,typename FB::AllTypes>>;

  // "Replace" can be used to replace any occurrence of a sub-formula in a formula
  // For example Add<Var<0,2,0>,Var<1,2,1>>::Replace<Var<1,2,1>,Var<1,2,0>> will be Add<Var<0,2,0>,Var<1,2,0>>
  template<class A, class B>
  using Replace = CondType< B, OP<typename FA::template Replace<A,B>,typename FB::template Replace<A,B>, PARAMS...>, IsSameType<A,THIS>::val >;

  // VARS gives the list of all "Vars" of a given category inside a formula
  // Here we must take the union of Vars that are inside FA and Vars that are inside FB
  template < int CAT >
  using VARS = MergePacks<typename FA::template VARS<CAT>,typename FB::template VARS<CAT>>;

};


// binary operator class : default Eval method
template <template<class,class,class...> class OP, class FA, class FB, class... PARAMS >
struct BinaryOp : BinaryOp_base<OP,FA,FB,PARAMS...> {

  using THIS = OP<FA,FB,PARAMS...>;

  template < class INDS, typename... ARGS >
  static HOST_DEVICE INLINE void Eval(__TYPE__* out, ARGS... args) {
    // we create vectors of sizes FA::DIM and FB::DIM
    __TYPE__ outA[FA::DIM], outB[FB::DIM];
    // then we call the Eval function of FA and FB
    FA::template Eval<INDS>(outA,args...);
    FB::template Eval<INDS>(outB,args...);
    // then we call the Operation function
    THIS::Operation(out,outA,outB);
  }
};

// specialization when left template is of type Var
template < template<class,class,class...> class OP, int N, int DIM, int CAT, class FB, class... PARAMS >
struct BinaryOp<OP,Var<N,DIM,CAT>,FB,PARAMS...>  : BinaryOp_base<OP,Var<N,DIM,CAT>,FB,PARAMS...> {

using THIS = OP<Var<N,DIM,CAT>,FB,PARAMS...>;

template < class INDS, typename... ARGS >
static HOST_DEVICE INLINE void Eval(__TYPE__* out, ARGS... args) {
  // we create a vector and call Eval only for FB
  __TYPE__ outB[FB::DIM];
  FB::template Eval<INDS>(outB,args...);
  // access the Nth argument of args
  __TYPE__* outA = Get<IndVal_Alias<INDS,N>::ind>(args...); // outA = the "ind"-th argument.
  // then we call the Operation function
  THIS::Operation(out,outA,outB);
}
};

// specialization when right template is of type Var
template < template<class,class,class...> class OP, class FA, int N, int DIM, int CAT, class... PARAMS >
struct BinaryOp<OP,FA,Var<N,DIM,CAT>,PARAMS...>  : BinaryOp_base<OP,FA,Var<N,DIM,CAT>,PARAMS...> {

using THIS = OP<FA,Var<N,DIM,CAT>,PARAMS...>;

template < class INDS, typename... ARGS >
static HOST_DEVICE INLINE void Eval(__TYPE__* out, ARGS... args) {
  // we create a vector and call Eval only for FA
  __TYPE__ outA[FA::DIM];
  FA::template Eval<INDS>(outA,args...);
  // access the Nth argument of args
  __TYPE__* outB = Get<IndVal_Alias<INDS,N>::ind>(args...); // outB = the "ind"-th argument.
  // then we call the Operation function
  THIS::Operation(out,outA,outB);
}
};

// specialization when both templates are of type Var
template < template<class,class, class...> class OP, int NA, int DIMA, int CATA, int NB, int DIMB, int CATB, class... PARAMS>
struct BinaryOp<OP,Var<NA,DIMA,CATA>,Var<NB,DIMB,CATB>,PARAMS...> :
BinaryOp_base<OP,Var<NA,DIMA,CATA>,Var<NB,DIMB,CATB>,PARAMS...> {

using THIS = OP<Var<NA,DIMA,CATA>,Var<NB,DIMB,CATB>,PARAMS...>;

template < class INDS, typename... ARGS >
static HOST_DEVICE INLINE void Eval(__TYPE__* out, ARGS... args) {
  // we access the NAth and NBth arguments of args
  __TYPE__* outA = Get<IndVal_Alias<INDS,NA>::ind>(args...);
  __TYPE__* outB = Get<IndVal_Alias<INDS,NB>::ind>(args...);
  // then we call the Operation function
  THIS::Operation(out,outA,outB);
}
};


// iterate binary operator

template < template<class,class> class OP, class PACK >
struct IterBinaryOp_Impl {
  using type = OP<typename PACK::FIRST,typename IterBinaryOp_Impl<OP,typename PACK::NEXT>::type>;
};

template < template<class,class> class OP, class F >
struct IterBinaryOp_Impl<OP,univpack<F>> {
using type = F;
};

template < template<class,class> class OP, class PACK >
using IterBinaryOp = typename IterBinaryOp_Impl<OP,PACK>::type;




/*
 *  // iterate binary operator
 *
 *template < template<class,class> class OP, class PACK >
 *struct IterBinaryOp_Impl {
 *    using type = OP<typename PACK::FIRST,typename IterBinaryOp_Impl<OP,typename PACK::NEXT>::type>;
 *};
 *
 *template < template<class,class> class OP, class F >
 *struct IterBinaryOp_Impl<OP,univpack<F>> {
 *    using type = F;
 *};
 *
 *template < template<class,class> class OP, class PACK >
 *using IterBinaryOp = typename IterBinaryOp_Impl<OP,PACK>::type;
 */





}
