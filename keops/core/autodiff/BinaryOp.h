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
// needed parameters. 
template < template<class,class,class...> class OP, class FA, class FB, class... PARAMS >
struct BinaryOp_base {

  using ARG1 = FA;
  using ARG2 = FB;

  using THIS = OP<FA,FB,PARAMS...>;

  template < int DIMCHK >
  using CHUNKED_VERSION = void;

  static const bool IS_CHUNKABLE = false;

  template < int DIMCHK >
  using CHUNKED_FORMULAS = ConcatPacks < typename FA::template CHUNKED_FORMULAS<DIMCHK>, typename FB::template CHUNKED_FORMULAS<DIMCHK> >;

  static const int NUM_CHUNKED_FORMULAS = FA::NUM_CHUNKED_FORMULAS + FB::NUM_CHUNKED_FORMULAS;

  // NB. The following commented code should be ok but it dos not compile with Cuda 11 as of 2020 aug 13th...
  /*
  template < int IND >
  using POST_CHUNK_FORMULA = OP < typename FA::template POST_CHUNK_FORMULA<IND>, 
  * 								typename FB::template POST_CHUNK_FORMULA<IND+FA::NUM_CHUNKED_FORMULAS>, 
  * 								PARAMS... >;
  */
  // ... so we use an additional "_Impl" structure to specialize in case of empty PARAMS pack : 
  template < int IND, int SIZE_PARAMS >
  struct POST_CHUNK_FORMULA_Impl {
	  using type = OP < typename FA::template POST_CHUNK_FORMULA<IND>, 
   								typename FB::template POST_CHUNK_FORMULA<IND+FA::NUM_CHUNKED_FORMULAS>, 
   								PARAMS... >;
  };
  
  template < int IND >
  struct POST_CHUNK_FORMULA_Impl < IND, 0 > {
	  using type = OP < typename FA::template POST_CHUNK_FORMULA<IND>, 
   								typename FB::template POST_CHUNK_FORMULA<IND+FA::NUM_CHUNKED_FORMULAS> >;
  };
  
  template < int IND >
  using POST_CHUNK_FORMULA = typename POST_CHUNK_FORMULA_Impl<IND,sizeof...(PARAMS)>::type;
  
  
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
  // N.B we comment out AutoFactorize and AllTypes in all code as of oct 2020 to speed up compile time
  // using AllTypes = MergePacks<univpack<OP<FA,FB,PARAMS...>>,MergePacks<typename FA::AllTypes,typename FB::AllTypes>>;

  // "Replace" can be used to replace any occurrence of a sub-formula in a formula
  // For example Add<Var<0,2,0>,Var<1,2,1>>::Replace<Var<1,2,1>,Var<1,2,0>> will be Add<Var<0,2,0>,Var<1,2,0>>

  // NB. The following commented code should be ok but it dos not compile with Cuda 11 as of 2020 aug 13th...
  /*
  template<class A, class B>
  using Replace = CondType< B, OP<typename FA::template Replace<A,B>,typename FB::template Replace<A,B>, PARAMS...>, IsSameType<A,THIS>::val >;
  */
  
  // ... so we use an additional "_Impl" structure to specialize in case of empty PARAMS pack :
  template < class A, class B, int SIZE_PARAMS >
  struct Replace_Impl {
    using type = CondType < B, OP < typename FA::template Replace<A,B>,
        typename FB::template Replace<A,B> , PARAMS... >, IsSameType<A,THIS>::val >;
  };

  template < class A, class B >
  struct Replace_Impl<A,B,0> {
    using type = CondType < B, OP < typename FA::template Replace<A,B>,
        typename FB::template Replace<A,B> >, IsSameType<A,THIS>::val >;
  };

  template < class A, class B >
  using Replace = typename Replace_Impl<A,B,sizeof...(PARAMS)>::type;
  
  
  
  // version with two replacements of Vars at a time (two consecutive Replace might not work because of non compatible dimensions)
  
  // NB. The following commented code should be ok but it dos not compile with Cuda 11 as of 2020 aug 13th...
  /*
  template<class A1, class B1, class A2, class B2>
  using ReplaceVars2 = OP<typename FA::template ReplaceVars2<A1,B1,A2,B2>,typename FB::template ReplaceVars2<A1,B1,A2,B2>, PARAMS...>;
  */
  
  // ... so we use an additional "_Impl" structure to specialize in case of empty PARAMS pack :  
  template<class A1, class B1, class A2, class B2, int SIZE_PARAMS>
  struct ReplaceVars2_Impl { 
    using type = OP<typename FA::template ReplaceVars2<A1,B1,A2,B2>,
                        typename FB::template ReplaceVars2<A1,B1,A2,B2>, PARAMS...>;
  };

  template<class A1, class B1, class A2, class B2>
  struct ReplaceVars2_Impl<A1,B1,A2,B2,0> {
    using type = OP<typename FA::template ReplaceVars2<A1,B1,A2,B2>,
                        typename FB::template ReplaceVars2<A1,B1,A2,B2>>;
  };

  template<class A1, class B1, class A2, class B2>
  using ReplaceVars2 = typename ReplaceVars2_Impl<A1,B1,A2,B2,sizeof...(PARAMS)>::type;



  // VARS gives the list of all "Vars" of a given category inside a formula
  // Here we must take the union of Vars that are inside FA and Vars that are inside FB
  template < int CAT >
  using VARS = MergePacks<typename FA::template VARS<CAT>,typename FB::template VARS<CAT>>;

  template < int CAT >
  using CHUNKED_VARS = univpack<>;

  template < int CAT >
  using NOTCHUNKED_VARS = univpack<>;
  
  // operator as shortcut to Eval...
  template < typename INDS >
  struct EvalFun {
	template < typename... Args >
	DEVICE INLINE void operator()(Args... args) {
		THIS::template Eval<INDS>(args...);
	}
  };

};


// binary operator class : default Eval method
template <template<class,class,class...> class OP, class FA, class FB, class... PARAMS >
struct BinaryOp : BinaryOp_base<OP,FA,FB,PARAMS...> {

  using THIS = OP<FA,FB,PARAMS...>;

  template < class INDS, typename TYPE, typename... ARGS >
  static HOST_DEVICE INLINE void Eval(TYPE* out, ARGS... args) {
    // we create vectors of sizes FA::DIM and FB::DIM
    TYPE outA[FA::DIM], outB[FB::DIM];
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

	template < class INDS, typename TYPE, typename... ARGS >
	static HOST_DEVICE INLINE void Eval(TYPE *out, ARGS... args) {
	  // we create a vector and call Eval only for FB
	  TYPE outB[FB::DIM];
	  FB::template Eval<INDS>(outB,args...);
	  // access the Nth argument of args
	  TYPE *outA = Get<IndVal_Alias<INDS,N>::ind>(args...); // outA = the "ind"-th argument.
	  // then we call the Operation function
	  THIS::Operation(out,outA,outB);
	}
};

// specialization when right template is of type Var
template < template<class,class,class...> class OP, class FA, int N, int DIM, int CAT, class... PARAMS >
struct BinaryOp<OP,FA,Var<N,DIM,CAT>,PARAMS...>  : BinaryOp_base<OP,FA,Var<N,DIM,CAT>,PARAMS...> {

	using THIS = OP<FA,Var<N,DIM,CAT>,PARAMS...>;

	template < class INDS, typename TYPE, typename... ARGS >
	static HOST_DEVICE INLINE void Eval(TYPE *out, ARGS... args) {
	  // we create a vector and call Eval only for FA
	  TYPE outA[FA::DIM];
	  FA::template Eval<INDS>(outA,args...);
	  // access the Nth argument of args
	  TYPE *outB = Get<IndVal_Alias<INDS,N>::ind>(args...); // outB = the "ind"-th argument.
	  // then we call the Operation function
	  THIS::Operation(out,outA,outB);
	}
};

// specialization when both templates are of type Var
template < template<class,class, class...> class OP, int NA, int DIMA, int CATA, int NB, int DIMB, int CATB, class... PARAMS>
struct BinaryOp<OP,Var<NA,DIMA,CATA>,Var<NB,DIMB,CATB>,PARAMS...> :
BinaryOp_base<OP,Var<NA,DIMA,CATA>,Var<NB,DIMB,CATB>,PARAMS...> {

	using THIS = OP<Var<NA,DIMA,CATA>,Var<NB,DIMB,CATB>,PARAMS...>;

	template < class INDS, typename TYPE, typename... ARGS >
	static HOST_DEVICE INLINE void Eval(TYPE *out, ARGS... args) {
	  // we access the NAth and NBth arguments of args
	  TYPE *outA = Get<IndVal_Alias<INDS,NA>::ind>(args...);
	  TYPE *outB = Get<IndVal_Alias<INDS,NB>::ind>(args...);
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
