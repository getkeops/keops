#pragma once

#include <sstream>

#include "core/pack/CondType.h"
#include "core/pack/UnivPack.h"
#include "core/pack/MergePacks.h"
#include "core/pack/IndVal.h"
#include "core/autodiff/Var.h"

namespace keops {


// Ternary operator class : common methods

// ternary operators are of the type OP<F,G,H>
// There template parameters are three sub-formulas FA, FB and FC and PARAMS... is a pack of index_sequences containing the
// needed parameters.
template < template<class,class,class,class...> class OP, class FA, class FB, class FC, class... PARAMS >
struct TernaryOp_base {

  using THIS = OP<FA,FB,FC,PARAMS...>;

  template < int DIMCHK >
  using CHUNKED_VERSION = void;

  static const bool IS_CHUNKABLE = false;

  template < int DIMCHK >
  using CHUNKED_FORMULAS = ConcatPacks < typename FA::template CHUNKED_FORMULAS<DIMCHK>, 
				ConcatPacks < typename FB::template CHUNKED_FORMULAS<DIMCHK>, 
						typename FC::template CHUNKED_FORMULAS<DIMCHK> > >;

  static const int NUM_CHUNKED_FORMULAS = FA::NUM_CHUNKED_FORMULAS + FB::NUM_CHUNKED_FORMULAS + FC::NUM_CHUNKED_FORMULAS;

  // NB. The following commented code should be ok but it dos not compile with Cuda 11 as of 2020 aug 13th...
  /*
  template < int IND >
  using POST_CHUNK_FORMULA = OP < typename FA::template POST_CHUNK_FORMULA<IND>, 
  * 								typename FB::template POST_CHUNK_FORMULA<IND+FA::NUM_CHUNKED_FORMULAS>, 
  * 								typename FC::template POST_CHUNK_FORMULA<IND+FA::NUM_CHUNKED_FORMULAS+FB::NUM_CHUNKED_FORMULAS>, 
  * 								PARAMS... >;
  */
  // ... so we use an additional "_Impl" structure to specialize in case of empty PARAMS pack : 
  template < int IND, int SIZE_PARAMS >
  struct POST_CHUNK_FORMULA_Impl {
	  using type = OP < typename FA::template POST_CHUNK_FORMULA<IND>, 
						typename FB::template POST_CHUNK_FORMULA<IND+FA::NUM_CHUNKED_FORMULAS>, 
						typename FC::template POST_CHUNK_FORMULA<IND+FA::NUM_CHUNKED_FORMULAS+FB::NUM_CHUNKED_FORMULAS>, 
   								PARAMS... >;
  };
  
  template < int IND >
  struct POST_CHUNK_FORMULA_Impl < IND, 0 > {
	  using type = OP < typename FA::template POST_CHUNK_FORMULA<IND>, 
   				typename FB::template POST_CHUNK_FORMULA<IND+FA::NUM_CHUNKED_FORMULAS>,
   					typename FC::template POST_CHUNK_FORMULA<IND+FA::NUM_CHUNKED_FORMULAS+FB::NUM_CHUNKED_FORMULAS> >;
  };
  
  template < int IND >
  using POST_CHUNK_FORMULA = typename POST_CHUNK_FORMULA_Impl<IND,sizeof...(PARAMS)>::type;
  
  
  // recursive function to print the formula as a string
  static void PrintId(::std::stringstream& str) {
    THIS::PrintIdString(str);                    // prints the id string of the operator
    str << "(";                                  // prints "("
    FA::PrintId(str);                            // prints the formula FA
    str << ",";                                  // prints ","
    FB::PrintId(str);                            // prints the formula FB
    str << ",";                                  // prints ","
    FC::PrintId(str);                            // prints the formula FC
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
  //using AllTypes = MergePacks<
  //	  					univpack<OP<FA,FB,FC,PARAMS...>>,
  //						MergePacks<typename FA::AllTypes,MergePacks<typename FB::AllTypes,typename FC::AllTypes>>>;

  // "Replace" can be used to replace any occurrence of a sub-formula in a formula
  // For example Add<Var<0,2,0>,Var<1,2,1>>::Replace<Var<1,2,1>,Var<1,2,0>> will be Add<Var<0,2,0>,Var<1,2,0>>

  // NB. The following commented code should be ok but it dos not compile with Cuda 11 as of 2020 aug 13th...
  /*
  template<class A, class B>
  using Replace = CondType< B, OP<typename FA::template Replace<A,B>,
								typename FB::template Replace<A,B>, 
								typename FC::template Replace<A,B>, PARAMS...>, IsSameType<A,THIS>::val >;
  */
  
  // ... so we use an additional "_Impl" structure to specialize in case of empty PARAMS pack :
  template < class A, class B, int SIZE_PARAMS >
  struct Replace_Impl {
    using type = CondType < B, OP < typename FA::template Replace<A,B>,
        							typename FB::template Replace<A,B>, 
									typename FC::template Replace<A,B>, PARAMS... >, IsSameType<A,THIS>::val >;
  };

  template < class A, class B >
  struct Replace_Impl<A,B,0> {
    using type = CondType < B, OP < typename FA::template Replace<A,B>,
        							typename FB::template Replace<A,B>,
									typename FC::template Replace<A,B> >, IsSameType<A,THIS>::val >;
  };

  template < class A, class B >
  using Replace = typename Replace_Impl<A,B,sizeof...(PARAMS)>::type;
  
  
  
  // version with two replacements of Vars at a time (two consecutive Replace might not work because of non compatible dimensions)
  
  // NB. The following commented code should be ok but it dos not compile with Cuda 11 as of 2020 aug 13th...
  /*
  template<class A1, class B1, class A2, class B2>
  using ReplaceVars2 = OP<typename FA::template ReplaceVars2<A1,B1,A2,B2>,
  							typename FB::template ReplaceVars2<A1,B1,A2,B2>, 
  								typename FB::template ReplaceVars2<A1,B1,A2,B2>, PARAMS...>;
  */
  
  // ... so we use an additional "_Impl" structure to specialize in case of empty PARAMS pack :  
  template<class A1, class B1, class A2, class B2, int SIZE_PARAMS>
  struct ReplaceVars2_Impl { 
    using type = OP<typename FA::template ReplaceVars2<A1,B1,A2,B2>,
                        typename FB::template ReplaceVars2<A1,B1,A2,B2>, 
							typename FC::template ReplaceVars2<A1,B1,A2,B2>, PARAMS...>;
  };

  template<class A1, class B1, class A2, class B2>
  struct ReplaceVars2_Impl<A1,B1,A2,B2,0> {
    using type = OP<typename FA::template ReplaceVars2<A1,B1,A2,B2>,
                        typename FB::template ReplaceVars2<A1,B1,A2,B2>,
							typename FC::template ReplaceVars2<A1,B1,A2,B2> >;
  };

  template<class A1, class B1, class A2, class B2>
  using ReplaceVars2 = typename ReplaceVars2_Impl<A1,B1,A2,B2,sizeof...(PARAMS)>::type;



  // VARS gives the list of all "Vars" of a given category inside a formula
  // Here we must take the union of Vars that are inside FA and Vars that are inside FB
  template < int CAT >
  using VARS = MergePacks<typename FA::template VARS<CAT>,
  					MergePacks<typename FB::template VARS<CAT>,
									typename FC::template VARS<CAT> > >;

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
template <template<class,class,class,class...> class OP, class FA, class FB, class FC, class... PARAMS >
struct TernaryOp : TernaryOp_base<OP,FA,FB,FC,PARAMS...> {

  using THIS = OP<FA,FB,FC,PARAMS...>;

  template < class INDS, typename TYPE, typename... ARGS >
  static HOST_DEVICE INLINE void Eval(TYPE* out, ARGS... args) {
    // we create vectors of sizes FA::DIM, FB::DIM and FC::DIM
    TYPE outA[FA::DIM], outB[FB::DIM], outC[FC::DIM];
    // then we call the Eval function of FA, FB and FC
    FA::template Eval<INDS>(outA,args...);
    FB::template Eval<INDS>(outB,args...);
    FC::template Eval<INDS>(outC,args...);
    // then we call the Operation function
    THIS::Operation(out,outA,outB,outC);
  }

};

// specialization when first template is of type Var
template < template<class,class,class,class...> class OP, int N, int DIM, int CAT, class FB, class FC, class... PARAMS >
struct TernaryOp<OP,Var<N,DIM,CAT>,FB,FC,PARAMS...>  : TernaryOp_base<OP,Var<N,DIM,CAT>,FB,FC,PARAMS...> {

	using THIS = OP<Var<N,DIM,CAT>,FB,FC,PARAMS...>;

	template < class INDS, typename TYPE, typename... ARGS >
	static HOST_DEVICE INLINE void Eval(TYPE *out, ARGS... args) {
	  // we create a vector and call Eval only for FB and FC
	  TYPE outB[FB::DIM], outC[FC::DIM];
	  FB::template Eval<INDS>(outB,args...);
	  FC::template Eval<INDS>(outC,args...);
	  // access the Nth argument of args
	  TYPE *outA = Get<IndVal_Alias<INDS,N>::ind>(args...); // outA = the "ind"-th argument.
	  // then we call the Operation function
	  THIS::Operation(out,outA,outB,outC);
	}
};

// specialization when second template is of type Var
template < template<class,class,class,class...> class OP, class FA, int N, int DIM, int CAT, class FC, class... PARAMS >
struct TernaryOp<OP,FA,Var<N,DIM,CAT>,FC,PARAMS...>  : TernaryOp_base<OP,FA,Var<N,DIM,CAT>,FC,PARAMS...> {

	using THIS = OP<FA,Var<N,DIM,CAT>,FC,PARAMS...>;

	template < class INDS, typename TYPE, typename... ARGS >
	static HOST_DEVICE INLINE void Eval(TYPE *out, ARGS... args) {
	  // we create a vector and call Eval only for FA and FC
	  TYPE outA[FA::DIM], outC[FC::DIM];
	  FA::template Eval<INDS>(outA,args...);
	  FC::template Eval<INDS>(outC,args...);
	  // access the Nth argument of args
	  TYPE *outB = Get<IndVal_Alias<INDS,N>::ind>(args...); // outB = the "ind"-th argument.
	  // then we call the Operation function
	  THIS::Operation(out,outA,outB,outC);
	}
};

// specialization when third template is of type Var
template < template<class,class,class,class...> class OP, class FA, class FB, int N, int DIM, int CAT, class... PARAMS >
struct TernaryOp<OP,FA,FB,Var<N,DIM,CAT>,PARAMS...>  : TernaryOp_base<OP,FA,FB,Var<N,DIM,CAT>,PARAMS...> {

	using THIS = OP<FA,FB,Var<N,DIM,CAT>,PARAMS...>;

	template < class INDS, typename TYPE, typename... ARGS >
	static HOST_DEVICE INLINE void Eval(TYPE *out, ARGS... args) {
	  // we create a vector and call Eval only for FA and FC
	  TYPE outA[FA::DIM], outB[FB::DIM];
	  FA::template Eval<INDS>(outA,args...);
	  FB::template Eval<INDS>(outB,args...);
	  // access the Nth argument of args
	  TYPE *outC = Get<IndVal_Alias<INDS,N>::ind>(args...); // outB = the "ind"-th argument.
	  // then we call the Operation function
	  THIS::Operation(out,outA,outB,outC);
	}
};

// specialization when first and second templates are of type Var
template < template<class,class,class,class...> class OP, int NA, int DIMA, int CATA, int NB, int DIMB, int CATB, class FC, class... PARAMS>
struct TernaryOp<OP,Var<NA,DIMA,CATA>,Var<NB,DIMB,CATB>,FC,PARAMS...> :
TernaryOp_base<OP,Var<NA,DIMA,CATA>,Var<NB,DIMB,CATB>,FC,PARAMS...> {

	using THIS = OP<Var<NA,DIMA,CATA>,Var<NB,DIMB,CATB>,FC,PARAMS...>;

	template < class INDS, typename TYPE, typename... ARGS >
	static HOST_DEVICE INLINE void Eval(TYPE *out, ARGS... args) {
  	  // we create a vector and call Eval only for FC
  	  TYPE outC[FC::DIM];
  	  FC::template Eval<INDS>(outC,args...);
	  // we access the NAth and NBth arguments of args
	  TYPE *outA = Get<IndVal_Alias<INDS,NA>::ind>(args...);
	  TYPE *outB = Get<IndVal_Alias<INDS,NB>::ind>(args...);
	  // then we call the Operation function	
	  THIS::Operation(out,outA,outB,outC);
	}  
};

// specialization when first and third templates are of type Var
template < template<class,class,class,class...> class OP, int NA, int DIMA, int CATA, class FB, int NC, int DIMC, int CATC, class... PARAMS>
struct TernaryOp<OP,Var<NA,DIMA,CATA>,FB,Var<NC,DIMC,CATC>,PARAMS...> :
TernaryOp_base<OP,Var<NA,DIMA,CATA>,FB,Var<NC,DIMC,CATC>,PARAMS...> {

	using THIS = OP<Var<NA,DIMA,CATA>,FB,Var<NC,DIMC,CATC>,PARAMS...>;

	template < class INDS, typename TYPE, typename... ARGS >
	static HOST_DEVICE INLINE void Eval(TYPE *out, ARGS... args) {
  	  // we create a vector and call Eval only for FB
  	  TYPE outB[FB::DIM];
  	  FB::template Eval<INDS>(outB,args...);
	  // we access the NAth and NCth arguments of args
	  TYPE *outA = Get<IndVal_Alias<INDS,NA>::ind>(args...);
	  TYPE *outC = Get<IndVal_Alias<INDS,NC>::ind>(args...);
	  // then we call the Operation function	
	  THIS::Operation(out,outA,outB,outC);
	}  
};

// specialization when second and third templates are of type Var
template < template<class,class,class,class...> class OP, class FA, int NB, int DIMB, int CATB, int NC, int DIMC, int CATC, class... PARAMS>
struct TernaryOp<OP,FA,Var<NB,DIMB,CATB>,Var<NC,DIMC,CATC>,PARAMS...> :
TernaryOp_base<OP,FA,Var<NB,DIMB,CATB>,Var<NC,DIMC,CATC>,PARAMS...> {

	using THIS = OP<FA,Var<NB,DIMB,CATB>,Var<NB,DIMB,CATB>,PARAMS...>;

	template < class INDS, typename TYPE, typename... ARGS >
	static HOST_DEVICE INLINE void Eval(TYPE *out, ARGS... args) {
  	  // we create a vector and call Eval only for FA
  	  TYPE outA[FA::DIM];
  	  FA::template Eval<INDS>(outA,args...);
	  // we access the NBth and NCth arguments of args
	  TYPE *outB = Get<IndVal_Alias<INDS,NB>::ind>(args...);
	  TYPE *outC = Get<IndVal_Alias<INDS,NC>::ind>(args...);
	  // then we call the Operation function	
	  THIS::Operation(out,outA,outB,outC);
	}  
};

// specialization when the three templates are of type Var
template < template<class,class,class,class...> class OP, int NA, int DIMA, int CATA, int NB, int DIMB, int CATB, int NC, int DIMC, int CATC, class... PARAMS>
struct TernaryOp<OP,Var<NA,DIMA,CATA>,Var<NB,DIMB,CATB>,Var<NC,DIMC,CATC>,PARAMS...> :
TernaryOp_base<OP,Var<NA,DIMA,CATA>,Var<NB,DIMB,CATB>,Var<NC,DIMC,CATC>,PARAMS...> {

	using THIS = OP<Var<NA,DIMA,CATA>,Var<NB,DIMB,CATB>,Var<NB,DIMB,CATB>,PARAMS...>;

	template < class INDS, typename TYPE, typename... ARGS >
	static HOST_DEVICE INLINE void Eval(TYPE *out, ARGS... args) {
	  // we access the NAth, NBth and NCth arguments of args
  	  TYPE *outA = Get<IndVal_Alias<INDS,NA>::ind>(args...);
	  TYPE *outB = Get<IndVal_Alias<INDS,NB>::ind>(args...);
	  TYPE *outC = Get<IndVal_Alias<INDS,NC>::ind>(args...);
	  // then we call the Operation function	
	  THIS::Operation(out,outA,outB,outC);
	}  
};







}
