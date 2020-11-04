#pragma once

#include <sstream>

#include "core/pack/UnivPack.h"
#include "core/pack/Pack.h"
#include "core/pack/MergePacks.h"
#include "core/pack/CondType.h"
#include "core/autodiff/Var.h"

namespace keops {


//////////////////////////////////////////////////////////////////////////////////////////////////
////                      Unary and Binary operations wrappers                                ////
//////////////////////////////////////////////////////////////////////////////////////////////////

// Unary and Binary structures for defining common methods for the math operations
// (Add, Scal, Scalprod, Exp, etc., see files math.h and norms.h)
// we define common methods in a base class
// and then define a derived class to be able to specialize the evaluation method
// when dealing with the Var class as template parameters
// in order to avoid the use of Eval member function of the Var class which does a useless vector copy.

// unary operator base class : common methods
// unary operators are of the type OP<F,NS..> : for example Exp<F>, Log<F>, Pow<F,N>
// There template parameters are : one subformula F, plus optionally some integers
template < template<class,int...> class OP, class F, int... NS >
struct UnaryOp_base {

  using THIS = OP<F,NS...>;

  using ARG = F;

  template < int DIMCHK >
  using CHUNKED_VERSION = void;

  static const bool IS_CHUNKABLE = false;

  template < int DIMCHK >
  using CHUNKED_FORMULAS = typename F::template CHUNKED_FORMULAS<DIMCHK>;

  static const int NUM_CHUNKED_FORMULAS = F::NUM_CHUNKED_FORMULAS;


  // NB. The following commented code should be ok but it dos not compile with Cuda 11 as of 2020 aug 13th...
  /*
  template < int IND >
  using POST_CHUNK_FORMULA = OP < typename F::template POST_CHUNK_FORMULA<IND>, NS... >;
  */
  // ... so we use an additional "_Impl" structure to specialize in case of empty PARAMS pack : 
  template < int IND, int SIZE_NS >
  struct POST_CHUNK_FORMULA_Impl {
	  using type = OP < typename F::template POST_CHUNK_FORMULA<IND>, NS... >;
  };
  
  template < int IND >
  struct POST_CHUNK_FORMULA_Impl < IND, 0 > {
	  using type = OP < typename F::template POST_CHUNK_FORMULA<IND> >;
  };
  
  template < int IND >
  using POST_CHUNK_FORMULA = typename POST_CHUNK_FORMULA_Impl<IND,sizeof...(NS)>::type;
 
  
  // recursive function to print the formula as a string
  static void PrintId(::std::stringstream& str) {
    THIS::PrintIdString(str);      // prints the id string of the operator : "Exp", "Log", "Pow",...
    str << "(";                    // prints "("
    F::PrintId(str);               // prints the formula F
    pack<NS...>::PrintComma(str);  // prints a "," if there is at least one integer in NS..., otherwise nothing
    pack<NS...>::PrintAll(str);    // prints the integers, with commas between them
    str << ")";                    // prints ")"
  }

  // AllTypes is a tuple of types which gives all sub-formulas in a formula (including the formula itself)
  // for example Exp<Pow<Var<0,1,0>,3>>::AllTypes is univpack< Exp<Pow<Var<0,1,0>,3>> , Pow<Var<0,1,0>,3> , Var<0,1,0> >
  // N.B we comment out AutoFactorize and AllTypes in all code as of oct 2020 to speed up compile time
  // using AllTypes = MergePacks<univpack<THIS>, typename F::AllTypes>;


  // "Replace" can be used to replace any occurrence of a sub-formula in a formula
  // For example Exp<Pow<Var<0,1,0>,3>>::Replace<Var<0,1,0>,Var<2,1,0>> will be Exp<Pow<Var<2,1,0>,3>>

  // NB. The following commented code should be ok but it dos not compile with Cuda 11 as of 2020 aug 13th...
  /*
  template<class A, class B>
  using Replace = CondType< B, OP<typename F::template Replace<A,B>,NS...>, IsSameType<A,THIS>::val >;
  */

  // ... so we use an additional "_Impl" structure to specialize in case of empty NS pack : 
  template < class A, class B, int SIZE_NS >
  struct Replace_Impl {
    using type = CondType < B, OP < typename F::template Replace<A,B>, NS... >, IsSameType<A,THIS>::val >;
  };

  template < class A, class B >
  struct Replace_Impl<A,B,0> {
    using type = CondType < B, OP < typename F::template Replace<A,B> >, IsSameType<A,THIS>::val >;
  };

  template < class A, class B >
  using Replace = typename Replace_Impl<A,B,sizeof...(NS)>::type;


  // version with two replacements of Vars at a time (two consecutive Replace might not work because of non compatibl>

  // NB. The following commented code should be ok but it dos not compile with Cuda 11 as of 2020 aug 13th...
  /*
  template<class A1, class B1, class A2, class B2>
  using ReplaceVars2 = OP<typename F::template ReplaceVars2<A1,B1,A2,B2>,NS...>;
  */

  // ... so we use an additional "_Impl" structure to specialize in case of empty NS pack : 
  template<class A1, class B1, class A2, class B2, int SIZE_NS>
  struct ReplaceVars2_Impl {
    using type = OP<typename F::template ReplaceVars2<A1,B1,A2,B2>, NS...>;
  };

  template<class A1, class B1, class A2, class B2>
  struct ReplaceVars2_Impl<A1,B1,A2,B2,0> {
    using type = OP<typename F::template ReplaceVars2<A1,B1,A2,B2>>;
  };

  template<class A1, class B1, class A2, class B2>
  using ReplaceVars2 = typename ReplaceVars2_Impl<A1,B1,A2,B2,sizeof...(NS)>::type;


  // VARS gives the list of all "Vars" of a given category inside a formula
  // Here it is simple : the variables inside the formula OP<F,NS..> are the variables in F
  template < int CAT >
  using VARS = typename F::template VARS<CAT>;

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

// unary operator class : default Eval method
template < template<class,int...> class OP, class F, int... NS >
struct UnaryOp : UnaryOp_base<OP,F,NS...> {

  using THIS = OP<F,NS...>;

  template < class INDS, typename TYPE, typename... ARGS >
  static HOST_DEVICE INLINE void Eval(TYPE *out, ARGS... args) {
    // we create a vector of size F::DIM
    TYPE outA[F::DIM];
    // then we call the Eval function of F
    F::template Eval<INDS>(outA,args...);
    // then we call the Operation function
    THIS::Operation(out,outA);
  }
};

// specialization when template F is of type Var
template < template< class, int... > class OP, int N, int DIM, int CAT, int... NS >
struct UnaryOp< OP, Var< N, DIM, CAT >, NS... >  : UnaryOp_base< OP,Var< N, DIM, CAT >, NS... > {

using THIS = OP< Var< N, DIM, CAT>, NS... >;

template < class INDS, typename TYPE, typename... ARGS >
static HOST_DEVICE INLINE void Eval(TYPE *out, ARGS... args) {
  // we do not need to create a vector ; just access the Nth argument of args
  TYPE *outA = Get<IndVal_Alias<INDS,N>::ind>(args...); // outA = the "ind"-th argument.
  // then we call the Operation function
  THIS::Operation(out,outA);
}
};




}
