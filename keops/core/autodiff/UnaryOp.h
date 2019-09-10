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
  using AllTypes = MergePacks<univpack<THIS>, typename F::AllTypes>;

  // "Replace" can be used to replace any occurrence of a sub-formula in a formula
  // For example Exp<Pow<Var<0,1,0>,3>>::Replace<Var<0,1,0>,Var<2,1,0>> will be Exp<Pow<Var<2,1,0>,3>>
  template<class A, class B>
  using Replace = CondType< B, OP<typename F::template Replace<A,B>,NS...>, IsSameType<A,THIS>::val >;

  // VARS gives the list of all "Vars" of a given category inside a formula
  // Here it is simple : the variables inside the formula OP<F,NS..> are the variables in F
  template < int CAT >
  using VARS = typename F::template VARS<CAT>;

};

// unary operator class : default Eval method
template < template<class,int...> class OP, class F, int... NS >
struct UnaryOp : UnaryOp_base<OP,F,NS...> {

  using THIS = OP<F,NS...>;

  template < class INDS, typename... ARGS >
  static HOST_DEVICE INLINE void Eval(__TYPE__* out, ARGS... args) {
    // we create a vector of size F::DIM
    __TYPE__ outA[F::DIM];
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

template < class INDS, typename... ARGS >
static HOST_DEVICE INLINE void Eval(__TYPE__* out, ARGS... args) {
  // we do not need to create a vector ; just access the Nth argument of args
  __TYPE__* outA = Get<IndVal_Alias<INDS,N>::ind>(args...); // outA = the "ind"-th argument.
  // then we call the Operation function
  THIS::Operation(out,outA);
}
};

}
