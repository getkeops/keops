#pragma once

#include <sstream>

#include "core/pack/UnivPack.h"
#include "core/pack/IsSame.h"
#include "core/pack/CondType.h"
#include "core/pack/IndVal.h"
#include "core/autodiff/IdOrZero.h"
#include "core/pre_headers.h"

namespace keops {


//////////////////////////////////////////////////////////////
////                      VARIABLE                        ////
//////////////////////////////////////////////////////////////

/*
 * Class for base variable
 * It is the atomic block of our autodiff engine.
 * A variable is given by :
 * - an index number _N (is it x1i, x2i, x3i or ... ?)
 * - a dimension _DIM of the vector
 * - a category CAT, equal to 0 if Var is "a  parallel variable" xi,
 *                   equal to 1 if Var is "a summation variable" yj,
 *                     equal to 2 if Var is "a parameter variable" p,
 */
template < int _N, int _DIM, int _CAT = 0 >
struct Var {
  static const int N = _N;   // The index and dimension of Var, formally specified using the
  static const int DIM = _DIM; // templating syntax, are accessible using Var::N, Var::DIM.
  static const int CAT = _CAT;

  using THIS = Var< N, DIM, CAT >;

  template < int DIMCHK >
  using CHUNKED_VERSION = Var< N, DIMCHK, CAT >;

  static const bool IS_CHUNKABLE = true;

  template < int DIMCHK >
  using CHUNKED_FORMULAS = univpack<>;

  static const int NUM_CHUNKED_FORMULAS = 0;

  template < int IND >
  using POST_CHUNK_FORMULA = Var< N, DIM, CAT >;

  // prints the variable as a string
  // we just print e.g. x0, y2, p1 to simplify reading, forgetting about dimensions
  static void PrintId(::std::ostream &str) {
    if (CAT == 0)
      str << "x";
    else if (CAT == 1)
      str << "y";
    else if (CAT == 2)
      str << "p";
    else
      str << "z";    // "z" is used for intermediate variables, which are used in "Factorize" (see factorize.h)
    str << N;
  }

  // "Replace" can be used to replace any occurrence of a sub-formula in a formula
  // It must be recursively implemented but here we are in a terminal case,
  // because Var types do not depend on other sub-formulas
  // So here we just replace THIS by B if A=THIS, otherwise we keep THIS
  template < class A, class B >
  using Replace = CondType< B, THIS, IsSameType< A, THIS >::val >;

  // Replacement of two Vars at a time
  template < class A1, class B1, class A2, class B2 >
  using ReplaceVars2 = typename THIS::template Replace<A1,B1>::template Replace<A2,B2>;

  // AllTypes is a tuple of types which gives all sub-formulas in a formula (including the formula itself)
  // here there is just one type in the Var type : itself
  // because it does not depend on other sub-formula
  // N.B we comment out AutoFactorize and AllTypes in all code as of oct 2020 to speed up compile time
  // using AllTypes = univpack< Var< N, DIM, CAT>>;

  // VARS gives the list of all Vars of a given category in a formula
  // Here we add the current Var to the list if it is of the requested category, otherwise nothing
  template < int CAT_=-1 >        // Var::VARS<1> = [Var(with CAT=0)] if Var::CAT=1, [] otherwise
  using VARS = CondType< univpack< Var< N, DIM, CAT>>, univpack<>, CAT==CAT_ || CAT_==-1 >;

  template < int CAT_=-1 >
  using CHUNKED_VARS = VARS<CAT_>;

  template < int CAT_=-1 >
  using NOTCHUNKED_VARS = univpack<>;

  // Evaluate a variable given a list of arguments:
  //
  // Var( 5, DIM )::Eval< [ 2, 5, 0 ], type2, type5, type0 >( out, var2, var5, var0 )
  //
  // will see that the index 1 is targeted,
  // assume that "var5" is of size DIM, and copy its value in "out".
  template < class INDS, typename TYPE, typename ...ARGS >
  static HOST_DEVICE INLINE void Eval(TYPE *out, ARGS... args) {
    // IndVal_Alias<INDS,N>::ind is the first index such that INDS[ind]==N. Let's call it "ind"
    TYPE *xi = Get< IndVal_Alias< INDS, N >::ind >(args...);   // xi = the "ind"-th argument.
    #pragma unroll
    for (int k = 0; k < DIM; k++)                                  // Assume that xi and out are of size DIM,
      out[k] = xi[k];                                       // and copy xi into out.
  }


  // Assuming that the gradient wrt. Var is GRADIN, how does it affect V ?
  // Var::DiffT<V, grad_input> = grad_input   if V == Var (in the sense that it represents the same symb. var.)
  //                             Zero(V::DIM) otherwise
  template < class V, class GRADIN >
  using DiffT = IdOrZero< Var< N, DIM, CAT >, V, GRADIN >;

  // operator as shortcut to Eval...
  template < typename INDS >
  struct EvalFun {
      template < typename... Args >
      DEVICE INLINE void operator()(Args... args) {
      	THIS::template Eval<INDS>(args...);
      }
  };

};

//////////////////////////////////////////////////////////////
////    STANDARD VARIABLES :_X<N,DIM>,_Y<N,DIM>,_P<N>     ////
//////////////////////////////////////////////////////////////

// N.B. : We leave "X", "Y" and "P" to the user
//        and restrict ourselves to "_X", "_Y", "_P".

template < int N, int DIM >
using _X = Var< N, DIM, 0 >;

template < int N, int DIM >
using _Y = Var< N, DIM, 1 >;

template < int N, int DIM >
using Param = Var< N, DIM, 2 >;

template < int N, int DIM >
using _P = Param< N, DIM >;

#define Var(N, DIM, CAT) KeopsNS<Var<N,DIM,CAT>>()

#define Vi(N, DIM) KeopsNS<_X<N,DIM>>()
#define Vj(N, DIM) KeopsNS<_Y<N,DIM>>()
#define Pm(N, DIM) KeopsNS<_P<N,DIM>>()



//////////////////////////////////////////////////////////////
////                      CHUNK FORMULA                   ////
//////////////////////////////////////////////////////////////

/*
 * Class for building a chunked computation scheme.
 */

template < class _CHUNKED_FORMULA, class _POST_CHUNK_FORMULA >
struct ChunkedFormula {

  using CHUNKED_FORMULA = univpack < _CHUNKED_FORMULA >;

  using POST_CHUNK_FORMULA = _POST_CHUNK_FORMULA;

};

}
