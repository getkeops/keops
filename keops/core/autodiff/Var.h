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

  // prints the variable as a string
  // we just print e.g. x0, y2, p1 to simplify reading, forgetting about dimensions
  static void PrintId(::std::stringstream &str) {
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

  // AllTypes is a tuple of types which gives all sub-formulas in a formula (including the formula itself)
  // here there is just one type in the Var type : itself
  // because it does not depend on other sub-formula
  using AllTypes = univpack< Var< N, DIM, CAT>>;

  // VARS gives the list of all Vars of a given category in a formula
  // Here we add the current Var to the list if it is of the requested category, otherwise nothing
  template < int CAT_ >        // Var::VARS<1> = [Var(with CAT=0)] if Var::CAT=1, [] otherwise
  using VARS = CondType< univpack< Var< N, DIM, CAT>>, univpack<>, CAT == CAT_ >;

  // Evaluate a variable given a list of arguments:
  //
  // Var( 5, DIM )::Eval< [ 2, 5, 0 ], type2, type5, type0 >( out, var2, var5, var0 )
  //
  // will see that the index 1 is targeted,
  // assume that "var5" is of size DIM, and copy its value in "out".
  template < class INDS, typename ...ARGS >
  static HOST_DEVICE INLINE void Eval(__TYPE__ *out, ARGS... args) {
    // IndVal_Alias<INDS,N>::ind is the first index such that INDS[ind]==N. Let's call it "ind"
    __TYPE__ *xi = Get< IndVal_Alias< INDS, N >::ind >(args...);   // xi = the "ind"-th argument.
    for (int k = 0; k < DIM; k++)                                  // Assume that xi and out are of size DIM,
      out[k] = xi[k];                                       // and copy xi into out.
  }


  // Assuming that the gradient wrt. Var is GRADIN, how does it affect V ?
  // Var::DiffT<V, grad_input> = grad_input   if V == Var (in the sense that it represents the same symb. var.)
  //                             Zero(V::DIM) otherwise
  template < class V, class GRADIN >
  using DiffT = IdOrZero< Var< N, DIM, CAT >, V, GRADIN >;

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

}