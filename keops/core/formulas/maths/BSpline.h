#pragma once

#include <sstream>
#include <assert.h>

#include "core/autodiff/BinaryOp.h"
#include "core/pre_headers.h"

namespace keops {


// Utility template to convert an "Ind(k)" sequence into an integer k.
template <class C> struct GetFirst {};

template <int N>
struct GetFirst<std::integer_sequence<int,N>>
{
    static const int value = N;
};


//////////////////////////////////////////////////////////////
////        BSPLINE VECTOR : BSPLINE<FT,FX,FK>            ////
//////////////////////////////////////////////////////////////

/*
This operation returns the BSpline vector of order FK (int, >= 1) associated to the knot vector FT (vector of T non-decreasing scalar values) and evaluated at location FX (scalar). The dot product of this vector of length T-FK with a vector of coefficients can be used to evaluate any Spline function on the knots FT.
*/
//template< class FT, class FX, int FK >
//struct BSpline : BinaryOp< BSpline, FT, FX, FK > {

template< class FT, class FX, class FKK >
struct BSpline_Impl : BinaryOp< BSpline_Impl, FT, FX, FKK > {
  static const int DIM = FT::DIM - 1; //FT::DIM - FK;

  static const int FK = GetFirst<FKK>::value;
  static_assert(FT::DIM >= FK + 1, "BSpline Knot vector must be of length >= order + 1");
  static_assert(FK >= 1, "BSpline order must be >= 1");
  static_assert(FX::DIM == 1, "BSpline can only be evaluated at a scalar (DIM=1) location");

  static void PrintIdString(::std::stringstream &str) { str << "BSpline"; }


  template < typename TYPE > 
  static DEVICE INLINE void Operation(TYPE *out, TYPE *inT, TYPE *inX) {
        TYPE x = inX[0];
        // Order 1 Spline: one-hot encoding of whether t[i] <= x < t[i+1]
        #pragma unroll
        for (int i = 0; i < FT::DIM - 1; i++) {
            out[i] = cast_to<TYPE>(inT[i] <= x && x < inT[i+1]);
        }

        // Recursive De Boor's algorithm:
        TYPE ratio_1, ratio_2;
        #pragma unroll
        for (int k = 1; k < FK; k++) {  // Order 2, 3, ...
            // Compute the first ratio "omega_i,k" = (x - t_i) / (t_i+k - t_i) for i=0:
            ratio_1 = (inT[0] < inT[k]) ? (x - inT[0]) / (inT[k] - inT[0]) : cast_to<TYPE>(0.0f);

            #pragma unroll
            for (int i = 0; i < FT::DIM - k - 1; i++) {  // Loop over out[0:len(T)-k]

              // Compute the second ratio "omega_i+1,k" = (x - t_i+1) / (t_i+k+1 - t_i+1):
              ratio_2 = (inT[i+1] < inT[i+1+k]) ? (x - inT[i+1]) / (inT[i+1+k] - inT[i+1]) : cast_to<TYPE>(0.0f);
              // In place computation of B_i,k+1(x) as
              // omega_i,k(x) * B_i,k(x) + (1 - omega_i+1,k(x)) * B_i+1,k(x)
              out[i] = ratio_1 * out[i] + (cast_to<TYPE>(1.0f) - ratio_2) * out[i+1];

              // Update the ratios as i -> i+1:
              ratio_1 = ratio_2;
            }
                
        }
  }

  // The default version in UnivPack.h crashes at compile time
  // so we have to override it.
  static void PrintId(::std::stringstream& str) {
    str << "BSpline(";  
    FT::PrintId(str);                            // prints the formula FT
    str << ",";
    FX::PrintId(str);                            // prints the formula FX
    str << "," << FK<< ")";                       
  }
  

  template<class V, class GRADIN>
  using DiffTFT = typename FT::template DiffT<V, GRADIN>;

  template<class V, class GRADIN>
  using DiffTFX = typename FX::template DiffT<V, GRADIN>;

  // Currently not implemented:
  template<class V, class GRADIN>
  using DiffT = Zero< V::DIM >;

};


// N.B.: BSpline_Impl creates a vector of size "FT::DIM - 1 = n_knots - 1"
// since we need this amount of space as an intermediate buffer for the computation
// of the final "FT::DIM - FKK = n_knots - order" coefficients.
// In the final BSpline operation, we simply use "extract" to discard
// the FKK-1 irrelevant coefficients.
template< class FT, class FX, int FKK >
using BSpline = Extract<BSpline_Impl<FT, FX, std::integer_sequence<int, FKK> >, 0, FT::DIM - FKK>;


#define BSpline(t,x,k) KeopsNS<BSpline<decltype(InvKeopsNS(t)),decltype(InvKeopsNS(x)),k>>()

}
