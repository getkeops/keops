#pragma once

#include <assert.h>
#include <iostream>
#include <sstream>

#include "lib/sequences/include/tao/seq/integer_sequence.hpp"
#include "lib/sequences/include/tao/seq/concatenate.hpp"
#include "lib/sequences/include/tao/seq/sum.hpp"
#include "lib/sequences/include/tao/seq/make_integer_range.hpp"
#include "lib/sequences/include/tao/seq/permutate.hpp"
#include "lib/sequences/include/tao/seq/first.hpp"
#include "lib/sequences/include/tao/seq/sort_index.hpp"
#include "lib/sequences/include/tao/seq/functional.hpp"
// #include "lib/sequences/include/tao/seq/difference.hpp"
#include "lib/sequences/include/tao/seq/reverse.hpp"
#include "lib/sequences/include/tao/seq/multiplies.hpp"
#include "lib/sequences/include/tao/seq/prod.hpp"
#include "lib/sequences/include/tao/seq/exclusive_scan.hpp"

#include "core/autodiff/BinaryOp.h"
#include "core/pre_headers.h"

// TODO : wait for a fix from tao::seq
namespace tao {
namespace seq {
namespace impl {
template < typename, typename, typename >
struct difference;

template < typename T, T... As >
struct difference< T, integer_sequence< T >, integer_sequence< T, As... > > {
  using type = integer_sequence< T >;
};

template < typename T, T... As, T b, T... Bs >
struct difference< T, integer_sequence< T, b, Bs... >, integer_sequence< T, As... > > {
  constexpr static bool included = tao::seq::contains< T, b, As... >::value;
  using tail = typename difference< T, integer_sequence< T, Bs... >, integer_sequence< T, As... > >::type;
  using type = typename ::std::conditional< included, tail, typename tao::seq::concatenate< integer_sequence< T, b >, tail >::type >::type;
};

}  // namespace impl

template < typename, typename >
struct difference;

template < typename TA, TA... As, typename TB, TB... Bs >
struct difference< integer_sequence< TA, As... >, integer_sequence< TB, Bs... > > {
  using CT = typename ::std::common_type< TA, TB >::type;

  template < CT N >
  using check = contains< CT, N, Bs... >;

  // ERROR THERE
  // using type = concatenate_t< impl::conditional_t< check< As >::value, integer_sequence< CT >, integer_sequence< CT, As > >... >;
  // OK
  using type = typename impl::difference< CT, integer_sequence< CT, As... >, integer_sequence< CT, Bs... > >::type; // ERROR
};

template < typename A, typename B >
using difference_t = typename difference< A, B >::type;

}  // namespace seq

}  // namespace tao

namespace keops {

template < size_t... Ix >
using index_sequence = tao::seq::integer_sequence< size_t, Ix... >;

// Print an Index sequence

template < size_t... Is >
void print_index_sequence(index_sequence< Is... >) {
  (void) ::std::initializer_list< int >{(::std::cout << Is << " ", 0)...};
  ::std::cout << ::std::endl;
}


// Return the index_sequence containing the cumulative product of all element of index_sequences A
// except the first.
//  using namespace tao::seq;
//    static_assert(::std::is_same< product_red<index_sequence<4,5,6,7> >,
//                                            index_sequence<210,42,7,1> >::value, "ooops" );

template < typename S >
using cum_prod = tao::seq::reverse_t< tao::seq::exclusive_scan_t< tao::seq::op::multiplies, tao::seq::reverse_t< S >, 1 > >;

namespace loop_impl {
template < typename, size_t... I >
struct Looper;

template < size_t... Is >
struct Looper< index_sequence< Is...>> {
  template < typename Func >
  constexpr static DEVICE void f(Func &&func) {
    func(index_sequence< Is... >{});
  }
};

template < size_t I, size_t... Is, size_t... PIs >
struct Looper< index_sequence< PIs... >, I, Is... > {
  template < ::std::size_t... Idx, typename Func >
  constexpr static DEVICE void f_help(index_sequence< Idx... >, Func &&func) {
    (void) ::std::initializer_list< int >{(Looper< index_sequence< PIs..., Idx >, Is... >::f(func), 0)...};
  }

  template < typename Func >
  constexpr static DEVICE void f(Func &&func) {
    f_help(tao::seq::make_index_sequence< I >{}, func);
  }

};

template < typename >
struct loop_t;

template < size_t... Is >
struct loop_t< index_sequence< Is...>> {
  using type = Looper< index_sequence<>, Is... >;
};

}

template < typename Is >
using loop = typename loop_impl::loop_t< Is >::type;

// Dummy class that stores the indices computes for tensordot
struct tensordot_indices {
  size_t out_indices;
  size_t a_indices;
  size_t b_indices;
};

template < class DIMFA, class DIMFB, class CONTFA, class CONTFB, class PERMUTE >
struct tensordot_parameters {

  // Left hand-side
  using indices_keepdim_a_t = tao::seq::difference_t< tao::seq::make_index_sequence< tao::seq::impl::sequence_size< DIMFA >::value >, CONTFA >;
  using keepdim_a_t = tao::seq::map_t< indices_keepdim_a_t, DIMFA >;
  using contdim_a_t = tao::seq::map_t< CONTFA, DIMFA >;
#if C_CONTIGUOUS
  using list_stride_dim_a_t = cum_prod< DIMFA >;
#else
  using list_stride_dim_a_t = cum_prod< tao::seq::reverse_t< DIMFA>>;
#endif

  // Right hand-side
  using indices_keepdim_b_t = tao::seq::difference_t< tao::seq::make_index_sequence< tao::seq::impl::sequence_size< DIMFB >::value >, CONTFB >;
  using keepdim_b_t = tao::seq::map_t< indices_keepdim_b_t, DIMFB >;
  using contdim_b_t = tao::seq::map_t< CONTFB, DIMFB >;
#if C_CONTIGUOUS
  using list_stride_dim_b_t = cum_prod< DIMFB >;
#else
  using list_stride_dim_b_t = cum_prod< tao::seq::reverse_t< DIMFB>>;
#endif

  static_assert(::std::is_same< contdim_a_t, contdim_b_t >::value,
                "In TensorDot: contracting dimensions should be the same");

  // Output
  using keepdim_t = tao::seq::concatenate_t< keepdim_a_t, keepdim_b_t >;
#if C_CONTIGUOUS
  using list_stride_keepdim_t = cum_prod<tao::seq::permutate_t< PERMUTE, keepdim_t >>;
#else
  using list_stride_keepdim_t = cum_prod< tao::seq::reverse_t< tao::seq::permutate_t< PERMUTE, keepdim_t>> >;
#endif
  constexpr static size_t dimout = tao::seq::prod< keepdim_t >::value;

  static_assert(::std::is_same< tao::seq::permutate_t< PERMUTE, PERMUTE >, tao::seq::make_index_range< 0, keepdim_t::size()>>::value,
                "In TensorDot: PERMUTE should be a permutation index_sequence.");

  // Loop: in this code we choose to loop on the keepdims first and then on the contraction dims.
  using loopdim_t = tao::seq::concatenate_t< keepdim_t, contdim_a_t >;

  constexpr static size_t dimloop = tao::seq::prod< loopdim_t >::value;
  constexpr static size_t number_of_dimloop = loopdim_t::size();

  using ala = tao::seq::concatenate_t< tao::seq::make_index_range< 0, keepdim_a_t::size() >, tao::seq::make_index_range< keepdim_t::size(), number_of_dimloop>>;
  using ali = tao::seq::concatenate_t< indices_keepdim_a_t, CONTFA >;
  using list_indices_a_intot = tao::seq::permutate_t< ali, ala >;

  using bla = tao::seq::concatenate_t< tao::seq::make_index_range< keepdim_a_t::size(), keepdim_t::size() >, tao::seq::make_index_range< keepdim_t::size(), number_of_dimloop>>;
  using bli = tao::seq::concatenate_t< indices_keepdim_b_t, CONTFB >;
  using list_indices_b_intot = tao::seq::permutate_t< bli, bla >;

  // used to compute the Gradient
  using list_indices_keepdim_a_inout = typename tao::seq::make_index_range< 0, keepdim_a_t::size() >;
  using reordered_contfa = tao::seq::permutate_t< tao::seq::sort_index_t< tao::seq::op::less, CONTFB >, CONTFA >;
  using reordered_keepdim_a = tao::seq::permutate_t< tao::seq::sort_index_t< tao::seq::op::less, tao::seq::map_t< list_indices_keepdim_a_inout, PERMUTE>>, indices_keepdim_a_t >;
  using moveaxis_a = tao::seq::concatenate_t< reordered_keepdim_a, reordered_contfa >;

  using list_indices_keepdim_b_inout = typename tao::seq::make_index_range< keepdim_a_t::size(), keepdim_t::size() >;
  using reordered_contfb = tao::seq::permutate_t< tao::seq::sort_index_t< tao::seq::op::less, CONTFA >, CONTFB >;
  using reordered_keepdim_b = tao::seq::permutate_t< tao::seq::sort_index_t< tao::seq::op::less, tao::seq::map_t< list_indices_keepdim_b_inout, PERMUTE>>, indices_keepdim_b_t >;
  using moveaxis_b = tao::seq::concatenate_t< reordered_keepdim_b, reordered_contfb >;

  template < class IND >
  DEVICE constexpr static tensordot_indices compute_tensordot_indices(IND) {

    // a_indices
    using list_indices_a = tao::seq::map_t< list_indices_a_intot, IND >;
#if C_CONTIGUOUS
    size_t a_indices = tao::seq::sum<tao::seq::multiplies_t< list_stride_dim_a_t, list_indices_a > >::value;
#else
    size_t a_indices = tao::seq::sum< tao::seq::multiplies_t< list_stride_dim_a_t, tao::seq::reverse_t< list_indices_a>> >::value;
#endif

    // b_indices
    using list_indices_b = tao::seq::map_t< list_indices_b_intot, IND >;
#if C_CONTIGUOUS
    size_t b_indices = tao::seq::sum< tao::seq::multiplies_t< list_stride_dim_b_t, list_indices_b > >::value;
#else
    size_t b_indices = tao::seq::sum< tao::seq::multiplies_t< list_stride_dim_b_t, tao::seq::reverse_t< list_indices_b>> >::value;
#endif

    // out_indices
    using list_indices_keepdim = tao::seq::permutate_t< PERMUTE, tao::seq::first_t< tao::seq::impl::sequence_size< keepdim_t >::value, IND >>;
#if C_CONTIGUOUS
    size_t out_indices = tao::seq::sum<tao::seq::multiplies_t< list_stride_keepdim_t, list_indices_keepdim >>::value;
#else
    size_t out_indices = tao::seq::sum< tao::seq::multiplies_t< list_stride_keepdim_t, tao::seq::reverse_t< list_indices_keepdim >>>::value;
#endif

    //::std::cout << "list_stride_keepdim_t: "; tao::seq::print_index_sequence(list_stride_keepdim_t{});

    return tensordot_indices{out_indices, a_indices, b_indices};
  }

  template < typename Func >
  struct compute_tensordot_indices_t {
    template < size_t... Is >
    DEVICE void operator()(index_sequence< Is... > x) {
      _f(compute_tensordot_indices(x));
    }

    Func &_f;
    DEVICE compute_tensordot_indices_t(Func &&f) : _f(f) {}
  };

  template < typename Func >
  static DEVICE auto compute_tensordot_indices_apply(Func &&f) {
    return compute_tensordot_indices_t< Func >(::std::forward< Func >(f));
  }

};



/////////////////////////////////////////////////////////////////////////
////              Tensor dot product      A : b                      ////
/////////////////////////////////////////////////////////////////////////


template < class A, class B, class DIMFA, class DIMFB, class CONTFA, class CONTFB, class PERMUTEDIMOUT=::std::make_index_sequence<
    DIMFA::size() + DIMFB::size() - 2 * CONTFA::size()>>
struct TensorDot : BinaryOp< TensorDot, A, B, DIMFA, DIMFB, CONTFA, CONTFB, PERMUTEDIMOUT > {
  // A is vector of size p ** n, interpreted as matrix (column major), B is vector of size p ** m, interpreted as column vector
  // n=3 and m=2 are assume to be known
  // output is vector of size n

  static_assert(DIMFA::size() > 0, "Please provide a non empty DIMA");
  static_assert(DIMFB::size() > 0, "Please provide a non empty DIMB");
  static_assert(tao::seq::prod< DIMFA >::value == A::DIM, "DIMA is not consistant with dimension of A");
  static_assert(tao::seq::prod< DIMFB >::value == B::DIM, "DIMB is not consistant with dimension of B");

  using parameters = tensordot_parameters< DIMFA, DIMFB, CONTFA, CONTFB, PERMUTEDIMOUT >;

  static const int DIM = parameters::dimout;

  static void PrintIdString(::std::stringstream &str) {
    str << ":";
  }

  static DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *inA, __TYPE__ *inB) {
#pragma unroll
    for (int i = 0; i < DIM; i++)
      out[i] = 0;

    loop< typename parameters::loopdim_t >::f(parameters::compute_tensordot_indices_apply([&out, &inA, &inB](
        tensordot_indices td) {
      out[td.out_indices] += inA[td.a_indices] * inB[td.b_indices];
    }));
  }

  template < class V, class GRADIN >
  using DiffTA = typename A::template DiffT< V, GRADIN >;

  template < class V, class GRADIN >
  using DiffTB = typename B::template DiffT< V, GRADIN >;


  template < class V, class GRADIN >
  using DiffT = Add<
      DiffTA< V, TensorDot< GRADIN, B,
                            tao::seq::permutate_t< PERMUTEDIMOUT, typename parameters::keepdim_t >,                   // 3
                            DIMFB,                                                                                    // 4 2
                            tao::seq::map_t< typename parameters::list_indices_keepdim_b_inout, PERMUTEDIMOUT >,      // .
                            typename parameters::indices_keepdim_b_t,                                                 // .
                            typename parameters::moveaxis_a>>,                                                        // 1, 2 0
      DiffTB< V, TensorDot< GRADIN, A,
                            tao::seq::permutate_t< PERMUTEDIMOUT, typename parameters::keepdim_t >,                   // 3
                            DIMFA,                                                                                    // 2, 3, 4
                            tao::seq::map_t< typename parameters::list_indices_keepdim_a_inout, PERMUTEDIMOUT >,      //0
                            typename parameters::indices_keepdim_a_t,                                                 //1
                            typename parameters::moveaxis_b>>                                                         // . , 0 1
  >;

};

#define TensorDot(f, g, ...) KeopsNS<TensorDot<decltype(InvKeopsNS(f)),decltype(InvKeopsNS(g)), __VA_ARGS__>>()

}
