#pragma once

#include <iostream>
#include "lib/sequences/include/tao/seq/integer_sequence.hpp"
#include "lib/sequences/include/tao/seq/contains.hpp"
#include "lib/sequences/include/tao/seq/concatenate.hpp"
#include "lib/sequences/include/tao/seq/map.hpp"
#include "lib/sequences/include/tao/seq/zip.hpp"
#include "lib/sequences/include/tao/seq/index_of.hpp"
#include "lib/sequences/include/tao/seq/select.hpp"
#include "lib/sequences/include/tao/seq/sum.hpp"
#include "lib/sequences/include/tao/seq/make_integer_range.hpp"

namespace tao {
namespace seq {
// -------------------------------------------------------------- //
//      Various operation for reordering index_sequences          //
// -------------------------------------------------------------- //


namespace impl {

template<typename, typename>
struct permute_i;

template<std::size_t... Ns, std::size_t... Is>
struct permute_i<tao::seq::index_sequence<Ns...>, tao::seq::index_sequence<Is...> > {
  using type = index_sequence<tao::seq::index_of<size_t, Is, Ns...>::value...>;
};

} // namespace impl


// Sort :
//
//    using namespace tao::seq;
//    static_assert(std::is_same<tao::seq::sort<ord, index_sequence<39,2,4,10>>::type, index_sequence<2,4,10,39>>::value, "ooops");
//    static_assert(std::is_same<tao::seq::sort_index<ord, index_sequence<39,2,4,10>>::type, index_sequence<3,0,1,2>>::value, "ooops");
//    print_index_sequence(index_sequence<3,5,2,8>{});
//
//
namespace sort_impl {
template < template <size_t, size_t> class Pred, class Ia, class Ib> struct merge;

template <template <size_t, size_t> class Pred, class Ib>
struct merge<Pred, index_sequence<>, Ib> {
  using type = Ib;
};

template <template <size_t, size_t> class Pred, class Ia>
struct merge<Pred, Ia, index_sequence<>> {
   using type = Ia;
};

            template <template <size_t, size_t> class Pred, size_t a, size_t... Ia, size_t b, size_t... Ib>
            struct merge<Pred, index_sequence<a,Ia...>, index_sequence<b,Ib...> > {
                using type = std::conditional_t< Pred<a,b>::value,
                        tao::seq::concatenate_t<index_sequence<a>, typename merge<Pred, index_sequence<Ia...>, index_sequence<b, Ib...> >::type>,
                        tao::seq::concatenate_t<index_sequence<b>, typename merge<Pred, index_sequence<a, Ia...>, index_sequence<Ib...> >::type>>;
            };

        template< typename, typename >
        struct permute_i;

        template< std::size_t... Ns, std::size_t... Is  >
        struct permute_i< index_sequence< Ns... >, index_sequence< Is... > > {
            using type = index_sequence< index_of<size_t, Is, Ns...>::value... >;
        };
}  // namespace sortimpl

       template< template <size_t, size_t> class Pred, class Is>
       struct sort;

       template< template <size_t, size_t> class Pred, size_t a, size_t... Is>
       struct sort<Pred, index_sequence<a, Is...> > {
           using tail_sorted = typename sort<Pred, index_sequence<Is...> >::type;
           using type = typename sort_impl::merge<Pred, index_sequence<a>, tail_sorted>::type;
       };


       template < template <size_t, size_t> class Pred >
       struct sort<Pred, index_sequence<> > {
           using type = index_sequence<>;
       };

        template< template <size_t, size_t> class Pred, class Is>
        struct sort_index;


        template< template <size_t, size_t> class Pred, size_t... Ns>
        struct sort_index<Pred, index_sequence<Ns...> > {
            using sorted = typename sort<Pred, index_sequence<Ns...> >::type;
            using type  = typename sort_impl::permute_i<sorted, index_sequence<Ns...>  >::type;
        };

template <size_t a, size_t b> struct ord {
    static const bool value = a < b;
};

template <class Is>
struct sort_index_asc{
   using type = typename sort_index<ord, Is>::type;
};


// Print an Index sequence

template<size_t... Is>
void print_index_sequence(index_sequence<Is...>) {
  (void)std::initializer_list<int>{ (std::cout << Is << " ", 0)...};
  std::cout << std::endl;
}



//  Permute reorder elemenents of M with the inversed permutation given by S
//  using namespace tao::seq;
//    static_assert(std::is_same< permute_t<index_sequence<3,0,2,1>, index_sequence<4,5,6,7> >,
//                                          index_sequence<5,7,6,4> >::value, "ooops" );

template<typename, typename>
struct permute;

template<std::size_t... Ns, typename M>
struct permute<index_sequence<Ns...>, M> {
  using tmp  = typename impl::permute_i<index_sequence<Ns...>, make_index_sequence<sizeof...(Ns)> >::type;
  using type = typename map<tmp, M>::type;
};

template<typename S, typename M>
using permute_t = typename permute<S, M>::type;


//  filter_out remove elements of B given by the indexes stored inA
//  using namespace tao::seq;
//    static_assert(std::is_same< filter_out<index_sequence<0,2>, index_sequence<4,5,6,7> >,
//                                          index_sequence<5,7> >::value, "ooops" );

template<typename, typename>
struct filter_out;

template<size_t... As>
struct filter_out<index_sequence<As...>, index_sequence<>> {
  using type = index_sequence<>;
};

template<size_t... As, size_t b, size_t... Bs>
struct filter_out<index_sequence<As...>, index_sequence<b, Bs...>> {
  constexpr static bool included = tao::seq::contains<size_t, b, As...>::value;
  using tail = typename filter_out<index_sequence<As...>, index_sequence<Bs...>>::type;
  using type = typename std::conditional<
      included,
      tail,
      typename tao::seq::concatenate<index_sequence<b>, tail>::type>::type;
};

//  Reverse the order of the index_sequence A
//  using namespace tao::seq;
//    static_assert(std::is_same< reverse<index_sequence<4,5,6,7> >,
//                                        index_sequence<7,6,5,4> >::value, "ooops" );

template<typename>
struct reverse;

template<>
struct reverse<index_sequence<>> {
  using type = index_sequence<>;
};

template<size_t a, size_t... As>
struct reverse<index_sequence<a, As...>> {
  using reversed = typename reverse<index_sequence<As...>>::type;
  using type = typename tao::seq::concatenate<reversed, index_sequence<a>>::type;
};

// Return the index_sequence containing the elementwise product of index_sequences A and B
//  using namespace tao::seq;
//    static_assert(std::is_same< product<index_sequence<0,5,7,1>, index_sequence<4,5,6,7> >,
//                                        index_sequence<0,25,42,4> >::value, "ooops" );
namespace impl {

struct prod {
  template<typename T, T A, T B>
  using apply = std::integral_constant<T, A * B>;
};
} // namespace impl

template<typename A, typename B>
using prod = zip<impl::prod, A, B>;

template<typename A, typename B>
using prod_t = typename prod<A, B>::type;


// Return the size_t containing the product of all element from index_sequences A
//  using namespace tao::seq;
//    static_assert(std::is_same< product_red<index_sequence<4,5,6,7> >,
//                                            index_sequence<840> >::value, "ooops" );
template<size_t... X>
constexpr auto prod_red(std::index_sequence<X...>) {
  size_t res = 1;
  (void) std::initializer_list<int>{(res *= X, 0)...};
  return res;
}

// Return the index_sequence containing the cumulative product of all element of index_sequences A
// except the first.
//  using namespace tao::seq;
//    static_assert(std::is_same< product_red<index_sequence<4,5,6,7> >,
//                                            index_sequence<210,42,7,1> >::value, "ooops" );
template<typename>
struct cum_prod;

template<>
struct cum_prod<index_sequence<>> {
  using type = index_sequence<>;
};

template<size_t a, size_t... X>
struct cum_prod<index_sequence<a, X...>> {
  using type = typename tao::seq::concatenate<
      index_sequence<prod_red(index_sequence<X...>{})>,
      typename cum_prod<index_sequence<X...>>::type>::type;
};

} // namespace seq

} // namespace tao


namespace keops {

template<size_t... Ix>
using index_sequence = tao::seq::integer_sequence<size_t, Ix...>;

namespace loop_impl {
template<typename, size_t... I>
struct Looper;

template<size_t... Is>
struct Looper<index_sequence<Is...>> {
  template<typename Func>
  constexpr static HOST_DEVICE void f(Func &&func) {
    func(index_sequence<Is...>{});
  }
};

template<size_t I, size_t... Is, size_t... PIs>
struct Looper<index_sequence<PIs...>, I, Is...> {
  template<std::size_t... Idx, typename Func>
  constexpr static HOST_DEVICE void f_help(index_sequence<Idx...>, Func &&func) {
    (void) std::initializer_list<int>{(Looper<index_sequence<PIs..., Idx>, Is...>::f(func), 0)...};
  }

  template<typename Func>
  constexpr static HOST_DEVICE void f(Func &&func) {
    f_help(tao::seq::make_index_sequence<I>{}, func);
  }

};

template<typename>
struct loop_t;

template<size_t... Is>
struct loop_t<index_sequence<Is...>> {
  using type = Looper<index_sequence<>, Is...>;
};

}

template<typename Is>
using loop = typename loop_impl::loop_t<Is>::type;

// Dummy class that stores the indices computes for tensordot
struct tensordot_indices {
  size_t out_indices;
  size_t a_indices;
  size_t b_indices;
};


template<class DIMFA, class DIMFB, class CONTFA, class CONTFB, class PERMUTE>
struct tensordot_parameters {

  // Left hand-side
  using indices_keepdim_a_t = typename tao::seq::filter_out<CONTFA, tao::seq::make_index_sequence<DIMFA::size()>>::type;
  using keepdim_a_t = typename tao::seq::map<indices_keepdim_a_t, DIMFA>::type;
  using contdim_a_t = typename tao::seq::map<CONTFA, DIMFA>::type;
#if C_CONTIGUOUS
  using list_stride_dim_a_t = typename tao::seq::cum_prod<DIMFA>::type;
#else
  using list_stride_dim_a_t = typename tao::seq::cum_prod<typename tao::seq::reverse<DIMFA>::type>::type;
#endif

  // Right hand-side
  using indices_keepdim_b_t = typename tao::seq::filter_out<CONTFB, tao::seq::make_index_sequence<DIMFB::size()>>::type;
  using keepdim_b_t = typename tao::seq::map<indices_keepdim_b_t, DIMFB>::type;
  using contdim_b_t = typename tao::seq::map<CONTFB, DIMFB>::type;
#if C_CONTIGUOUS
  using list_stride_dim_b_t = typename tao::seq::cum_prod<DIMFB>::type;
#else
  using list_stride_dim_b_t = typename tao::seq::cum_prod<typename tao::seq::reverse<DIMFB>::type>::type;
#endif


  static_assert(std::is_same<contdim_a_t, contdim_b_t>::value,
                "In TensorDot: contracting dimensions should be the same");

  // Output
  using keepdim_t = typename tao::seq::concatenate<keepdim_a_t, keepdim_b_t>::type;
#if C_CONTIGUOUS
  using list_stride_keepdim_t = typename tao::seq::cum_prod<typename tao::seq::permute<PERMUTE, keepdim_t>::type>::type;
#else
  using list_stride_keepdim_t = typename tao::seq::cum_prod<typename tao::seq::reverse<typename tao::seq::permute<PERMUTE, keepdim_t>::type>::type>::type;
  //using list_stride_keepdim_t = typename tao::seq::cum_prod<typename tao::seq::reverse<keepdim_t>::type>::type;
#endif
  constexpr static size_t dimout = tao::seq::prod_red(keepdim_t{});

  static_assert(std::is_same<typename tao::seq::permute<PERMUTE, PERMUTE>::type,
                             tao::seq::make_index_range<0, keepdim_t::size()>>::value,
                "In TensorDot: PERMUTE should be a permutation index_sequence.");

  // Loop: in this code we choose to loop on the keepdims first and then on the contraction dims.
  using loopdim_t = typename tao::seq::concatenate<keepdim_t, contdim_a_t>::type;

  constexpr static size_t dimloop = tao::seq::prod_red(loopdim_t{});

  using ala = typename tao::seq::concatenate<tao::seq::make_index_range<0, keepdim_a_t::size()>,
                                             tao::seq::make_index_range<keepdim_t::size(), dimloop>>::type;

  using ali = typename tao::seq::concatenate<indices_keepdim_a_t, CONTFA>::type;

  using list_indices_a_intot = typename tao::seq::permute<ali, ala>::type;

  using bla = typename tao::seq::concatenate<tao::seq::make_index_range<keepdim_a_t::size(), keepdim_t::size()>,
                                             tao::seq::make_index_range<keepdim_t::size(), dimloop>>::type;
  using bli = typename tao::seq::concatenate<indices_keepdim_b_t, CONTFB>::type;

  using list_indices_b_intot = typename tao::seq::permute<bli, bla>::type;

  // used to compute the Gradient
  using list_indices_keepdim_a_inout = typename tao::seq::make_index_range<0, keepdim_a_t::size()>;
  using reordered_contfa = typename tao::seq::permute<typename tao::seq::sort_index_asc<CONTFB>::type, CONTFA>::type;
  using reordered_keepdim_a = typename tao::seq::permute<typename tao::seq::sort_index_asc<typename tao::seq::map<list_indices_keepdim_a_inout, PERMUTE>::type>::type, indices_keepdim_a_t>::type;
  using moveaxis_a = typename tao::seq::concatenate<reordered_keepdim_a, reordered_contfa>::type;

  using list_indices_keepdim_b_inout = typename tao::seq::make_index_range<keepdim_a_t::size(), keepdim_t::size()>;
  using reordered_contfb = typename tao::seq::permute<typename tao::seq::sort_index_asc<CONTFA>::type, CONTFB>::type;
  using reordered_keepdim_b = typename tao::seq::permute<typename tao::seq::sort_index_asc<typename tao::seq::map<list_indices_keepdim_b_inout, PERMUTE>::type>::type, indices_keepdim_b_t>::type;
  using moveaxis_b = typename tao::seq::concatenate<reordered_keepdim_b, reordered_contfb>::type;

  template<class IND>
  HOST_DEVICE constexpr static tensordot_indices compute_tensordot_indices(IND) {

    // a_indices
    using list_indices_a = typename tao::seq::map<list_indices_a_intot,
                                                  IND>::type;

#if C_CONTIGUOUS
    size_t a_indices = tao::seq::sum<tao::seq::prod_t<list_stride_dim_a_t, list_indices_a>>::value;
#else
    size_t a_indices = tao::seq::sum<tao::seq::prod_t<list_stride_dim_a_t, typename tao::seq::reverse<list_indices_a>::type>>::value;
#endif


    // b_indices
    using list_indices_b = typename tao::seq::map<list_indices_b_intot,
                                                  IND>::type;
#if C_CONTIGUOUS
    size_t b_indices = tao::seq::sum<tao::seq::prod_t<list_stride_dim_b_t,list_indices_b>>::value;
#else
    size_t b_indices = tao::seq::sum<tao::seq::prod_t<list_stride_dim_b_t, typename tao::seq::reverse<list_indices_b>::type>>::value;
#endif
    // out_indices
    // using list_indices_keepdim = typename tao::seq::map<tao::seq::make_index_range<0, keepdim_t::size()>, IND>::type;
//    using list_indices_keepdim = typename tao::seq::map<PERMUTE, IND>::type;
    using list_indices_keepdim = typename tao::seq::permute<PERMUTE, IND>::type;

#if C_CONTIGUOUS
    size_t out_indices = tao::seq::sum<tao::seq::prod_t<list_stride_keepdim_t, list_indices_keepdim>>::value;
#else
    size_t out_indices = tao::seq::sum<tao::seq::prod_t<list_stride_keepdim_t, typename tao::seq::reverse<list_indices_keepdim>::type>>::value;
#endif
    /*std::cout << "list_stride_keepdim_t: ";tao::seq::print_index_sequence(list_stride_keepdim_t{});
    std::cout << "permute : "; tao::seq::print_index_sequence(PERMUTE{});
    std::cout << "permute(permute): "; tao::seq::print_index_sequence(typename tao::seq::permute<PERMUTE, PERMUTE>::type{});
    std::cout << "permute(permute): "; tao::seq::print_index_sequence(tao::seq::make_index_range<0, keepdim_t::size()>{});
    std::cout << "moveaxis_a: "; tao::seq::print_index_sequence(moveaxis_a{});
    std::cout << "moveaxis_b: "; tao::seq::print_index_sequence(moveaxis_b{});

    std::cout << "dimout: " << dimout  << " ("<< out_indices << " " << a_indices << " " << b_indices << ")" << std::endl;

    tao::seq::print_index_sequence(typename tao::seq::permute<PERMUTE, PERMUTE>::type{});
    tao::seq::print_index_sequence(tao::seq::make_index_range<0, keepdim_t::size()>{});
    std::cout << "--" << std::endl; */
    return tensordot_indices{out_indices, a_indices, b_indices};
  }

  template<typename Func>
  struct compute_tensordot_indices_t {
    template<size_t... Is>
    HOST_DEVICE void operator()(index_sequence<Is...> x) {
      _f(compute_tensordot_indices(x));
    }

    Func &_f;
    HOST_DEVICE compute_tensordot_indices_t(Func &&f) : _f(f) {}
  };

  template<typename Func>
  static HOST_DEVICE auto compute_tensordot_indices_apply(Func &&f) {
    return compute_tensordot_indices_t<Func>(std::forward<Func>(f));
  }

};




/////////////////////////////////////////////////////////////////////////
////              Tensor dot product      A : b                      ////
/////////////////////////////////////////////////////////////////////////


template<class A, class B, class DIMFA, class DIMFB, class CONTFA, class CONTFB, class PERMUTEDIMOUT=std::make_index_sequence<DIMFA::size() + DIMFB::size() - 2 * CONTFA::size()>>
struct TensorDot : BinaryOp<TensorDot, A, B, DIMFA, DIMFB, CONTFA, CONTFB, PERMUTEDIMOUT> {
  // A is vector of size p ** n, interpreted as matrix (column major), B is vector of size p ** m, interpreted as column vector
  // n=3 and m=2 are assume to be known
  // output is vector of size n

  static_assert(DIMFA::size() > 0, "Please provide a non empty DIMA");
  static_assert(DIMFB::size() > 0, "Please provide a non empty DIMB");
  static_assert(tao::seq::prod_red(DIMFA{}) == A::DIM, "DIMA is not consistant with dimension of A");
  static_assert(tao::seq::prod_red(DIMFB{}) == B::DIM, "DIMB is not consistant with dimension of B");

  using parameters = tensordot_parameters<DIMFA, DIMFB, CONTFA, CONTFB, PERMUTEDIMOUT>;

  static const int DIM = parameters::dimout;

  static void PrintIdString(std::stringstream &str) {
    str << ":";
  }

  static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *inA, __TYPE__ *inB) {
#pragma unroll
    for (int i =0 ; i < DIM ; i ++)
      out[i] = 0;

    loop<typename parameters::loopdim_t >::f(parameters::compute_tensordot_indices_apply([&out,&inA,&inB](tensordot_indices td){
      out[td.out_indices] += inA[td.a_indices] * inB[td.b_indices];
    }));
  }

  template<class V, class GRADIN>
  using DiffTA = typename A::template DiffT<V, GRADIN>;

  template<class V, class GRADIN>
  using DiffTB = typename B::template DiffT<V, GRADIN>;


  template<class V, class GRADIN>
  using DiffT = Add<
      DiffTA<V, TensorDot<GRADIN, B,
                          typename tao::seq::permute<PERMUTEDIMOUT, typename parameters::keepdim_t>::type,                    // 3
                          DIMFB,                                                                                              // 4 2
                          typename tao::seq::map<typename parameters::list_indices_keepdim_b_inout, PERMUTEDIMOUT>::type,     // .
                          typename parameters::indices_keepdim_b_t,                                                           // .
                          typename parameters::moveaxis_a>>,                                                                  // 1, 2 0
      DiffTB<V, TensorDot<GRADIN, A,
                          typename tao::seq::permute<PERMUTEDIMOUT, typename parameters::keepdim_t>::type,                    // 3
                          DIMFA,                                                                                              // 2, 3, 4
                          typename tao::seq::map<typename parameters::list_indices_keepdim_a_inout, PERMUTEDIMOUT>::type,    //0
                          typename parameters::indices_keepdim_a_t,                                                           //1
                          typename parameters::moveaxis_b>>                                                                   // . , 0 1
  >;


};

}
