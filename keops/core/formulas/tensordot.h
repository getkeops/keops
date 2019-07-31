#pragma once

#include <tao/seq/integer_sequence.hpp>
#include <tao/seq/contains.hpp>
#include <tao/seq/concatenate.hpp>
#include <tao/seq/map.hpp>
#include <tao/seq/zip.hpp>
#include <tao/seq/select.hpp>
#include <tao/seq/sum.hpp>
#include <tao/seq/make_integer_range.hpp>


namespace tao {
namespace seq {
namespace impl {
struct prod {
    template <typename T, T A, T B>
    using apply = std::integral_constant<T, A * B>;
};

} // namespace impl

template <typename A, typename B>
using prod = zip<impl::prod, A, B>;

template <typename A, typename B>
using prod_t = typename prod<A, B>::type;

template <typename, typename>
struct filter_out;

template <size_t... As>
struct filter_out<index_sequence<As...>, index_sequence<>> {
    using type = index_sequence<>;
};

template <size_t... As, size_t b, size_t... Bs>
struct filter_out<index_sequence<As...>, index_sequence<b, Bs...>> {
    constexpr static bool included = tao::seq::contains<size_t, b, As...>::value;
    using tail = typename filter_out<index_sequence<As...>, index_sequence<Bs...>>::type;
    using type = typename std::conditional<
                 included,
                 tail,
                 typename tao::seq::concatenate<index_sequence<b>, tail>::type>::type;
};

template <typename>
struct reverse;

template <>
struct reverse<index_sequence<>> {
    using type = index_sequence<>;
};

template <size_t a, size_t... As>
struct reverse<index_sequence<a, As...>> {
    using reversed = typename reverse<index_sequence<As...>>::type;
    using type = typename tao::seq::concatenate<reversed, index_sequence<a>>::type;
};

template <size_t... X>
constexpr auto prod_red(std::index_sequence<X...>) {
    size_t res = 1;
    (void)std::initializer_list<int>{ (res *= X, 0)... };
    return res;
}

template <typename>
struct cum_prod;

template <>
struct cum_prod<index_sequence<>> {
    using type = index_sequence<>;
};

template <size_t a, size_t... X>
struct cum_prod<index_sequence<a, X...>> {
    using type = typename tao::seq::concatenate<
                 index_sequence<prod_red(index_sequence<X...> {})>,
                 typename cum_prod<index_sequence<X...>>::type>::type;
};

} // namespace seq

} // namespace tao


namespace keops {


template <size_t... Ix>
using index_sequence = tao::seq::integer_sequence<size_t, Ix...>;

namespace loop_impl {
    template<typename, size_t... I>
    struct Looper;

    template<size_t... Is>
    struct Looper<index_sequence<Is...>> {
        template <typename Func>
        constexpr static HOST_DEVICE void f(Func&& func) {
            func(index_sequence<Is...>{});
        }
    };

    template <size_t I, size_t... Is, size_t... PIs>
    struct Looper<index_sequence<PIs...>, I,Is...> {
        template <std::size_t... Idx, typename Func>
        constexpr static HOST_DEVICE void f_help (index_sequence<Idx...>, Func&& func) {
            (void)std::initializer_list<int>{ (Looper<index_sequence<PIs...,Idx>,Is...>::f(func), 0)... };
        }

        template <typename Func>
        constexpr static HOST_DEVICE void f(Func&& func) {
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

struct KD {
    size_t I;
    size_t a;
    size_t b;
};

//template <typename, typename, typename, typename> struct tensordot_parameters;

template <class DIMFA, class DIMFB, class CONTFA, class CONTFB>
struct tensordot_parameters {

    constexpr static auto size_listdim_a = DIMFA::size();
    using indices_dim_a_t = tao::seq::make_index_sequence<size_listdim_a>;
    using indices_keepdim_a_t = typename tao::seq::filter_out<
                                CONTFA,
                                indices_dim_a_t>::type;
    using keepdim_a_t = typename tao::seq::map<
                                indices_keepdim_a_t,
                                DIMFA>::type;
    using cont_dim_a_t = typename tao::seq::map<
                                CONTFA,
                                DIMFA>::type;

    constexpr static auto size_listdim_b = CONTFB::size();
    using indices_dim_b_t = tao::seq::make_index_sequence<size_listdim_b>;
    using indices_keepdim_b_t = typename tao::seq::filter_out<
                                CONTFB,
                                indices_dim_b_t>::type;
    using keepdim_b_t = typename tao::seq::map<
                                indices_keepdim_b_t,
                                DIMFB>::type;
    using cont_dim_b_t = typename tao::seq::map<
                                CONTFB,
                                DIMFB>::type;

    static_assert(std::is_same<cont_dim_a_t, cont_dim_b_t>::value, "Contracting dimensions should  be the same");

    constexpr static auto size_list_contdim = cont_dim_a_t::size();

    using dim_keep_t = typename tao::seq::concatenate<
                                 keepdim_a_t,
                                 keepdim_b_t>::type;
    using dim_tot_t = typename tao::seq::concatenate<
                                 dim_keep_t,
                                 cont_dim_a_t>::type;
    using list_stride_dim_a_t = typename tao::seq::cum_prod<DIMFA>::type;
    using list_stride_dim_b_t = typename tao::seq::cum_prod<DIMFB>::type;
    using list_stride_keepdim_t = typename tao::seq::cum_prod<
                                 typename tao::seq::concatenate<
                                 keepdim_a_t,
                                 typename tao::seq::reverse<keepdim_b_t>::type>::type>::type;

    using list_indices_strides_tot = typename tao::seq::cum_prod<dim_tot_t>::type;

    constexpr static size_t dimout = tao::seq::prod_red(dim_keep_t{});
    constexpr static size_t dimtot = tao::seq::prod_red(dim_tot_t{});

    template <class IND>
    constexpr static KD kdvar(IND) {

        // Kda first pass
        using list_indices_keepdim_a = typename tao::seq::map<
                                       tao::seq::make_index_range<0, indices_keepdim_a_t::size()>,
                                       IND>::type;

        using list_indices_strides_keepdim_a = typename tao::seq::map<
                                               tao::seq::make_index_range<0, indices_keepdim_a_t::size()>,
                                               list_stride_dim_a_t>::type;

        size_t kda = tao::seq::sum<
                     tao::seq::prod_t<
                     list_indices_strides_keepdim_a,
                     list_indices_keepdim_a>>::value;

        // Kdb first pass
        using list_indices_keepdim_b = typename tao::seq::map<
                                       tao::seq::make_index_range<indices_keepdim_a_t::size(), indices_keepdim_a_t::size() + indices_keepdim_b_t::size()>,
                                       IND>::type;

        using list_indices_strides_keepdim_b = typename tao::seq::map<
                                               tao::seq::make_index_range<0, indices_keepdim_b_t::size()>,
                                               typename tao::seq::reverse<list_stride_dim_b_t>::type>::type;

        size_t kdb = tao::seq::sum<
                     tao::seq::prod_t<
                     list_indices_strides_keepdim_b,
                     list_indices_keepdim_b>>::value;

        // Contdim
        using list_indices_contdim = typename tao::seq::map<
                                     tao::seq::make_index_range<dim_keep_t::size(), IND::size()>,
                                     IND>::type;

        using list_indices_strides_contdim_a = typename tao::seq::map<
                                               tao::seq::make_index_range<indices_keepdim_a_t::size(), indices_keepdim_a_t::size() + list_indices_contdim::size()>,
                                               list_stride_dim_a_t>::type;

        using list_indices_strides_contdim_b = typename tao::seq::map<
                                               tao::seq::make_index_range<0, list_indices_contdim::size()>,
                                               list_stride_dim_b_t>::type;

        kda += tao::seq::sum<
               tao::seq::prod_t<
               list_indices_strides_contdim_a,
               list_indices_contdim>>::value;

        kdb += tao::seq::sum<
               tao::seq::prod_t<
               list_indices_strides_contdim_b,
               list_indices_contdim>>::value;

        using list_indices_keepdim = typename tao::seq::map<
                                     tao::seq::make_index_range<0, dim_keep_t::size()>,
                                     IND>::type;

        size_t I = tao::seq::sum<
                   tao::seq::prod_t<
                   list_stride_keepdim_t,
                   list_indices_keepdim>>::value;

      return KD{I, kda, kdb};
    }


    template<typename Func>
    struct kdvar_t {
        template<size_t... Is>
        HOST_DEVICE void operator()(index_sequence<Is...> x) {
            _f(kdvar(x));
        }

        Func& _f;
        HOST_DEVICE kdvar_t(Func&& f) : _f(f) {}
    };

    template<typename Func>
    static HOST_DEVICE auto kdvar_apply(Func&& f) {
        return kdvar_t<Func>(std::forward<Func>(f));
    }

};

}
