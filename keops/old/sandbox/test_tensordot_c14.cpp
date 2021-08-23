
// g++ -I.. -D__TYPE__=float -std=c++14 -O3 -o build/test_tensordot_c14 test_tensordot_c14.cpp

#include <array>
#include <iostream>
#include <iomanip>

#include <lib/sequences/include/tao/seq/integer_sequence.hpp>
#include <lib/sequences/include/tao/seq/contains.hpp>
#include <lib/sequences/include/tao/seq/concatenate.hpp>
#include <lib/sequences/include/tao/seq/map.hpp>
#include <lib/sequences/include/tao/seq/zip.hpp>
#include <lib/sequences/include/tao/seq/select.hpp>
#include <lib/sequences/include/tao/seq/sum.hpp>
#include <lib/sequences/include/tao/seq/make_integer_range.hpp>

namespace tao
{
namespace seq
{
namespace impl
{
struct prod
{
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
struct filter_out<index_sequence<As...>, index_sequence<>>
{
    using type = index_sequence<>;
};

template <size_t... As, size_t b, size_t... Bs>
struct filter_out<index_sequence<As...>, index_sequence<b, Bs...>>
{
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
struct reverse<index_sequence<>>
{
    using type = index_sequence<>;
};

template <size_t a, size_t... As>
struct reverse<index_sequence<a, As...>>
{
    using reversed = typename reverse<index_sequence<As...>>::type;
    using type = typename tao::seq::concatenate<reversed, index_sequence<a>>::type;
};

template <size_t... X>
constexpr auto prod_red(index_sequence<X...>)
{
    constexpr std::array<size_t, sizeof...(X)> x{X...};
    size_t res = 1;
    for (size_t i = 0; i < sizeof...(X); i++)
        res *= x[i];
    return res;
}

template <typename>
struct cum_prod;

template <>
struct cum_prod<index_sequence<>>
{
    using type = index_sequence<>;
};

template <size_t a, size_t... X>
struct cum_prod<index_sequence<a, X...>>
{
    using type = typename tao::seq::concatenate<
        index_sequence<prod_red(index_sequence<X...>{})>,
        typename cum_prod<index_sequence<X...>>::type>::type;
};

} // namespace seq

} // namespace tao

template <size_t... Ix>
using index_sequence = tao::seq::integer_sequence<size_t, Ix...>;

#define Ind(...) index_sequence<__VA_ARGS__>

template <size_t... Ix>
constexpr auto make_array_from_seq(index_sequence<Ix...>) -> std::array<size_t, sizeof...(Ix)>
{
    return std::array<size_t, sizeof...(Ix)>{Ix...};
}

struct KD
{
    size_t I;
    size_t a;
    size_t b;
};

template <typename, typename, typename, typename>
struct tensordot_parameters;

template <size_t... DIMFA, size_t... DIMFB, size_t... CONTFA, size_t... CONTFB>
struct tensordot_parameters<
    index_sequence<DIMFA...>,
    index_sequence<DIMFB...>,
    index_sequence<CONTFA...>,
    index_sequence<CONTFB...>>
{
    constexpr static auto size_listdim_a = sizeof...(DIMFA);
    using indices_dim_a_t = tao::seq::make_index_sequence<size_listdim_a>;
    using indices_keepdim_a_t = typename tao::seq::filter_out<
        index_sequence<CONTFA...>,
        indices_dim_a_t>::type;
    using keepdim_a_t = typename tao::seq::map<
        indices_keepdim_a_t,
        index_sequence<DIMFA...>>::type;
    using cont_dim_a_t = typename tao::seq::map<
        index_sequence<CONTFA...>,
        index_sequence<DIMFA...>>::type;

    constexpr static auto size_listdim_b = sizeof...(DIMFB);
    using indices_dim_b_t = tao::seq::make_index_sequence<size_listdim_b>;
    using indices_keepdim_b_t = typename tao::seq::filter_out<
        index_sequence<CONTFB...>,
        indices_dim_b_t>::type;
    using keepdim_b_t = typename tao::seq::map<
        indices_keepdim_b_t,
        index_sequence<DIMFB...>>::type;
    using cont_dim_b_t = typename tao::seq::map<
        index_sequence<CONTFB...>,
        index_sequence<DIMFB...>>::type;

    static_assert(std::is_same<cont_dim_a_t, cont_dim_b_t>::value, "Contracting dimensions should  be the same");
    constexpr static auto size_list_contdim = cont_dim_a_t::size();

    using dim_keep_t = typename tao::seq::concatenate<keepdim_a_t, keepdim_b_t>::type;
    using dim_tot_t = typename tao::seq::concatenate<dim_keep_t, cont_dim_a_t>::type;
    using list_stride_dim_a_t = typename tao::seq::cum_prod<index_sequence<DIMFA...>>::type;
    using list_stride_dim_b_t = typename tao::seq::cum_prod<index_sequence<DIMFB...>>::type;
    using list_stride_keepdim_t = typename tao::seq::cum_prod<
        typename tao::seq::concatenate<
            keepdim_a_t,
            typename tao::seq::reverse<keepdim_b_t>::type>::type>::type;

    using list_indices_strides_tot = typename tao::seq::cum_prod<dim_tot_t>::type;

    constexpr static size_t dimout = tao::seq::prod_red(dim_keep_t{});
    constexpr static size_t dimtot = tao::seq::prod_red(dim_tot_t{});
    template <size_t... IND>
    constexpr static KD kdvar(index_sequence<IND...>)
    {
        using list_indices_tot = index_sequence<IND...>;

        // Kda first pass
        using list_indices_keepdim_a = typename tao::seq::map<
            tao::seq::make_index_range<0, indices_keepdim_a_t::size()>,
            list_indices_tot>::type;

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
            list_indices_tot>::type;

        using list_indices_strides_keepdim_b = typename tao::seq::map<
            tao::seq::make_index_range<0, indices_keepdim_b_t::size()>,
            typename tao::seq::reverse<list_stride_dim_b_t>::type>::type;

        size_t kdb = tao::seq::sum<
            tao::seq::prod_t<
                list_indices_strides_keepdim_b,
                list_indices_keepdim_b>>::value;

        // Contdim
        using list_indices_contdim = typename tao::seq::map<
            tao::seq::make_index_range<dim_keep_t::size(), list_indices_tot::size()>,
            list_indices_tot>::type;

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
            list_indices_tot>::type;

        size_t I = tao::seq::sum<
            tao::seq::prod_t<
                list_stride_keepdim_t,
                list_indices_keepdim>>::value;

        return KD{I, kda, kdb};
    }

    template <size_t dim_i, size_t... IND, std::enable_if_t<sizeof...(IND) == list_indices_strides_tot::size()> * = nullptr>
    static constexpr auto get_indices()
    {
        using internal = typename tao::seq::reverse<index_sequence<IND...>>::type;
        return kdvar(internal{});
        //    return kdvar(index_sequence<IND...>{});
    }

    template <size_t dim_i, size_t... IND, std::enable_if_t<sizeof...(IND) < list_indices_strides_tot::size()> * = nullptr>
    static constexpr auto get_indices()
    {
        return get_indices<dim_i % tao::seq::select<sizeof...(IND), list_indices_strides_tot>::value,
                           dim_i / tao::seq::select<sizeof...(IND), list_indices_strides_tot>::value,
                           IND...>();
    }

    using dimout_seq = tao::seq::make_index_sequence<dimtot>;
    template <size_t... DIMOUT_SEQ>
    static constexpr auto get_KD(index_sequence<DIMOUT_SEQ...>)
    {
        return std::array<KD, sizeof...(DIMOUT_SEQ)>{(get_indices<DIMOUT_SEQ>())...};
    }

    constexpr static std::array<KD, dimtot> kd_seq = get_KD(dimout_seq{});
};

///---------------------------------------------

using DimFa = Ind(2, 2, 2);
using DimFb = Ind(2, 2);

using ContFa = Ind(2);
using ContFb = Ind(0);
using ContFa4 = Ind(1, 2);
using ContFb4 = Ind(0, 1);

using DimFa6 = Ind(5, 4, 3);
using DimFb6 = Ind(4, 3, 2);

using ContFa6 = Ind(1, 2);
using ContFb6 = Ind(0, 1);

using ContFa8 = Ind(0, 1, 2);

int main()
{

    double FA[8] = {4.4, 5.4, 6.2, 6.5, 7.5, 6.1, 8.7, 1.3};
    double FB[4] = {1.4, 1.2, 1.5, 1.22};

    using ma4 = tensordot_parameters<
        DimFa,
        DimFb,
        ContFa,
        ContFb>;

    double out_td[ma4::dimout];
    std::fill(out_td, out_td + ma4::dimout, 0);

    constexpr std::array<KD, ma4::dimtot> list_kd = ma4::kd_seq;
    for (auto kd : list_kd)
    {
        //  std::cout << kd.a << " " << kd.b <<  " " << kd.I << std::endl;
        out_td[kd.I] += FA[kd.a] * FB[kd.b];
    }

    // --------

    double out_loop[2 * 2 * 2] = {0, 0, 0, 0, 0, 0, 0, 0};

    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 2; j++)
            for (size_t k = 0; k < 2; k++)
                for (size_t l = 0; l < 2; l++)
                {
                    // size_t kda = 4 * i + 2 * j + l;
                    // size_t kdb = l * 2 + k;
                    // size_t I = 4 * i + 2 * j + k;
                    // std::cout << "(" << i << "," << j <<  "," << k <<  "," << l <<")     " << kda << " " << kdb << " " << I << std::endl;
                    out_loop[4 * i + 2 * j + k] += FA[4 * i + 2 * j + l] * FB[l * 2 + k];
                }
    double res = 0;
    for (auto i = 0; i < 8; i++)
    {
        // std::cout << "out_tensordot = " << out_td[i] << std::endl;
        // std::cout << "out_loop = " << out_loop[i] << std::endl;
        res += (out_td[i] - out_loop[i]) * (out_td[i] - out_loop[i]);
    }

    std::cout << "error = " << res << std::endl
              << std::endl;

    // -------------------------------------------------------------------------------------------------------------------
    using ma5 = tensordot_parameters<
        DimFa,
        DimFb,
        ContFa4,
        ContFb4>;

    double out5[ma5::dimout];
    std::fill(out5, out5 + ma5::dimout, 0);

    constexpr std::array<KD, ma5::dimtot> list_kd5 = ma5::kd_seq;
    for (auto kd : list_kd5)
    {
        //  std::cout << kd.a << " " << kd.b <<  " " << kd.I << std::endl;
        out5[kd.I] += FA[kd.a] * FB[kd.b];
    }

    // -------------

    double out4[2] = {0, 0};

    for (size_t i = 0; i < 2; i++)
        for (size_t k = 0; k < 2; k++)
            for (size_t l = 0; l < 2; l++)
            {
                    //size_t kda = 4 * i + 2 * k + l;
                    //size_t kdb = k * 2 + l;
                    //size_t I = i;
                    //std::cout << I << "," << kda << "," << kdb << "," << std::endl;
                out4[i] += FA[4 * i + 2 * k + l] * FB[k * 2 + l];
            }

    double res2 = 0;
    for (auto i = 0; i < 2; i++)
    {
        // std::cout << "out4 = " << out4[i] << std::endl;
        // std::cout << "out5 = " << out5[i] << std::endl;
        res2 += (out4[i] - out5[i]) * (out4[i] - out5[i]);
    }

    std::cout << "error = " << res2 << std::endl
              << std::endl;

    // -------------------------------------------------------------------------------------------------------------------

    double FAA[60] = {7, 9, 9, 5, 8, 3, 6, 9, 6, 0, 5, 7, 3, 4, 3, 5, 3, 3, 0, 9, 9, 6, 0, 3, 3, 7, 0, 8, 6, 0, 6, 1, 3, 1, 4, 7, 3, 9, 8, 8, 3, 7, 2, 3, 1, 9, 5, 7, 7, 5, 9, 7, 0, 1, 9, 7, 5, 0, 3, 8};
    double FBB[24] = {6, 4, 2, 9, 9, 5, 1, 6, 7, 8, 2, 4, 1, 9, 7, 8, 5, 4, 3, 2, 3, 8, 5, 7};

    using ma6 = tensordot_parameters<
        DimFa6,
        DimFb6,
        ContFa6,
        ContFb6>;

    double out6[ma6::dimout];
    std::fill(out6, out6 + ma6::dimout, 0);

    constexpr std::array<KD, ma6::dimtot> list_kd6 = ma6::kd_seq;
    for (auto kd : list_kd6)
    {
        //  std::cout << kd.a << " " << kd.b <<  " " << kd.I << std::endl;
        out6[kd.I] += FAA[kd.a] * FBB[kd.b];
    }

    // -------------

    double out7[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    for (size_t i = 0; i < 5; i++)
        for (size_t j = 0; j < 2; j++)
            for (size_t k = 0; k < 4; k++)
                for (size_t l = 0; l < 3; l++)
                {
                    // size_t kda = 12 * i + 3 * k + l;
                    // size_t kdb = 6 * k + 2 * l + j;
                    // size_t I = 2 * i + j;
                    // std::cout << "(" << i << "," << j <<  "," << k <<  "," << l <<")     " << kda << " " << kdb << " " << I << std::endl;
                    out7[2 * i + j] += FAA[12 * i + 3 * k + l] * FBB[6 * k + 2 * l + j];
                }

    double res3 = 0;
    for (auto i = 0; i < 10; i++)
    {
        // std::cout << "out7 = " << out7[i] << std::endl;
        // std::cout << "out6 = " << out6[i] << std::endl;
        res3 += (out6[i] - out7[i]) * (out6[i] - out7[i]);
    }

    std::cout << "error = " << res3 << std::endl
              << std::endl;

    // -------------------------------------------------------------------------------------------------------------------

    using ma8 = tensordot_parameters<
        DimFa6,
        DimFa6,
        ContFa8,
        ContFa8>;

    double out8[ma8::dimout];
    std::fill(out8, out8 + ma8::dimout, 0);

    constexpr std::array<KD, ma8::dimtot> list_kd8 = ma8::kd_seq;
    for (auto kd : list_kd8)
    {
        // std::cout << kd.a << " " << kd.b <<  " " << kd.I << std::endl;
        out8[kd.I] += FAA[kd.a] * FAA[kd.b];
    }

    // -------------

    double out9[1] = {0};

    for (size_t i = 0; i < 5; i++)
        for (size_t j = 0; j < 4; j++)
            for (size_t k = 0; k < 3; k++)
            {
                // size_t kda = 12 * i + 3 * j + k;
                // size_t kdb = 12 * i + 3 * j + k;
                // size_t I = 0;
                // std::cout << "(" << i << "," << j <<  "," << k <<  "," << l <<")     " << kda << " " << kdb << " " << I << std::endl;
                out9[0] += FAA[12 * i + 3 * j + k] * FAA[12 * i + 3 * j + k];
            }

    double res4 = 0;
    for (auto i = 0; i < 1; i++)
    {
        // std::cout << "out9 = " << out9[i] << std::endl;
        // std::cout << "out8 = " << out8[i] << std::endl;
        res4 += (out8[i] - out9[i]) * (out8[i] - out9[i]);
    }

    std::cout << "error = " << res4 << std::endl
              << std::endl;

    return 0;
}
