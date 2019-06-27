#include <iostream>
#include <tuple>
#include <array>

constexpr std::size_t operator"" _z(unsigned long long n)
{
    return n;
}

#define Ind(...) std::index_sequence<__VA_ARGS__>

// TensorDot(A,B,Ind(2,2,2), Ind(2,2), Ind(0,1), Ind(0,1))
// TensorDot(A,B,{2,2,2}, {2,2}, {0,1}, {0,1))

using DimFa = Ind(2, 2, 2);
using DimFb = Ind(2, 2);
using ContFa = Ind(2);
using ContFb = Ind(0);



template <size_t... DIMFA, size_t... DIMFB, size_t... CONTFA, size_t... CONTFB>
static constexpr std::tuple<std::array<size_t, sizeof...(DIMFA)>,
                            std::array<size_t, sizeof...(DIMFB)>,
                            std::array<size_t, sizeof...(DIMFA) - sizeof...(CONTFA)>,
                            std::array<size_t, sizeof...(DIMFB) - sizeof...(CONTFB)>,
                            std::array<size_t, sizeof...(DIMFA) + sizeof...(DIMFB) - sizeof...(CONTFA) - sizeof...(CONTFB)>,
                            std::array<size_t, sizeof...(DIMFA) + sizeof...(DIMFB) - sizeof...(CONTFA)>>
build_array4(std::index_sequence<DIMFA...>,
             std::index_sequence<DIMFB...>,
             std::index_sequence<CONTFA...>,
             std::index_sequence<CONTFB...>) noexcept
{

    // Cast index_sequence to array
    auto dimfa = std::array<size_t, sizeof...(DIMFA)>{DIMFA...};
    auto dimfb = std::array<size_t, sizeof...(DIMFB)>{DIMFB...};
    auto contfa = std::array<size_t, sizeof...(CONTFA)>{CONTFA...};
    auto contfb = std::array<size_t, sizeof...(CONTFB)>{CONTFB...};

    // get size of the contraction
    constexpr size_t number_of_dim_a = dimfa.size() - contfa.size();
    constexpr size_t number_of_dim_b = dimfb.size() - contfb.size();

    std::array<size_t, number_of_dim_b + number_of_dim_a> dim_keep{};
    std::array<size_t, number_of_dim_a> dim_keep_a{};
    std::array<size_t, number_of_dim_b> dim_keep_b{};

    for (auto i = 0; i < number_of_dim_a; i++)
    {
        dim_keep_a[i] = dimfa[i];
        dim_keep[i] = dimfa[i];
    }

    for (auto i = 0; i < number_of_dim_b; i++)
    {
        dim_keep_b[i] = dimfb[contfb.size() + i];
        dim_keep[i + number_of_dim_a] = dim_keep_b[i];
    }

    std::array<size_t, contfa.size()> dim_sum{};
    for (auto i = 0; i < contfa.size(); i++)
    {
        dim_sum[i] = dimfa[number_of_dim_a + i];
    }

    std::array<size_t, contfa.size() + dim_keep.size()> dim_tot{};
    for (auto i = 0; i < dim_keep.size(); i++)
    {
        dim_tot[i] = dim_keep[i];
    }
    for (auto i = 0; i < contfa.size(); i++)
    {
        dim_tot[dim_keep.size() + i] = dim_sum[i];
    }

    return std::make_tuple(dimfa, dimfb,
                           dim_keep_a, dim_keep_b, dim_keep,
                           dim_tot);
}

template <size_t shape_index, size_t shape_size>
struct Looper
{
    template <typename Functor>
    void operator()(const std::array<size_t, shape_size> &shape, Functor functor)
    {
        for (size_t index = 0; index < shape[shape_index]; ++index)
        {
            Looper<shape_index + 1, shape_size>()(
                shape,
                [index, &functor](auto... tail) { functor(std::tuple_cat(std::tuple<size_t>(index), tail...)); });
        }
    }
};

template <size_t shape_size>
struct Looper<shape_size, shape_size>
{
    template <typename Functor>
    void operator()(const std::array<size_t, shape_size> &, Functor functor)
    {
        functor();
    }
};

template <size_t shape_size, typename Functor>
void loop(const std::array<size_t, shape_size> &shape, Functor functor)
{
    Looper<0, shape_size>()(shape, functor);
}

//constexpr unsigned cilog2(unsigned val) { return val ? 1 + cilog2(val >> 1) : -1; }

template <size_t size1, size_t size2>
constexpr std::tuple<size_t, size_t, size_t> kd(std::array<size_t, size1> dim_a, std::array<size_t, size2> dim_b, size_t i, size_t j, size_t k)
{
    size_t kda = dim_a[1] * dim_a[2] * i + dim_a[2] * j;

    size_t kdb = k;
    size_t I = kda + kdb;
    return std::make_tuple(I, kda, kdb);
}

template <size_t size_a, size_t size_b, size_t ia, size_t ib, size_t oa, size_t ob, size_t on, size_t in>
constexpr std::tuple<size_t, size_t, size_t> kdvar(std::array<size_t, size_a> dim_a,
                                                   std::array<size_t, size_b> dim_b,
                                                   std::array<size_t, ia> list_dim_a,     // {2,1}
                                                   std::array<size_t, ib> list_dim_b,     // {2}
                                                   std::array<size_t, oa> list_dim_aout,     // {}
                                                   std::array<size_t, ob> list_dim_bout,     // {1}
                                                   std::array<size_t, ia> list_indices_a, // {0,1}
                                                   std::array<size_t, ib> list_indices_b, // {2}
                                                   std::array<size_t, on> list_indices_out, // {3}
                                                   std::array<size_t, in> indices)   // {i,j,k}
{
    size_t kda = 0;
    size_t kdb = 0;

    for (auto i = 0; i < ia; i++)
        kda = (list_dim_a[i] >= size_a ? 1 : dim_a[list_dim_a[i]]) * (indices[list_indices_a[i]] + kda);
    for (auto i = 0; i < ib; i++)
        kdb = (list_dim_b[i] >= size_b ? 1 : dim_b[list_dim_b[i]]) * (indices[list_indices_b[i]] + kdb);

    size_t I = kda + kdb;

    size_t kda_r = 0;
    size_t kdb_r = 0;

    for (auto i = 0; i < on; i++) {
        size_t r = indices[list_indices_out[i]];
        kda_r = (list_dim_aout[i] >= size_a ? 1 : dim_a[list_dim_aout[i]]) * (r + kda_r);
        kdb_r = (list_dim_bout[i] >= size_b ? 1 : dim_b[list_dim_bout[i]]) * (r + kdb_r);
    }

    return std::make_tuple(I, kda + kda_r, kdb + kdb_r);
}

template <size_t NumberOfKeepDim>
constexpr size_t dimout(std::array<size_t, NumberOfKeepDim> keep_dim)
{
    size_t out = 1;
    for (size_t i : keep_dim)
        out *= i;

    return out;
}

//template< size_t ... A>
//static constexpr std::array< size_t, sizeof...(A)> toArray( std::index_sequence<A... >) noexcept {
//return std::array<size_t, sizeof...(A ) > { A... };
//}
//

template<typename tuple_t>
constexpr auto get_array_from_tuple(tuple_t&& tuple)
{
    constexpr auto get_array = [](auto&& ... x){ return std::array{std::forward<decltype(x)>(x) ... }; };
    return std::apply(get_array, std::forward<tuple_t>(tuple));
}

template <size_t, class T>
using T_ = T;

template <class T, size_t... Is>
auto gen(std::index_sequence<Is...>) { return std::tuple<T_<Is, T>...>{}; }

template <class T, size_t N>
auto gen() { return gen<T>(std::make_index_sequence<N>{}); }

struct KD_T {
    
};

// constexpr std::tuple<size_t,size_t,size_t> kd(std::array<size_t, size1> dim_a, std::array<size_t, size2> dim_b, size_t i, size_t j , size_t k)

int main()
{

    double FA[8] = {4.4, 5.4, 6.2, 6.5, 7.5, 6.1, 8.7, 1.3};
    double FB[4] = {1.4, 1.2, 1.5, 1.22};

    // generate tuple (at compile time)
    constexpr auto ma4 = build_array4(DimFa(),
                                      DimFb(),
                                      ContFa(),
                                      ContFb());

    // constexpr size_t sum_dim_a = 2;
    // constexpr size_t keep_dim_a[2] = [0,1];

    // constexpr size_t sum_dim_b[1] = 0;
    // constexpr size_t keep_dim_b[1] = [1];

    // print (at runtime)

    constexpr size_t dimout_var = dimout(std::get<4>(ma4));
    double out[dimout_var];
    std::fill(out, out + dimout_var, 0);

    const auto &my_lambda = [&out, &FA, &FB, &ma4](decltype(gen<size_t, std::get<5>(ma4).size()>()) it) {
        const auto &dim_a = std::get<0>(ma4);
        const auto &dim_b = std::get<1>(ma4);

        std::tuple KD = kd(dim_a, dim_b, std::get<0>(it), std::get<1>(it), std::get<2>(it));
        size_t I = std::get<0>(KD);
        size_t kda = std::get<1>(KD);
        size_t kdb = std::get<2>(KD);

        out[I] += FA[kda + std::get<3>(it)] * FB[kdb + dim_b[1] * std::get<3>(it)];
    };

    loop(std::get<5>(ma4), my_lambda);

    double out3[dimout_var];
    std::fill(out3, out3 + dimout_var, 0);

    constexpr const size_t indices_number = std::get<5>(ma4).size();
    const auto &my_lambda2 = [&out3, &FA, &FB, &ma4,indices_number](decltype(gen<size_t, indices_number>()) it) {
        const auto &dim_a = std::get<0>(ma4);
        const auto &dim_b = std::get<1>(ma4);


        std::tuple KD = kdvar<3,2,2,1,1,1,1,4>(dim_a,
                              dim_b,
                              {2, 1},
                              {2},
                              {3},
                              {1},
                              {0, 1},
                              {2},
                              {3},
                              get_array_from_tuple(it));
        size_t I = std::get<0>(KD);
        size_t kda = std::get<1>(KD);
        size_t kdb = std::get<2>(KD);
    
        out3[I] += FA[kda] * FB[kdb];
    };
    loop(std::get<5>(ma4), my_lambda2);

    // -------------------------------------------------------------------------------------------------------------------------------------
    double out2[2 * 2 * 2] = {0, 0, 0, 0, 0, 0, 0, 0};

    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 2; j++)
            for (size_t k = 0; k < 2; k++)
                for (size_t l = 0; l < 2; l++)
                {
                    out2[4 * i + 2 * j + k] += FA[4 * i + 2 * j + l] * FB[l * 2 + k];
                }

    for (auto i = 0; i < 8; i++)
    {
        std::cout << "out  = " << out[i] << std::endl;
    }

    for (auto i = 0; i < 8; i++)
    {
        std::cout << "out2 = " << out2[i] << std::endl;
    }

    for (auto i = 0; i < 8; i++)
    {
        std::cout << "out3 = " << out3[i] << std::endl;
    }
}
