#include <iostream>
#include <tuple>
#include <array>

#include <range/v3/all.hpp>
using namespace ranges;

constexpr std::size_t operator "" _z ( unsigned long long n )
    { return n; }

# define Ind(...) std::index_sequence<__VA_ARGS__>

// TensorDot(A,B,Ind(2,2,2), Ind(2,2), Ind(0,1), Ind(0,1))

using DimFa  = Ind(2,2,2);
using DimFb  = Ind(2,2);
using ContFa = Ind(2);
using ContFb = Ind(0);

using DimKeepA = Ind(2,2);
using DimKeepB = Ind(2);

template<size_t... Is, size_t... is>
static constexpr size_t get_columnmajor(std::index_sequence<is...>);

template<>
constexpr size_t get_columnmajor(std::index_sequence<>) {
    return 0_z;
}

template <size_t I, size_t... Is, size_t i, size_t... is>
static constexpr size_t get_columnmajor(std::index_sequence<i, is...>) {
    static_assert(i < I );
    return i + I * get_columnmajor<Is...>(std::index_sequence<is...>());
    return 0_z;
}


template< size_t ... DIMFA, size_t ... DIMFB, size_t ... CONTFA, size_t ... CONTFB >
static constexpr std::tuple<std::array< size_t, sizeof...(DIMFA)>,
                            std::array< size_t, sizeof...(DIMFB)>,
                            std::array< size_t, sizeof...(CONTFA)>,
                            std::array< size_t, sizeof...(CONTFB)>> build_array4( std::index_sequence<DIMFA... >,
                                                                                   std::index_sequence<DIMFB... >,
                                                                                   std::index_sequence<CONTFA...>,
                                                                                   std::index_sequence<CONTFB...> 
                                                                                 ) noexcept {
   return std::make_tuple(std::array<size_t, sizeof...(DIMFA ) > { DIMFA... },
                          std::array<size_t, sizeof...(DIMFB ) > { DIMFB... },
                          std::array<size_t, sizeof...(CONTFA) > { CONTFA... },
                          std::array<size_t, sizeof...(CONTFB) > { CONTFB... });
}

template <size_t shape_index, size_t shape_size>
struct Looper
{
    template <typename Functor>
    void operator()(const std::array<size_t, shape_size>& shape, Functor functor)
    {
        for (size_t index = 0; index < shape[shape_index]; ++index)
        {
            Looper<shape_index + 1, shape_size>()
                (
                    shape,
                    [index, &functor](auto... tail){ functor(index, tail...); }
                );
        }
    }
};

template <size_t shape_size>
struct Looper<shape_size, shape_size>
{
    template <typename Functor>
    void operator()(const std::array<size_t, shape_size>&, Functor functor)
    {
        functor();
    }
};

template <size_t shape_size, typename Functor>
void loop(const std::array<size_t, shape_size>& shape, Functor functor)
{
    Looper<0, shape_size>()(shape, functor);
}

constexpr unsigned cilog2(unsigned val) { return val ? 1 + cilog2(val >> 1) : -1; }

template <size_t size1, size_t size2>
constexpr std::tuple<size_t,size_t,size_t> kd(std::array<size_t, size1> dim_a, std::array<size_t, size2> dim_b, size_t i, size_t j , size_t k) {
    size_t kda = dim_a[1]*dim_a[2]*i + dim_a[2]*j;
    size_t kdb = k;
    size_t I   = kda + kdb;
    return std::make_tuple(I,kda,kdb);
}

int main() {

    // generate tuple (at compile time)
    constexpr auto ma4 = build_array4( DimFa(),
                                                                     DimFb(),
                                                                     ContFa(),
                                                                     ContFb() );
     
     
    double FA[8] = {4.4,5.4,6.2,6.5,7.5,6.1,8.7,1.3};
    double FB[4] = {1.4,1.2,1.5,1.22};
 
    // constexpr size_t sum_dim_a = 2;
    // constexpr size_t keep_dim_a[2] = [0,1];

    // constexpr size_t sum_dim_b[1] = 0;
    // constexpr size_t keep_dim_b[1] = [1];

    // print (at runtime)
    double out[2*2*2] = {0,0,0,0,0,0,0,0};

    size_t I = 0;
    loop(std::array<size_t,4>{2,2,2,2},[&I,&out,&FA,&FB,&ma4](size_t i,size_t j,size_t k, size_t r){
        const auto& dim_a = std::get<0>(ma4);
        const auto& dim_b = std::get<1>(ma4);

        std::tuple KD = kd(dim_a,dim_b,i,j,k);
        size_t I   = std::get<0>(KD);
        size_t kda = std::get<1>(KD);
        size_t kdb = std::get<2>(KD);
        out[I] += FA[kda + r] * FB[dim_b[1]*r + kdb];
    });
 
//    std::cout << "out = " << (out | view::all) << std::endl;

    // constexpr size_t base2 = cilog2(8); 
    // std::array<size_t, cilog2(8)> ijk;

    // loop(std::array<size_t,2>{8,2},[&out,&FA,&FB,&ijk](size_t I, size_t r){
    //     size_t i = I / 8;
    //     size_t j = (I-i) / 4;
    //     size_t k = (I-j-k) / 2;
    
    //     out[I] += FA[ 4*i + 2*j + r] * FB[ r*2 + k];
    // });
    double out2[2*2*2] = {0,0,0,0,0,0,0,0};

    for(size_t i=0; i<2; i++)
        for(size_t j=0; j<2; j++)
         for(size_t k=0; k<2; k++)
            for(size_t l=0; l< 2; l++) {
            out2[4*i + 2*j + k] += FA[ 4*i + 2*j + l] * FB[ l*2 + k];
        }



   std::cout << "out  = " << (out | view::all) << std::endl;
   std::cout << "out2 = " << (out2 | view::all) << std::endl;
}
