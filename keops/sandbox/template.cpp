#include <iostream>
#include <tuple>
#include <array>

# define Ind(...) std::index_sequence<__VA_ARGS__>

// TensorDot(A,B,Ind(2,2,2), Ind(2,2), Ind(0,1), Ind(0,1))

using DimFa  = Ind(2,2,2);
using DimFb  = Ind(2,2);
using ContFa = Ind(0,1);
using ContFb = Ind(0,1);

// using Is1 =  std::index_sequence<2,3,3,3> ;
// using Is2 =  std::index_sequence<2,3,7>;


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
                          std::array<size_t, sizeof...(CONTFB) > { CONTFB... },);
}


int main() {

    // generate tuple (at compile time)
    constexpr std::tuple<std::array<size_t, Is1().size()>,
               std::array<size_t, Is2().size()>,
               std::array<size_t, Is3().size()>,
               std::array<size_t, Is4().size()>> ma4 = build_array4( DimFa(),
                                                                     DimFb(),
                                                                     ContFa(),
                                                                     ContFb() );
     
     
    double FA[8] = {4.4,5.4,6.2,6.5,7.5,6.1,8.7,1.3};
    double FB[4] = {1.4,1.2,1.5,1.22};
    
    double out = 0;
    constexpr size_t dim1 = std::get<0>(std::get<2>(ma2)[0]);
    constexpr size_t dim2 = std::get<1>(std::get<3>(ma2)[1]);
    
    // print (at runtime)
    for(size_t k=0; k<dim1; k++)
        for(size_t l=0; l< dim2; l++) {
            out += FA[ l * dim1*dim1 + k * dim2] * FB[ l*dim1 + k];
            std::cout << "(" << k << "," << l << ")" << std::endl;
        }

   std::cout << "out = " << out << std::endl;
}

