#include <iostream>
#include <tuple>
#include <array>

# define Ind(...) std::index_sequence<__VA_ARGS__>

// Syntaxe : TensorDot(A,B,Ind(2,2,2), Ind(2,2), Ind(0,1), Ind(0,1))

// tensorDot 1

using DimFa  = Ind(2,2,2);
using DimFb  = Ind(2,2);
using ContFa = Ind(0,1);
using ContFb = Ind(0,1);

// using Is1 =  std::index_sequence<2,3,3,3> ;
// using Is2 =  std::index_sequence<2,3,7>;

using MA_TYPE = std::tuple<std::array<size_t,  DimF ContFa().size()>,
                           std::array<size_t,  DimFb().size()>,
                           std::array<size_t,  ContFa().size()>,
                           std::array<size_t,  ContFb().size()>>;


template< size_t ... DIMFA, size_t ... DIMFB, size_t ... CONTFA, size_t ... CONTFB >
static constexpr std::tuple<std::array< size_t, sizeof...(DIMFA)>,
                            std::array< size_t, sizeof...(DIMFB)>,
                            std::array< size_t, sizeof...(CONTFA)>,
                            std::array< size_t, sizeof...(CONTFB)>> build_array4( std::index_sequence<DIMFA... >,
                                                                                  std::index_sequence<DIMFB... >,
                                                                                  std::index_sequence<CONTFA...>,
                                                                                  std::index_sequence<CONTFB...> 
                                                                                 ) noexcept {
   return std::make_tuple(std::array<size_t, sizeof...(DIMFA ) > { DIMFA...  },
                          std::array<size_t, sizeof...(DIMFB ) > { DIMFB...  },
                          std::array<size_t, sizeof...(CONTFA) > { CONTFA... },
                          std::array<size_t, sizeof...(CONTFB) > { CONTFB... } );
}


struct TensorDot{
  static constexpr  std::tuple<std::array<size_t, DimFa().size()>,
               std::array<size_t, DimFb().size() >,
               std::array<size_t, ContFa().size()>,
               std::array<size_t, ContFb().size()>> ma4 = build_array4( DimFa(),
                                                                        DimFb(),
                                                                        ContFa(),
                                                                        ContFb() );
  
  template <size_t>
  double Operation(double*, double*) const;                                                                                                      
};

// Specialization
template<>
double TensorDot::Operation<1>(double* FA, double* FB) const {
    double out = 0;
    
    constexpr auto dimfa  = std::get<0>(ma4);
    constexpr auto contfa = std::get<2>(ma4);
    constexpr size_t dim1 = dimfa[contfa[0]];
    
    constexpr auto dimfb  = std::get<1>(ma4);
    constexpr auto contfb = std::get<3>(ma4);
    constexpr size_t dim2 = dimfb[contfb[1]];
    
    // print (at runtime)
    for(size_t k=0; k<dim1; k++)
        out += FA[ l * dim1 k ] * FB[ l*dim1 + k];
  return out;
}

template<>
double TensorDot::Operation<2>(double* FA, double* FB) const {
    double out = 0;
 
    constexpr auto dimfa  = std::get<0>(ma4);
    constexpr auto contfa = std::get<2>(ma4);
    constexpr size_t dim1 = dimfa[contfa[0]];
    
    constexpr auto dimfb  = std::get<1>(ma4);
    constexpr auto contfb = std::get<3>(ma4);
    constexpr size_t dim2 = dimfb[contfb[1]];
    
    // print (at runtime)
    for(size_t k=0; k<dim1; k++)
        for(size_t l=0; l< dim2; l++) {
            out += FA[ l * dim1*dim1 + k * dim2] * FB[ l*dim1 + k];
        }
  
  return out;
}



inline void inline_test(double* FA, double* FB) {     

    
    constexpr TensorDot td;
  
    std::cout <<  td.Operation<ContFb().size()>(FA, FB);
    
    
}


int main(){

    double FA[8] = {4.4,5.4,6.2,6.5,7.5,6.1,8.7,1.3};
    double FB[4] = {1.4,1.2,1.5,1.22};
 inline_test(FA, FB);
}

/*
a = np.arange(8).reshape(2,2,2)
b = 10 + np.arange(4).reshape(2,2)

af = np.arange(8)
bf = 10 + np.arange(4)

c = np.tensordot(a,b,1)

cf = np.zeros(8)

for i in range(2):
   for j in range(2):
      for k in range(2):
         for r in range(2):
          cf[2*2*i+2*j+k] += af[2*2*i+2*j + r] * bf[2*r+k]
      
      
cf = cf.reshape(2,2,2)
np.allclose(cf,c)



##############################################################

d = tuple(1+np.random.randint(10,size=3))
dd = (d[2],) + tuple(1+ np.random.randint(10,size=1))

a = np.arange(np.prod(d)).reshape(*d)
b = 10 + np.arange(np.prod(dd)).reshape(*dd)

af = np.arange(np.prod(d))
bf = 10 + np.arange(np.prod(dd))

c = np.tensordot(a,b,1)

cf = np.zeros(np.prod(d[:2])* dd[1])

dim = d[:2] + (dd[1],)

for i in range(dim[0]):
   for j in range(dim[1]):
      for k in range(dim[2]):
         for r in range(dd[0]):
          cf[dim[1]*dim[2]*i+dim[2]*j+k] += af[d[1]*d[2]*i+d[2]*j + r] * bf[dd[1]*r+k]
      
      
cf = cf.reshape(*dim)
np.allclose(cf,c)

#############################################################


dim_a = tuple(1+np.random.randint(10,size=3))
dim_b = (dim_a[2],) + tuple(1+ np.random.randint(10,size=1))

a = np.arange(np.prod(dim_a)).reshape(*dim_a)
b = 10 + np.arange(np.prod(dim_b)).reshape(*dim_b)

af = np.arange(np.prod(dim_a))
bf = 10 + np.arange(np.prod(dim_b))

c = np.tensordot(a,b,1) 

assert(np.allclose(np.tensordot(a,b,axes=[[-1],[0]]),c))

dim_sum_a = dim_a[-1:]
dim_sum_b = dim_b[:1]
assert(dim_sum_a == dim_sum_b)

dim_keep_a = tuple(dim_a[:-1]) 
dim_keep_b = tuple(dim_b[1:])
dim_keep = dim_keep_a + dim_keep_b

cf = np.zeros(np.prod(dim_keep))

for i in range(dim_keep[0]):
   for j in range(dim_keep[1]):
      for k in range(dim_keep[2]):
         for r in range(dim_sum_a[0]):
           cf[dim_keep[1]*dim_keep[2]*i+dim_keep[2]*j+k] += af[dim_a[1]*dim_a[2]*i+dim_a[2]*j + r] * bf[dim_b[1]*r+k]
      
      
cf = cf.reshape(*dim_keep)
np.allclose(cf,c)




############################################################

d = tuple(1+np.random.randint(10,size=3))
dd =  tuple(1+ np.random.randint(10,size=1)) + (d[1],)

a = np.arange(np.prod(d)).reshape(*d)
b = 10 + np.arange(np.prod(dd)).reshape(*dd)

af = np.arange(np.prod(d))
bf = 10 + np.arange(np.prod(dd))

c = np.tensordot(a,b,axes=[[1],[1]])


dim = (d[0], d[2]) + (dd[0],)
cf = np.zeros(np.prod(dim))

for i in range(dim[0]):
   for j in range(dim[1]):
      for k in range(dim[2]):
         for r in range(dd[1]):
          cf[dim[1]*dim[2]*i+dim[2]*j+k] += af[d[1]*d[2]*i+d[2]*r + j] * bf[dd[1]*k+r]
      
      
cf = cf.reshape(*dim)
np.allclose(cf,c)


#################################################################


*/

