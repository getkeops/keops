
// g++ -I.. -D__TYPE__=float -std=c++17 -O3 -o build/test_tensordot test_tensordot.cpp


#include <iostream>
#include <tuple>
#include <array>


#define Ind(...) std::index_sequence<__VA_ARGS__>



//----------------------------------------------------------------------------------------------------------------------

// Standard loop
template<size_t shape_index, size_t shape_size>
struct Looper {
  template<typename Functor>
  void operator()(const std::array<size_t, shape_size> &shape, Functor functor) {
    for (size_t index = 0; index < shape[shape_index]; ++index) {
      Looper<shape_index + 1, shape_size>()(
          shape,
          [index, &functor](auto... tail) { functor(std::tuple_cat(std::tuple<size_t>(index), tail...)); });
    }
  }
};

// Stopping condition
template<size_t shape_size>
struct Looper<shape_size, shape_size> {
  template<typename Functor>
  void operator()(const std::array<size_t, shape_size> &, Functor functor) {
    functor();
  }
};

template<size_t shape_size, typename Functor>
void loop(const std::array<size_t, shape_size> &shape, Functor functor) {
  Looper<0, shape_size>()(shape, functor);
}

// ---------------------------------------------------------------------------------------------------------------------

template<size_t SIZE_DIMFA, size_t SIZE_DIMFB, size_t SIZE_CONTA >
using packed_tensor_parameters = std::tuple<std::array<size_t, SIZE_DIMFA>,
                            std::array<size_t, SIZE_DIMFB>,
                            std::array<size_t, SIZE_DIMFA>,
                            std::array<size_t, SIZE_DIMFB>,
                            std::array<size_t, SIZE_DIMFA + SIZE_DIMFB - 2 * SIZE_CONTA>,
                            std::array<size_t, SIZE_DIMFA + SIZE_DIMFB - SIZE_CONTA>,
                            size_t>;

template<size_t... DIMFA, size_t... DIMFB, size_t... CONTFA, size_t... CONTFB>
static constexpr packed_tensor_parameters<sizeof... (DIMFA), sizeof... (DIMFB), sizeof... (CONTFA)>
tensordot_parameters(Ind(DIMFA...), Ind(DIMFB...), Ind(CONTFA...), Ind(CONTFB...)) noexcept {

  // Cast index_sequence to array
  constexpr auto list_dim_a = std::array<size_t, sizeof...(DIMFA)>{DIMFA...};
  constexpr auto list_dim_b = std::array<size_t, sizeof...(DIMFB)>{DIMFB...};
  constexpr auto indices_contdim_a = std::array<size_t, sizeof...(CONTFA)>{CONTFA...};
  constexpr auto indices_contdim_b = std::array<size_t, sizeof...(CONTFB)>{CONTFB...};

  constexpr size_t size_listdim_a = list_dim_a.size();
  constexpr size_t size_listdim_b = list_dim_b.size();
  // get size of the contraction
  constexpr size_t size_keepdim_a = list_dim_a.size() - indices_contdim_a.size();
  constexpr size_t size_keepdim_b = list_dim_b.size() - indices_contdim_b.size();
  size_t dimout = 1;




  // dim_keep : contains the list of kept dim
  std::array<size_t, size_keepdim_b + size_keepdim_a> dim_keep{};
  std::array<size_t, size_keepdim_a> list_keep_dim_a{};
  std::array<size_t, size_keepdim_b> list_keep_dim_b{};
  for (size_t i = 0; i < size_keepdim_a; i++) {
    list_keep_dim_a[i] = list_dim_a[i];
    dim_keep[i] = list_dim_a[i];
    dimout *= list_dim_a[i];
  }

  for (size_t i = 0; i < size_keepdim_b; i++) {
    list_keep_dim_b[i] = list_dim_b[indices_contdim_b.size() + i];
    dim_keep[i + size_keepdim_a] = list_keep_dim_b[i];
    dimout *= list_keep_dim_b[i];
  }

  // contdim
  std::array<size_t, indices_contdim_a.size()> dim_cont{};
  for (size_t i = 0; i < indices_contdim_a.size(); i++) {
    dim_cont[i] = list_dim_a[size_keepdim_a + i];
  }

  // dim_tot : contains all indices in that order [list_keep_dim_a, list_keep_dim_b, dim_cont]
  std::array<size_t, indices_contdim_a.size() + dim_keep.size()> dim_tot{};
  for (size_t i = 0; i < dim_keep.size(); i++) {
    dim_tot[i] = dim_keep[i];
  }
  for (size_t i = 0; i < indices_contdim_a.size(); i++) {
    dim_tot[dim_keep.size() + i] = dim_cont[i];
  }

  std::array<size_t, size_listdim_a> list_stride_dim_a{};
  for (size_t i = 0; i < size_listdim_a; i++) {
    list_stride_dim_a[i] = 1;
    for (size_t j=i+1; j < size_listdim_a; j++)
      list_stride_dim_a[i] *= list_dim_a[j];
  }
  std::array<size_t, size_listdim_b> list_stride_dim_b{};
  for (size_t i = 0; i < size_listdim_b; i++) {
    list_stride_dim_b[i] = 1;
    for (size_t j=i+1; j < size_listdim_b; j++ )
      list_stride_dim_b[i] *= list_dim_b[j];
  }

  std::array<size_t, size_keepdim_a + size_keepdim_b> list_stride_keepdim{};
  for (size_t i = 0; i < size_keepdim_a  + size_keepdim_b; i++) {
    list_stride_keepdim[i] = 1;
    for (size_t j = i + 1; j < size_keepdim_a + size_keepdim_b; j++)
      list_stride_keepdim[i] *= (j < size_keepdim_a) ? list_dim_a[j] : list_dim_b[j - size_keepdim_a
          + indices_contdim_a.size()];                                                       // {0,1}
  }

    return std::make_tuple(list_dim_a, list_dim_b,
                           list_stride_dim_a, list_stride_dim_b, list_stride_keepdim,
                           dim_tot, dimout);
}


template<size_t SIZE_DIMFA, size_t SIZE_DIMFB, size_t SIZE_CONT>
constexpr std::tuple<size_t, size_t, size_t> kdvar(const packed_tensor_parameters<SIZE_DIMFA, SIZE_DIMFB, SIZE_CONT> &ma4,
                                                   std::array<size_t, SIZE_DIMFA + SIZE_DIMFB - SIZE_CONT> list_indices_tot)
{

  const auto list_dim_a = std::get<0>(ma4);
  const auto list_dim_b = std::get<1>(ma4);
  const auto list_stride_dim_a = std::get<2>(ma4);
  const auto list_stride_dim_b = std::get<3>(ma4);
  const auto list_stride_keepdim = std::get<4>(ma4);

  constexpr size_t size_keep_dim_a = list_indices_tot.size() - list_dim_b.size();
  constexpr size_t size_keep_dim_b = list_indices_tot.size() - list_dim_a.size();
  constexpr size_t size_list_contdim = list_indices_tot.size() - size_keep_dim_a - size_keep_dim_b ;
  constexpr size_t size_list_dim_b = list_dim_b.size();



  // kda and kdb --------------------------

  size_t kda = 0;
  for (size_t i = 0; i < (size_keep_dim_a); i++) {
    size_t list_indices_keepdim_ai = list_indices_tot[i];
    kda += list_stride_dim_a[i] *  list_indices_keepdim_ai;
  }

  size_t kdb = 0;
  for (size_t i = 0; i < (size_keep_dim_b); i++) {
    size_t list_indices_keepdim_bi = list_indices_tot[size_keep_dim_a + i];
    kdb += list_stride_dim_b[size_list_dim_b-1 - i] * list_indices_keepdim_bi;
  }


  for (size_t i = 0; i < (size_list_contdim); i++) {
    size_t list_indices_contdimi = list_indices_tot[(size_keep_dim_a + size_keep_dim_b) + i];
    kda += list_stride_dim_a[size_keep_dim_a + i] * list_indices_contdimi;
    kdb += list_stride_dim_b[i] * list_indices_contdimi;
  }

  // ------------------

  size_t I = 0;
  for (size_t i = 0; i < size_keep_dim_a + size_keep_dim_b  ; i++) {
    I += list_stride_keepdim[i] * list_indices_tot[i];
  }

  // std::cout << "(" << list_indices_tot[0] << "," << list_indices_tot[1] <<  "," <<list_indices_tot[2] <<  "," <<list_indices_tot[3] <<")     " << kda << " " << kdb << " " << I << std::endl;

  return std::make_tuple(I, kda, kdb);
}


template<size_t KEEPDIM>
constexpr size_t dimout(std::array<size_t, KEEPDIM> keep_dim) {
  size_t out = 1;
  for (size_t i : keep_dim)
    out *= i;

  return out;
}

// ------------------- Utils
template<size_t, class T>
using T_ = T;

template<class T, size_t... Is>
auto gen(std::index_sequence<Is...>) { return std::tuple<T_<Is, T>...>{}; }

template<class T, size_t N>
auto gen() { return gen<T>(std::make_index_sequence<N>{}); }

template<typename tuple_t>
constexpr auto get_array_from_tuple(tuple_t &&tuple) {
  constexpr auto get_array = [](auto &&... x) { return std::array{std::forward<decltype(x)>(x) ...}; };
  return std::apply(get_array, std::forward<tuple_t>(tuple));
}




//----------------------------------------------------------------------------------------------------------------------
using DimFa = Ind(2, 2, 2);
using DimFb = Ind(2, 2);

using ContFa = Ind(2);
using ContFb = Ind(0);
using ContFa4 = Ind(1,2);
using ContFb4 = Ind(0,1);


using DimFa6 = Ind(5, 4, 3);
using DimFb6 = Ind(4, 3, 2);

using ContFa6 = Ind(1,2);
using ContFb6 = Ind(0,1);

using ContFa8 = Ind(1,2,3);



int main() {

  double FA[8] = {4.4, 5.4, 6.2, 6.5, 7.5, 6.1, 8.7, 1.3};
  double FB[4] = {1.4, 1.2, 1.5, 1.22};

  // -------------------------------------------------------------------------------------------------------------------
  // generate tuple (at compile time)
  constexpr auto ma4 = tensordot_parameters(DimFa(),
                                    DimFb(),
                                    ContFa(),
                                    ContFb());

  
  double out_td[std::get<6>(ma4)];
  std::fill(out_td, out_td + std::get<6>(ma4), 0);

  constexpr const size_t indices_number = std::get<5>(ma4).size();

  const auto &my_lambda2 = [&out_td, &FA, &FB, &ma4](decltype(gen<size_t, indices_number>()) it) {

    std::tuple KD = kdvar<DimFa::size(), DimFb::size(), ContFa::size()>(ma4, get_array_from_tuple(it));

    size_t I = std::get<0>(KD);
    size_t kda = std::get<1>(KD);
    size_t kdb = std::get<2>(KD);

    out_td[I] += FA[kda] * FB[kdb];
  };

  loop(std::get<5>(ma4), my_lambda2);

  // --------

  double out_loop[2 * 2 * 2] = {0, 0, 0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < 2; i++)
    for (size_t j = 0; j < 2; j++)
      for (size_t k = 0; k < 2; k++)
        for (size_t l = 0; l < 2; l++) {
          // size_t kda = 4 * i + 2 * j + l;
          // size_t kdb = l * 2 + k;
          // size_t I = 4 * i + 2 * j + k;
          // std::cout << "(" << i << "," << j <<  "," << k <<  "," << l <<")     " << kda << " " << kdb << " " << I << std::endl;
          out_loop[4 * i + 2 * j + k] += FA[4 * i + 2 * j + l] * FB[l * 2 + k];
        }


  double res = 0;
  for (auto i = 0; i < 8; i++) {
    // std::cout << "out_tensordot = " << out_td[i] << std::endl;
    // std::cout << "out_loop = " << out_loop[i] << std::endl;
    res += (out_td[i] - out_loop[i]) * (out_td[i] - out_loop[i]);
  }

  std::cout << "error = " << res << std::endl<< std::endl;


  // -------------------------------------------------------------------------------------------------------------------

  constexpr auto ma5 = tensordot_parameters(DimFa(),
                                    DimFb(),
                                    ContFa4(),
                                    ContFb4());

  double out5[std::get<6>(ma5)];
  std::fill(out5, out5 + std::get<6>(ma5), 0);
  
  constexpr const size_t indices_number4 = std::get<5>(ma5).size();
  const auto &my_lambda4 = [&out5, &FA, &FB, &ma5](decltype(gen<size_t, indices_number4>()) it) {

    std::tuple KD = kdvar<DimFa::size(), DimFb::size(), ContFa4::size()>(ma5, get_array_from_tuple(it));

    size_t I = std::get<0>(KD);
    size_t kda = std::get<1>(KD);
    size_t kdb = std::get<2>(KD);

    out5[I] += FA[kda] * FB[kdb];
  };

  loop(std::get<5>(ma5), my_lambda4);

  // -------------

  double out4[2] = {0, 0};

  for (size_t i = 0; i < 2; i++)
    for (size_t k = 0; k < 2; k++)
      for (size_t l = 0; l < 2; l++) {

        out4[ i ] += FA[4 * i + 2 * k + l] * FB[k * 2 + l];
      }

  double res2 = 0;
  for (auto i = 0; i < 2; i++) {
    // std::cout << "out4 = " << out4[i] << std::endl;
    // std::cout << "out5 = " << out5[i] << std::endl;
    res2 += (out4[i] - out5[i]) * (out4[i] - out5[i]);
  }

  std::cout << "error = " << res2 << std::endl<< std::endl;


  // -------------------------------------------------------------------------------------------------------------------

  double FAA[60] =  {7, 9, 9, 5, 8, 3, 6, 9, 6, 0, 5, 7, 3, 4, 3, 5, 3, 3,0, 9, 9,6, 0, 3,3, 7, 0,8, 6, 0,6, 1, 3,1, 4, 7,3, 9, 8,8, 3, 7,2, 3, 1,9, 5, 7,7, 5, 9,7, 0, 1,9, 7, 5,0, 3, 8};
  double FBB[24] =  {6, 4,2, 9,9, 5,1, 6,7, 8,2, 4,1, 9,7, 8,5, 4,3, 2,3, 8,5, 7};


  constexpr auto ma6 = tensordot_parameters(DimFa6(),
                                    DimFb6(),
                                    ContFa6(),
                                    ContFb6());


  double out6[std::get<6>(ma6)];
  std::fill(out6, out6 + std::get<6>(ma6), 0);

  constexpr const size_t indices_number6 = std::get<5>(ma6).size();
  const auto &my_lambda6 = [&out6, &FAA, &FBB, &ma6](decltype(gen<size_t, indices_number6>()) it) {

    std::tuple KD = kdvar<DimFa6::size(), DimFb6::size(), ContFa6::size()>(ma6, get_array_from_tuple(it));

    size_t I = std::get<0>(KD);
    size_t kda = std::get<1>(KD);
    size_t kdb = std::get<2>(KD);

    out6[I] += FAA[kda] * FBB[kdb];
  };

  loop(std::get<5>(ma6), my_lambda6);

  // -------------

  double out7[10] =  {0, 0, 0, 0 ,0, 0, 0, 0, 0 ,0};

  for (size_t i = 0; i < 5; i++)
    for (size_t j = 0; j < 2; j++)
      for (size_t k = 0; k < 4; k++)
        for (size_t l = 0; l < 3; l++) {
          // size_t kda = 12 * i + 3 * k + l;
          // size_t kdb = 6 * k + 2 * l + j;
          // size_t I = 2 * i + j;
          // std::cout << "(" << i << "," << j <<  "," << k <<  "," << l <<")     " << kda << " " << kdb << " " << I << std::endl;
          out7[2 * i + j] += FAA[ 12 * i + 3 * k + l] * FBB[6 * k + 2 * l + j];
        }

  double res3=0;
  for (auto i = 0; i < 10; i++) {
    // std::cout << "out7 = " << out7[i] << std::endl;
    // std::cout << "out6 = " << out6[i] << std::endl;
    res3 += (out6[i] - out7[i]) * (out6[i] - out7[i]);
  }

  std::cout << "error = " << res3 << std::endl<< std::endl;

// -------------------------------------------------------------------------------------------------------------------

constexpr auto ma8 = tensordot_parameters(DimFa6(),
                                  DimFa6(),
                                  ContFa8(),
                                  ContFa8());

double out8[std::get<6>(ma8)];
std::fill(out8, out8 + std::get<6>(ma8), 0);

constexpr const size_t indices_number8 = std::get<5>(ma8).size();
const auto &my_lambda8 = [&out8, &FAA, &ma8](decltype(gen<size_t, indices_number8>()) it) {

  std::tuple KD = kdvar<DimFa6::size(), DimFb6::size(), ContFa8::size()>(ma8, get_array_from_tuple(it));

  size_t I = std::get<0>(KD);
  size_t kda = std::get<1>(KD);
  size_t kdb = std::get<2>(KD);

  out8[I] += FAA[kda] * FAA[kdb];
};

loop(std::get<5>(ma8), my_lambda8);


// -------------

double out9[1] = {0};

for (size_t i = 0; i < 5; i++)
  for (size_t j = 0; j < 4; j++)
    for (size_t k = 0; k < 3; k++) {
        // size_t kda = 12 * i + 3 * j + k;
        // size_t kdb = 12 * i + 3 * j + k;
        // size_t I = 0;
        // std::cout << "(" << i << "," << j <<  "," << k <<  "," << l <<")     " << kda << " " << kdb << " " << I << std::endl;
        out9[0] += FAA[ 12 * i + 3 * j + k] * FAA[12 * i + 3 * j + k];
      }

double res4=0;
for (auto i = 0; i < 1; i++) {
  // std::cout << "out7 = " << out7[i] << std::endl;
  // std::cout << "out6 = " << out6[i] << std::endl;
  // std::cout << "out8 = " << out8[i] << std::endl;
  res4 += (out8[i] - out9[i]) * (out8[i] - out9[i]);
}

std::cout << "error = " << res4 << std::endl<< std::endl;

}


