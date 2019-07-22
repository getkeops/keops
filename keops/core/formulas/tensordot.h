#pragma once

#include <tuple>
#include <array>

namespace keops {

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


}