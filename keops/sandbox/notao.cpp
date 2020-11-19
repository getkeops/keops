#include <utility>
#include <array>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <functional>


template <class T, std::size_t N>
std::ostream& operator<<(std::ostream& o, const std::array<T, N>& arr)
{
    std::copy(arr.cbegin(), arr.cend(), std::ostream_iterator<T>(o, " "));
    return o;
}

template <auto& Arr, size_t... Is>
constexpr auto make_seq_impl(std::index_sequence<Is...>) {
    using T = typename std::decay_t<decltype(Arr)>::value_type;
    return std::integer_sequence<T, Arr[Is]...>{};
}

template <auto& Arr>
constexpr auto make_seq() {
    return make_seq_impl<Arr>(std::make_index_sequence<Arr.size()>());
}

template<std::size_t... I>
constexpr auto make_array(std::index_sequence<I...>) {
    return std::array<size_t, sizeof...(I)>{ {I...} };
}
template<std::size_t N>
constexpr auto make_index_array() {
    return make_array(std::make_index_sequence<N>{});
}

template<std::size_t N>
constexpr size_t prod_array(const std::array<size_t, N> arr) {
    size_t res = 1;
    for (size_t i=0; i < N; i++){
        res *= arr[i];
    }
    return res;
}

template<std::size_t N>
constexpr auto cumprod_array(const std::array<size_t, N> arr) {
    std::array<size_t, N> res;
    res[N-1] = 1;
    for (int i= N-2; i > -1 ; i--){
        res[i] = res[i+1] * arr[i+1];
    }
    return res;
}

template<std::size_t N>
constexpr auto reverse_array(const std::array<size_t, N> arr) {
    std::array<size_t, N> res;
    std::reverse_copy(arr.begin(), arr.end(), res.begin());
    return res;
}

template<std::size_t N, size_t M>
constexpr auto concat_array(const std::array<size_t, N> arr0, const std::array<size_t, M> arr1) {
    std::array<size_t, N+M> res{};
    //std::copy(arr0.begin(), arr0.end(), res.begin());
    //std::copy(arr1.begin(), arr1.end(), res.begin()+N);
    for (size_t i=0; i<N; i++){
        res[i] = arr0[i];
    }
    for (size_t i=0; i<M; i++){
        res[i+N] = arr1[i];
    }
    return res;
}

template<std::size_t N, size_t M>
constexpr auto map_array(const std::array<size_t, N> ind, const std::array<size_t, M> arr1) {
    std::array<size_t, N> res{};
    for (size_t i= 0; i < N; i++){
        res[i] = arr1[ind[i]];
    }
    return res;
}

template<std::size_t N>
constexpr auto permutate_array(const std::array<size_t, N> perm, const std::array<size_t, N> arr) {
    std::array<size_t, N> res{};
    for (size_t i= 0; i < N; i++){
        res[perm[i]] = arr[i];
    }
    return res;
}

template<std::size_t N>
constexpr size_t sum_multiplies_array(const std::array<size_t, N> arr0, const std::array<size_t, N>  arr1) {
    size_t res = 0;
    for (size_t i= 0; i < N; i++){
        res += arr0[i] * arr1[i];
    }
    return res;
}


template< std::size_t N, std::size_t M >
constexpr auto diff_array(std::array< size_t, M > ind) {
/*
    auto res = create_index_array< N - M >();
    size_t l = 0;
    for (size_t i= 0; i < N; i++){
        bool include = false;
        for (size_t j=0; j < M; j++) {
            if (ind[j] == i)
                include = true;
        }
        if (!include) {
            res[l] = i;
            l++;
        }

    }
    return res;
    */
    auto res = make_index_array< N - M >();
    auto tmp =  make_index_array< N >();
    std::sort(ind.begin(), ind.end());
    std::set_difference(tmp.begin(), tmp.end(), ind.begin(), ind.end(), res.begin());
    return res;
}

template<size_t B, size_t E>
constexpr auto make_range_array() {

    static_assert(E > B, "please provide E > B");
    std::array<size_t, E - B> res{};
    for (size_t i= B; i < E; i++){
        res[i - B] = i;
    }
    return res;
    /*
    std::array<size_t, E - B> res;
    std::iota(res.begin(), res.end(), B);
    return res
     */
}

namespace cstd {

    template <typename RAIt>
    constexpr RAIt next(RAIt it, typename std::iterator_traits<RAIt>::difference_type n = 1) {
        return it + n;
    }

    template <typename RAIt>
    constexpr auto distance(RAIt first, RAIt last) {
        return last - first;
    }

    template<class ForwardIt1, class ForwardIt2>
    constexpr void iter_swap(ForwardIt1 a, ForwardIt2 b) {
        auto temp = std::move(*a);
        *a = std::move(*b);
        *b = std::move(temp);
    }

    template<class InputIt, class UnaryPredicate>
    constexpr InputIt find_if_not(InputIt first, InputIt last, UnaryPredicate q) {
        for (; first != last; ++first) {
            if (!q(*first)) {
                return first;
            }
        }
        return last;
    }

    template<class ForwardIt, class UnaryPredicate>
    constexpr ForwardIt partition(ForwardIt first, ForwardIt last, UnaryPredicate p) {
        first = cstd::find_if_not(first, last, p);
        if (first == last) return first;

        for(ForwardIt i = cstd::next(first); i != last; ++i){
            if(p(*i)){
                cstd::iter_swap(i, first);
                ++first;
            }
        }
        return first;
    }

}

template<class RAIt, class Compare = std::less<>>
constexpr void quick_sort(RAIt first, RAIt last, Compare cmp = Compare{}) {
    auto const N = cstd::distance(first, last);
    if (N <= 1) return;
    auto const pivot = *cstd::next(first, N / 2);
    auto const middle1 = cstd::partition(first, last, [=](auto const& elem) { return cmp(elem, pivot); });
    auto const middle2 = cstd::partition(middle1, last, [=](auto const& elem){ return !cmp(pivot, elem); });
    quick_sort(first, middle1, cmp); // assert(std::is_sorted(first, middle1, cmp));
    quick_sort(middle2, last, cmp);  // assert(std::is_sorted(middle2, last, cmp));
}

template <size_t N>
constexpr auto sort_indexes_array2(const std::array< size_t, N > arr) {

    // initialize original index locations
    auto idx = make_index_array<N>();

    // sort indexes based on comparing values in v
    // using std::stable_sort instead of std::sort
    // to avoid unnecessary index re-orderings
    // when v contains elements of equal values
    //std::stable_sort(idx.begin(), idx.end(),
    //            [&arr](size_t i1, size_t i2) {return arr[i1] < arr[i2];});

    quick_sort(idx.begin(), idx.end(), [&arr](size_t i1, size_t i2) {return arr[i1] < arr[i2];});

    return idx;
}


template <size_t N>
constexpr auto sort_indexes_array(std::array< size_t, N > arr) {

    // initialize original index locations
    std::array<size_t, N> res{};
    const size_t max_elem = *std::max_element(arr.begin(), arr.end());

    if (N>1){
        for (int i=0; i < static_cast<int>(N); i++) {
            auto tmp =  std::distance(arr.begin(),std::min_element(arr.begin(), arr.end()));
            res[tmp] = i;
            arr[tmp] += max_elem +1;
        }
    }
    return res;
}

constexpr std::array<size_t, 3> a{7, 15, 28};
constexpr std::array<size_t, 3> b{4, 0, 2};
constexpr std::index_sequence<7,3,5,6> i_seq;
constexpr auto array0 = make_array(i_seq);
constexpr auto array1 = make_index_array<4>();
constexpr std::array<size_t, 4> array2{2, 0, 1, 3};
constexpr std::array<size_t, 4> array3{0, 3, 2, 1};
constexpr std::array<size_t, 0> arrayEmpy{};
constexpr auto array10 = make_range_array<0, 4>();

constexpr auto toto = concat_array(array0, array1);
constexpr auto tata = prod_array(array0);
constexpr auto eee = map_array(array2, array0);
constexpr auto tutu = sort_indexes_array(array0);

int main() {



    auto x = make_seq<a>();
    constexpr auto array4 = make_array(x);

    std::cout << "Array0 :" << array0 << std::endl;
    std::cout << "Array10 :" << array10 << std::endl;
    std::cout << "Array1 :" << array1 << std::endl;
    std::cout << "Array4 :" << array4 << std::endl;

    std::cout << "Prod Array0 : " << tata << std::endl;
    std::cout << "CumProd Array0 : " << cumprod_array(array0) << std::endl;
    std::cout << "Reverse Array0 : " << reverse_array(array0) << std::endl;
    std::cout << "concat Array0+Array1 : " << toto << std::endl;
    std::cout << "Map Array2 and Array1 :" << eee << std::endl;
    std::cout << "Permutate Array3 Array0 : " << permutate_array(array3, array0) << std::endl;
    std::cout << "Permutate Array0 Array0 : " << permutate_array(array3, array3) << std::endl;
    std::cout << "DiffArray Array0 Array0 : " << diff_array<8>(array0) << std::endl;
    std::cout << "sort_index Array0 : " << tutu << std::endl;
    return 0;
}
