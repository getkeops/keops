#pragma once
/*
 * This implementation of TensorDot avoid using TaoSeq lib (as in TensoDot.h). It uses the constexp std::array and
 * then requires std=c++17 dialect
 *
 */

#include <assert.h>
#include <iostream>
#include <iterator>
#include <sstream>
#include <utility>
#include <array>
#include <algorithm>

#include "core/utils/keops_math.h"
#include "core/utils/TypesUtils.h"

#include "core/autodiff/BinaryOp.h"
#include "core/pre_headers.h"

#include "core/formulas/maths/Mult.h"
#include "core/formulas/maths/Exp.h"

namespace keops {

    namespace loop_impl {

        template < typename, size_t... I >
        struct Looper_arr;

        template < size_t... Is >
        struct Looper_arr< std::index_sequence< Is...>> {
            template < typename Func >
            constexpr static DEVICE void f(Func &&func) {
                func(std::index_sequence< Is... >{});
            }
        };

        template < size_t I, size_t... Is, size_t... PIs >
        struct Looper_arr< std::index_sequence< PIs... >, I, Is... > {
            template < size_t... Idx, typename Func >
            constexpr static DEVICE void f_help(std::index_sequence< Idx... >, Func &&func) {
                (void) std::initializer_list< int >{(Looper_arr< std::index_sequence< PIs..., Idx >, Is... >::f(func), 0)...};
            }

            template < typename Func >
            constexpr static DEVICE void f(Func &&func) {
                f_help(std::make_index_sequence< I >{}, func);
            }

        };

        template < typename >
        struct loop_t_arr;

        template < size_t... Is >
        struct loop_t_arr< const std::index_sequence< Is...>> {
            using type = Looper_arr< std::index_sequence<>, Is... >;
        };

    }

    template < typename Is >
    using loop_arr = typename loop_impl::loop_t_arr< Is >::type;

    // Constexpr Array manipulation

    // Print an Index sequence

    template < size_t... Is >
    DEVICE void print_index_sequence(std::index_sequence< Is... >) {
        (void) std::initializer_list< int >{(printf("%lu ", Is), 0)...};
        printf("\n");
    }

    template <class T, std::size_t N>
    std::ostream& operator<<(std::ostream& o, const std::array<T, N>& arr)
    {
        std::copy(arr.cbegin(), arr.cend(), std::ostream_iterator<T>(o, " "));
        return o;
    }

template <auto& Arr, size_t... Is>
    static constexpr auto make_seq_impl(std::index_sequence<Is...>) {
        using T = typename std::decay_t<decltype(Arr)>::value_type;
        return std::integer_sequence<T, Arr[Is]...>{};
    }

    template <auto& Arr>
    static constexpr auto make_seq() {
        return make_seq_impl<Arr>(std::make_index_sequence<Arr.size()>());
    }

    template<std::size_t... I>
    DEVICE static constexpr auto make_array(std::index_sequence<I...>) {
        return std::array<size_t, sizeof...(I)>{ {I...} };
    }

    template<std::size_t N>
    static constexpr size_t prod_array(std::array<size_t, N> arr) {
        size_t res = 1;
        for (int i=0; i < static_cast<int>(N); i++){
            res *= arr[i];
        }
        return res;
    }

    template<std::size_t N>
    static constexpr auto cumprod_array(std::array<size_t, N> arr) {
        std::array<size_t, N> res{};
        if (N >= 1) {
            res[N - 1] = 1;
        }
        if (N > 1) {
            for (int i = static_cast<int>(N) - 2; i > -1; i--) {
                res[i] = res[i + 1] * arr[i + 1];
            }
        }
        return res;
    }

    template<std::size_t N>
    HOST_DEVICE static constexpr auto reverse_array(std::array<size_t, N> arr) {
        std::array<size_t, N> res{};
        for (int i=0; i < static_cast<int>(N); i++){
            res[i] = arr[N-i-1];
        }
        return res;
    }

    template<size_t N, size_t M>
    static constexpr auto concat_array(std::array<size_t, N> arr0, std::array<size_t, M> arr1) {
        std::array<size_t, N+M> res{};
        for (int i=0; i<static_cast<int>(N); i++){
            res[i] = arr0[i];
        }
        for (int i=0; i<static_cast<int>(M); i++){
            res[i+N] = arr1[i];
        }
        return res;
    }

    template<size_t N, size_t M>
    HOST_DEVICE static constexpr auto map_array(std::array<size_t, N> ind, std::array<size_t, M> arr1) {
        std::array<size_t, N> res{};
        for (int i= 0; i < static_cast<int>(N); i++){
            res[i] = arr1[ind[i]];
        }
        return res;
    }

    template<size_t N>
    HOST_DEVICE static constexpr auto permutate_array(std::array<size_t, N> perm, std::array<size_t, N> arr) {
        std::array<size_t, N> res{};
        for (int i= 0; i < static_cast<int>(N); i++){
            res[perm[i]] = arr[i];
        }
        return res;
    }

    template<std::size_t B, size_t E>
    constexpr auto make_range_array() {
        static_assert(E >= B, "please provide E >= B");
        std::array<size_t, E - B> res{};
        for (int i= static_cast<int>(B); i < static_cast<int>(E); i++){
            res[i - B] = i;
        }
        return res;
    }

    template< std::size_t N, std::size_t M >
    static constexpr auto diff_array(std::array< size_t, M > ind) {
        auto res = make_range_array<0, N - M >();
        size_t l = 0;
        for (int i= 0; i < static_cast<int>(N); i++){
            bool include = false;
            for (int j=0; j < static_cast<int>(M); j++) {
                if (ind[j] == static_cast<size_t>(i))
                    include = true;
            }
            if (!include) {
                res[l] = static_cast<size_t>(i);
                l++;
            }
        }
        return res;
    }


    template<std::size_t N>
    HOST_DEVICE constexpr size_t sum_multiplies_array(const std::array<size_t, N> arr0, const std::array<size_t, N>  arr1) {
        size_t res = 0;
        for (int i= 0; i < static_cast<int>(N); i++){
            res += arr0[i] * arr1[i];
        }
        return res;
    }

    template<std::size_t N, size_t M>
    HOST_DEVICE constexpr auto first_array(const std::array<size_t, M> arr) {
        std::array<size_t, N> res{};
        for (int i= 0; i < static_cast<int>(N); i++){
            res[i] = arr[i];
        }
        return res;
    }

    template <size_t M, size_t N>
    constexpr auto sort_indexes_array(std::array< size_t, N > arr) {

        // initialize original index locations
        std::array<size_t, N> res{};

        if (N>1){
            for (int i=0; i < static_cast<int>(N); i++) {
                auto tmp =  std::distance(arr.begin(),std::min_element(arr.begin(), arr.end()));
                res[tmp] = i;
                arr[tmp] += M +1;
            }
        }
        return res;
    }




    // Dummy class that stores the indices computes for tensordot
    struct tensordot_indices_arr {
        size_t out_indices;
        size_t a_indices;
        size_t b_indices;
    };


    template < class DIMFA, class DIMFB, class CONTFA, class CONTFB, class PERMUTE >
    struct tensordot_parameters_arr{


        static constexpr auto contfa_arr = make_array(CONTFA{});
        static constexpr auto dimfa_arr = make_array(DIMFA{});
        static constexpr size_t dimfa_size = DIMFA::size();
        static constexpr size_t contfa_size = CONTFA::size();

        static constexpr auto contfb_arr = make_array(CONTFB{});
        static constexpr auto dimfb_arr = make_array(DIMFB{});
        static constexpr size_t dimfb_size = DIMFB::size();
        static constexpr size_t contfb_size = CONTFB::size();

        static constexpr size_t keepdim_size = dimfa_size - contfa_size + dimfb_size - contfb_size;
        static constexpr  std::array<size_t, keepdim_size> permute_arr = make_array(PERMUTE{});

        // Left hand-side
        static constexpr std::array<size_t, dimfa_size - contfa_size > indices_keepdim_a = diff_array< dimfa_size >(contfa_arr); // tao::seq::make_index_sequence< tao::seq::impl::sequence_size< DIMFA >::value >, CONTFA >;
        static constexpr std::array<size_t, dimfa_size - contfa_size > keepdim_a = map_array(indices_keepdim_a, dimfa_arr);
        static constexpr std::array<size_t, contfa_size > contdim_a = map_array(contfa_arr, dimfa_arr);

#if C_CONTIGUOUS
        static constexpr std::array<size_t, dimfa_size> list_stride_dim_a = cumprod_array(dimfa_arr);
#else
        static constexpr std::array<size_t, dimfa_size> list_stride_dim_a = cumprod_array(reverse_array(dimfa_arr));
#endif

        // Right hand-side
        static constexpr std::array<size_t, dimfb_size - contfb_size > indices_keepdim_b = diff_array< dimfb_size >(contfb_arr); // tao::seq::make_index_sequence< tao::seq::impl::sequence_size< DIMFA >{} >, CONTFA >;
        static constexpr std::array<size_t, dimfb_size - contfb_size > keepdim_b = map_array(indices_keepdim_b, dimfb_arr);
        static constexpr std::array<size_t, contfb_size > contdim_b = map_array(contfb_arr, dimfb_arr);

#if C_CONTIGUOUS
        static constexpr std::array<size_t, dimfb_size> list_stride_dim_b = cumprod_array(dimfb_arr);
#else
        static constexpr std::array<size_t, dimfb_size> list_stride_dim_b = cumprod_array(reverse_array(dimfb_arr));
#endif

        // Output
        static constexpr std::array<size_t, keepdim_size > keepdim = concat_array(keepdim_a, keepdim_b);
#if C_CONTIGUOUS
        static constexpr std::array<size_t, keepdim_size > list_stride_keepdim = cumprod_array(permutate_array(permute_arr, keepdim));
#else
        static constexpr std::array<size_t, keepdim_size > list_stride_keepdim = cumprod_array(reverse_array(permutate_array(permute_arr, keepdim))) ;
#endif
        static constexpr size_t dimout = prod_array(keepdim);

        // test
        static constexpr auto contdim_a_impl = make_seq<contdim_a>();
        using contdim_a_t = decltype(contdim_a_impl);
        static constexpr auto contdim_b_impl = make_seq<contdim_b>();
        using contdim_b_t = decltype(contdim_b_impl);

        static_assert(std::is_same< contdim_a_t, contdim_b_t >::value,
                      "In TensorDot: contracting dimensions should be the same");

        static constexpr auto ttmp = permutate_array(permute_arr, permute_arr);
        static constexpr auto test_impl = make_seq<ttmp>();
        using test_impl_t = decltype(test_impl);
        static constexpr auto ttmp2 = make_range_array<0, keepdim_size>();
        static constexpr auto seeq_impl = make_seq<ttmp2>();
        using seeq_impl_t = decltype(seeq_impl);

        static_assert(std::is_same< test_impl_t, seeq_impl_t>::value,
                         "In TensorDot: PERMUTE should be a permutation index_sequence.");
        // end test

        // Loop: in this code we choose to loop on the keepdims first and then on the contraction dims.
        static constexpr std::array<size_t, keepdim_size + contfa_size > loopdim = concat_array(keepdim, contdim_a);

        constexpr static size_t dimloop = prod_array(loopdim);
        constexpr static size_t number_of_dimloop = dimfa_size - contfa_size + dimfb_size;

        static constexpr std::array<size_t, dimfa_size > ala =
                concat_array(make_range_array<0, keepdim_a.size() >(), make_range_array< keepdim.size(), number_of_dimloop>());
        static constexpr std::array<size_t, dimfa_size > ali = concat_array(indices_keepdim_a, contfa_arr);
        static constexpr std::array<size_t, dimfa_size > list_indices_a_intot = permutate_array(ali, ala);

        static constexpr std::array<size_t, dimfb_size > bla =
                concat_array(make_range_array< keepdim_a.size(), keepdim.size()>(), make_range_array< keepdim.size(), number_of_dimloop>());
        static constexpr std::array<size_t, dimfb_size > bli = concat_array(indices_keepdim_b, contfb_arr);
        static constexpr std::array<size_t, dimfb_size > list_indices_b_intot = permutate_array(bli, bla);

        // used to compute the Gradient
        static constexpr std::array<size_t, dimfa_size - contfa_size > list_indices_keepdim_a_inout = make_range_array< 0, keepdim_a.size() >();
        static constexpr std::array<size_t, contfa_size > reordered_contfa = permutate_array(sort_indexes_array<dimfb_size>(contfb_arr), contfa_arr);
        static constexpr std::array<size_t, dimfa_size - contfa_size > reordered_keepdim_a =
                permutate_array(sort_indexes_array<dimfa_size - contfa_size>(map_array(list_indices_keepdim_a_inout, permute_arr)), indices_keepdim_a );
        static constexpr std::array<size_t, dimfa_size > moveaxis_a = concat_array(reordered_keepdim_a, reordered_contfa);

        static constexpr std::array<size_t, dimfb_size - contfb_size >  list_indices_keepdim_b_inout = make_range_array< keepdim_a.size(), keepdim.size() >();
        static constexpr std::array<size_t, contfb_size >  reordered_contfb = permutate_array(sort_indexes_array<dimfa_size>(contfa_arr), contfb_arr);
        static constexpr std::array<size_t, dimfb_size - contfb_size >  reordered_keepdim_b =
                permutate_array(sort_indexes_array<dimfb_size - contfb_size>(map_array(list_indices_keepdim_b_inout, permute_arr)), indices_keepdim_b);
        static constexpr std::array<size_t, dimfb_size > moveaxis_b = concat_array(reordered_keepdim_b, reordered_contfb);

        //////////////////////////////////// cast to type
        static constexpr std::array< size_t, keepdim_size>  dimfa_grad = permutate_array(permute_arr, keepdim);
        static constexpr auto DIMFA_GRAD_imp = make_seq<dimfa_grad>();
        using DIMFA_GRAD = decltype(DIMFA_GRAD_imp);

        static constexpr std::array< size_t, dimfb_size - contfb_size > contfa_grad=map_array(list_indices_keepdim_b_inout, permute_arr);
        static constexpr auto CONTFA_GRAD_imp = make_seq<contfa_grad>();
        using CONTFA_GRAD = decltype(CONTFA_GRAD_imp);

        static constexpr std::array< size_t, dimfa_size - contfa_size > contfb_grad=map_array(list_indices_keepdim_a_inout, permute_arr);
        static constexpr auto CONTFB_GRAD_imp = make_seq<contfb_grad>();
        using CONTFB_GRAD = decltype(CONTFB_GRAD_imp);

        static constexpr auto loopdim_imp = make_seq<loopdim>();
        using loopdim_t = decltype(loopdim_imp);

        static constexpr auto indices_keepdim_b_impl = make_seq<indices_keepdim_b>();
        using indices_keepdim_b_t = decltype(indices_keepdim_b_impl);

        static constexpr auto indices_keepdim_a_impl = make_seq<indices_keepdim_a>();
        using indices_keepdim_a_t = decltype(indices_keepdim_a_impl);

        static constexpr auto moveaxis_a_impl = make_seq<moveaxis_a>();
        using moveaxis_a_t = decltype(moveaxis_a_impl);

        static constexpr auto moveaxis_b_impl = make_seq<moveaxis_b>();
        using moveaxis_b_t = decltype(moveaxis_b_impl);

        /////////////////////////////////////

        template < class IND >
        DEVICE constexpr static tensordot_indices_arr compute_tensordot_indices_arr(IND) {
            auto ind_arr = make_array(IND{});

            // a_indices
            std::array<size_t, dimfa_size > list_indices_a = map_array(list_indices_a_intot, ind_arr);
#if C_CONTIGUOUS
            size_t a_indices = sum_multiplies_array(list_stride_dim_a, list_indices_a);
#else
            size_t a_indices = sum_multiplies_array(list_stride_dim_a, reverse_array(list_indices_a));
#endif

            // b_indices
            std::array<size_t, dimfb_size >  list_indices_b  = map_array(list_indices_b_intot, ind_arr);
#if C_CONTIGUOUS
            size_t b_indices = sum_multiplies_array(list_stride_dim_b, list_indices_b);
#else
            size_t b_indices = sum_multiplies_array(list_stride_dim_b, reverse_array(list_indices_b));
#endif

            // out_indices
            std::array<size_t, keepdim_size >  list_indices_keepdim = permutate_array(permute_arr, first_array< keepdim_size >(ind_arr ));
#if C_CONTIGUOUS
            size_t out_indices = sum_multiplies_array(list_stride_keepdim, list_indices_keepdim);
#else
            size_t out_indices = sum_multiplies_array(list_stride_keepdim, reverse_array(list_indices_keepdim));
#endif

            return tensordot_indices_arr{out_indices, a_indices, b_indices};
        }

        template < typename Func >
        struct compute_tensordot_indices_t_arr {
            template < size_t... Is >
            DEVICE void operator()(std::index_sequence< Is... > x) {
                _f(compute_tensordot_indices_arr(x));
            }

            Func &_f;
            DEVICE compute_tensordot_indices_t_arr(Func &&f) : _f(f) {}
        };

        template < typename Func >
        static DEVICE auto compute_tensordot_indices_apply_arr(Func &&f) {
            return compute_tensordot_indices_t_arr< Func >(std::forward< Func >(f));
        }
/*
        tensordot_parameters_arr(){
            printf("\nDIMFA: "); print_index_sequence(DIMFA{});
            printf("DIMFB: "); print_index_sequence(DIMFB{});
            printf("CONTFA: "); print_index_sequence(CONTFA{});
            printf("CONTFB: "); print_index_sequence(CONTFB{});
            printf("PERMUTE: "); print_index_sequence(PERMUTE{});


            std::cout << "list_indices_keepdim_a_inout: " << list_indices_keepdim_a_inout << std::endl;
            std::cout << "reordered_contfa: " << reordered_contfa << std::endl;
            std::cout << "reordered_keepdim_a: " << reordered_keepdim_a << std::endl;
            std::cout << "moveaxis_a: " << moveaxis_a << std::endl;
            std::cout << "list_indices_keepdim_b_inout: " << list_indices_keepdim_b_inout << std::endl;
            std::cout << "reordered_contfb: " << reordered_contfb << std::endl;
            std::cout << "reordered_keepdim_b: " << reordered_keepdim_b << std::endl;
            std::cout << "moveaxis_b: " << moveaxis_b << std::endl;

            printf("-----------------------\n");
        }
*/
    };

/////////////////////////////////////////////////////////////////////////
////              Tensor dot product      A : b                      ////
/////////////////////////////////////////////////////////////////////////


    template < class A, class B, class DIMFA, class DIMFB, class CONTFA, class CONTFB, class PERMUTEDIMOUT=std::make_index_sequence<
            DIMFA::size() + DIMFB::size() - 2 * CONTFA::size()>>
    struct TensorDot : BinaryOp< TensorDot, A, B, DIMFA, DIMFB, CONTFA, CONTFB, PERMUTEDIMOUT > {
        // A is vector of size p ** n, interpreted as matrix (column major), B is vector of size p ** m, interpreted as column vector
        // n=3 and m=2 are assume to be known
        // output is vector of size n

        static_assert(DIMFA::size() > 0, "Please provide a non empty DIMA");
        static_assert(DIMFB::size() > 0, "Please provide a non empty DIMB");

        using parameters = tensordot_parameters_arr< DIMFA, DIMFB, CONTFA, CONTFB, PERMUTEDIMOUT >;
        //parameters pap = tensordot_parameters_arr< DIMFA, DIMFB, CONTFA, CONTFB, PERMUTEDIMOUT >{};

        static_assert(prod_array(parameters::dimfa_arr) == A::DIM, "DIMA is not consistant with dimension of A");
        static_assert(prod_array(parameters::dimfb_arr) == B::DIM, "DIMB is not consistant with dimension of B");

        static const int DIM = parameters::dimout;

        static void PrintIdString(std::stringstream &str) { str << ":"; }

        template < typename TYPE >
        static DEVICE INLINE void Operation(TYPE *out, TYPE *inA, TYPE *inB) {
#pragma unroll
            for (int i = 0; i < DIM; i++)
                out[i] = cast_to<TYPE>(0.0f);
            loop_arr< typename parameters::loopdim_t >::f(parameters::compute_tensordot_indices_apply_arr([&out, &inA, &inB](
                    tensordot_indices_arr td) {
                out[td.out_indices] = keops_fma(inA[td.a_indices], inB[td.b_indices], out[td.out_indices]);
            }));
        }

        template < class V, class GRADIN >
        using DiffTA = typename A::template DiffT< V, GRADIN >;

        template < class V, class GRADIN >
        using DiffTB = typename B::template DiffT< V, GRADIN >;

  template < class V, class GRADIN >
  using DiffT = Add<
      DiffTA< V, TensorDot< GRADIN, B,
                            typename parameters::DIMFA_GRAD,                //
                            DIMFB,                                          // 4 2
                            typename parameters::CONTFA_GRAD,               // .
                            typename parameters::indices_keepdim_b_t,       // .
                            typename parameters::moveaxis_a_t>>,            // 1, 2 0
      DiffTB< V, TensorDot< GRADIN, A,
                            typename parameters::DIMFA_GRAD,                //
                            DIMFA,                                          // 2, 3, 4
                            typename parameters::CONTFB_GRAD,               //0
                            typename parameters::indices_keepdim_a_t,       //1
                            typename parameters::moveaxis_b_t>>             // . , 0 1
  >;

    };

#define TensorDot(f, g, ...) KeopsNS<TensorDot<decltype(InvKeopsNS(f)),decltype(InvKeopsNS(g)), __VA_ARGS__>>()

}
