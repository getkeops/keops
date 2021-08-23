# The Art of C++ / Sequences

[![Release](https://img.shields.io/github/release/taocpp/sequences.svg)](https://github.com/taocpp/sequences/releases/latest)
[![Download](https://api.bintray.com/packages/taocpp/public-conan/sequences%3Ataocpp/images/download.svg)](https://bintray.com/taocpp/public-conan/sequences%3Ataocpp/_latestVersion)
[![TravisCI](https://travis-ci.org/taocpp/sequences.svg)](https://travis-ci.org/taocpp/sequences)
[![AppVeyor](https://ci.appveyor.com/api/projects/status/github/taocpp/sequences?svg=true)](https://ci.appveyor.com/project/taocpp/sequences)

[The Art of C++](https://taocpp.github.io/) / Sequences is a zero-dependency C++11 header-only library that provides efficient algorithms to generate and work on variadic templates and [`std::integer_sequence`](http://en.cppreference.com/w/cpp/utility/integer_sequence).

## Compatibility

* Requires C++11 or newer.
* Tested with GCC 4.8+, Clang 3.4+, Xcode 6+ and Visual Studio 2017+.

## Provided algorithms and examples

* All provided templates are in the nested namespace `tao::seq`.
* All templates don't use C++14 features, therefore being compatible with C++11. Sometimes, C++14/C++17 features are used conditionally, taking advantage of newer language features when available but providing C++11-compatible implementations otherwise.
* All templates use `tao::seq::integer_sequence`, etc. internally, therefore being compatible with C++11.
* All templates use `tao::seq::make_integer_sequence`, etc. internally, therefore using the most scalable solution available.

#### Header `tao/seq/integer_sequence.hpp`

Provides:

* `integer_sequence< typename T, T N >`
* `index_sequence< std::size_t N >`

Notes:

* When available (C++14 or newer), the above are type-aliases for `std::integer_sequence` and `std::index_sequence`.

#### Header `tao/seq/make_integer_sequence.hpp`

Efficient versions of sequence generators.

* `make_integer_sequence< typename T, T N >`
* `make_index_sequence< std::size_t N >`
* `index_sequence_for< typename... Ts >`

Examples:

* `make_integer_sequence<int,0>` ➙ `integer_sequence<int>`
* `make_integer_sequence<int,1>` ➙ `integer_sequence<int,0>`
* `make_integer_sequence<int,3>` ➙ `integer_sequence<int,0,1,2>`
* `make_index_sequence<0>` ➙ `index_sequence<>`
* `make_index_sequence<1>` ➙ `index_sequence<0>`
* `make_index_sequence<5>` ➙ `index_sequence<0,1,2,3,4>`
* `index_sequence_for<int,void,long>` ➙ `index_sequence<0,1,2>`

Notes:

libc++ already has very efficient versions for the above, so they are pulled in with a using-declaration.
Only if we don't know if the STL's versions are at least O(log N) we provide our own implementations.

Our own implementation has O(log N) instantiation depth.
This allows for very large sequences without the need to increase the compiler's default instantiation depth limits.
For example, GCC and Clang generate `index_sequence<10000>` in ~0.15s (on my machine, of course).
The standard library version from libstdc++, when trying to create `index_sequence<5000>` and with its O(N) implementation, requires ~30s, >3GB of RAM and `-ftemplate-depth=5100`.

#### Header `tao/seq/make_integer_range.hpp`

Generate half-open ranges of integers.

* `make_integer_range< typename T, T N, T M >`
* `make_index_range< std::size_t N, std::size_t M >`

Examples:

* `make_integer_range<int,3,7>` ➙ `integer_sequence<int,3,4,5,6>`
* `make_integer_range<int,7,3>` ➙ `integer_sequence<int,7,6,5,4>`
* `make_integer_sequence<int,-2,2>` ➙ `integer_sequence<int,-2,-1,0,1>`
* `make_index_range<5,5>` ➙ `index_sequence<>`
* `make_index_range<2,5>` ➙ `index_sequence<2,3,4>`

#### Header `tao/seq/sum.hpp`

Integral constant to provide the sum of `Ns`.
If no `Ns` are given, the result is `T(0)`.

* `sum< typename T, T... Ns >`
* `sum< typename S >`

Examples:

* `sum<int,1,4,3,1>::value` ➙ `9`
* `sum<make_index_sequence<5>>::value` ➙ `10`

#### Header `tao/seq/prod.hpp`

Integral constant to provide the product of `Ns`.
If no `Ns` are given, the result is `T(1)`.

* `prod< typename T, T... Ns >`
* `prod< typename S >`

Examples:

* `prod<int>::value` ➙ `1`
* `prod<int,1,4,3,-1>::value` ➙ `-12`

#### Header `tao/seq/partial_sum.hpp`

Integral constant to provide the sum of the first `I` elements.

* `partial_sum< std::size_t I, typename T, T... Ns >`
* `partial_sum< std::size_t I, typename S >`

Examples:

* `partial_sum<0,int,1,4,3,1>::value` ➙ `0`
* `partial_sum<2,int,1,4,3,1>::value` ➙ `5`
* `partial_sum<4,make_index_sequence<5>>::value` ➙ `6`

#### Header `tao/seq/partial_prod.hpp`

Integral constant to provide the product of the first `I` elements of `Ns`.

* `partial_prod< std::size_t I, typename T, T... Ns >`
* `partial_prod< std::size_t I, typename S >`

Examples:

* `partial_prod<0,int,2,5,3,2>::value` ➙ `1`
* `partial_prod<1,int,2,5,3,2>::value` ➙ `2`
* `partial_prod<2,int,2,5,3,2>::value` ➙ `10`
* `partial_prod<4,int,2,5,3,2>::value` ➙ `60`

#### Header `tao/seq/exclusive_scan.hpp`

Provides a sequence with the exclusive scan of the input sequence.

* `exclusive_scan_t< typename OP, typename T, T Init, T... Ns >`
* `exclusive_scan_t< typename OP, typename S, T Init >`

Examples:

* `exclusive_scan_t<op::plus,int,0,1,4,0,3,1>` ➙ `integer_sequence<int,0,1,5,5,8>`
* `using S = index_sequence<3,1,4,1,5,9,2,6>;
* `exclusive_scan_t<op::multiplies,S,1>` ➙ `index_sequence<3,3,12,12,60,540,1080,6480>`

#### Header `tao/seq/inclusive_scan.hpp`

Provides a sequence with the inclusive scan of the input sequence.

* `inclusive_scan_t< typename OP, typename T, T... Ns >`
* `inclusive_scan_t< typename OP, typename S >`

Examples:

* `inclusive_scan_t<op::plus,int,1,4,0,3,1>` ➙ `integer_sequence<int,1,5,5,8,9>`

#### Header `tao/seq/zip.hpp`

Applies a binary operation to elements from two sequences.

* `zip_t< typename OP, typename L, typename R >`

Notes:

Both sequences may have a different element type, the resulting sequence's type is calculated with `std::common_type_t`.

#### Header `tao/seq/plus.hpp`

Provides a sequence which is the element-wise sum of its input sequences.

* `plus_t< typename L, typename R >`

Notes:

Both sequences may have a different element type, the resulting sequence's type is calculated with `std::common_type_t`.

Examples:

* `using A = index_sequence<1,4,0,3,1>`
* `using B = make_index_sequence<5>`
* `plus_t<A,B>` ➙ `index_sequence<1,5,2,6,5>`

#### Header `tao/seq/minus.hpp`

Provides a sequence which is the element-wise sum of its input sequences.

* `minus_t< typename L, typename R >`

Notes:

Both sequences may have a different element type, the resulting sequence's type is calculated with `std::common_type_t`.

Examples:

* `using A = integer_sequence<int,1,4,0,3,1>`
* `using B = integer_sequence<int,0,1,2,3,4>`
* `minus_t<A,B>` ➙ `integer_sequence<int,1,3,-2,0,-3>`
* `minus_t<B,A>` ➙ `integer_sequence<int,-1,-3,2,0,3>`

#### Header `tao/seq/multiply.hpp`

Provides a sequence which is the element-wise product of its input sequences.

* `multiply_t< typename L, typename R >`

Notes:

Both sequences may have a different element type, the resulting sequence's type is calculated with `std::common_type_t`.

Examples:

* `using A = index_sequence<1,5,2,3,1>`
* `using B = index_sequence<3,0,2,4,1>`
* `multiply_t<A,B>` ➙ `index_sequence<3,0,4,12,1>`

#### Header `tao/seq/head.hpp`

Integral constant to provide the first element of a non-empty sequence.

* `head< typename T, T... >`
* `head< typename S >`

#### Header `tao/seq/tail.hpp`

Removed the first element of a non-empty sequence.

* `tail_t< typename T, T... >`
* `tail_t< typename S >`

#### Header `tao/seq/select.hpp`

Integral constant to provide the `I`-th element of a non-empty sequence.

* `select< std::size_t I, typename T, T... >`
* `select< std::size_t I, typename S >`

#### Header `tao/seq/first.hpp`

Sequence that contains only the first `I` elements of a given sequence.

* `first_t< std::size_t I, typename T, T... >`
* `first_t< std::size_t I, typename S >`

#### Header `tao/seq/concatenate.hpp`

Concatenate the values of all sequences.

* `concatenate_t< typename... Ts >`

Notes:

The sequences may have different element types, the resulting sequence's type is calculated with `std::common_type_t`.

#### Header `tao/seq/difference.hpp`

Builds the difference of two sequences, i.e. a sequence that contains all elements of `T` that are not in `U`.

* `difference_t< typename T, typename U >`

Examples:

* `using A = index_sequence<1,5,2,3,1,7>`
* `using B = index_sequence<2,1>`
* `difference_t<A,B>` ➙ `index_sequence<5,3,7>`

Notes:

Both sequences may have a different element type, the resulting sequence's type is calculated with `std::common_type_t`.

#### Header `tao/seq/accumulate.hpp`

Result of a left fold of the given values over `OP`.

* `accumulate< typename OP, typename T, T... >`
* `accumulate< typename OP, typename S >`

#### Header `tao/seq/reduce.hpp`

Reduces the given values in an unspecified order over `OP`.

* `reduce< typename OP, typename T, T... >`
* `reduce< typename OP, typename S >`

#### Header `tao/seq/min.hpp`

Integral constant to provide the minimum value.

* `min< typename T, T... >`
* `min< typename S >`

#### Header `tao/seq/max.hpp`

Integral constant to provide the maximum value.

* `max< typename T, T... >`
* `max< typename S >`

#### Header `tao/seq/map.hpp`

Map a sequence of indices to a sequence of values.

* `map_t< typename I, typename M >`

Examples:

* `using I = index_sequence<1,0,3,2,1,1,3>`
* `using M = integer_sequence<int,5,6,-7,8,9>`
* `map_t<I,M>` ➙ `integer_sequence<int,6,5,8,-7,6,6,8>`

#### Header `tao/seq/is_all.hpp`

Integral constant which is true if all boolean parameters are true (logical and).

* `is_all< bool... >`

Examples:

* `is_all<true,true,true,true>::value` ➙ `true`
* `is_all<true,true,false,true>::value` ➙ `false`
* `is_all<>::value` ➙ `true`

#### Header `tao/seq/is_any.hpp`

Integral constant which is true if any boolean parameter is true (logical or).

* `is_any< bool... >`

Examples:

* `is_any<false,true,false,false>::value` ➙ `true`
* `is_any<false,false,false,false>::value` ➙ `false`
* `is_any<>::value` ➙ `false`

#### Header `tao/seq/contains.hpp`

Integral constant which is true if an element `N` is part of a list of elements `Ns...`.

* `contains< typename T, T N, T... Ns>`
* `contains< typename S, T N>`

Examples:

* `contains<int,0>` ➙ `false`
* `contains<int,0,0>` ➙ `true`
* `contains<int,0,1>` ➙ `false`
* `contains<int,0,1,2,3,4,5>` ➙ `false`
* `contains<int,3,1,2,3,4,5>` ➙ `true`
* `using A = integer_sequence<int,1,2,3,4,5>`
* `contains<A,0>` ➙ `false`
* `contains<A,3>` ➙ `true`

#### Header `tao/seq/index_of.hpp`

Integral constant which is the smallest index of an element `N` in a list of elements `Ns...`.

* `index_of< typename T, T N, T... Ns>`
* `index_of< typename S, T N>`

Note: `Ns...` must contain `N`, otherwise a `static_assert` is triggered.

Examples:

* `index_of<int,0,0>` ➙ `0`
* `index_of<int,3,1,2,3,4,5>` ➙ `2`
* `using A = integer_sequence<int,1,2,3,4,5>`
* `index_of<A,3>` ➙ `2`

#### Header `tao/seq/scale.hpp`

Scales a sequence by a factor `F`.

* `scale< typename T, T F, T... Ns>`
* `scale< typename S, T F>`

Examples:

* `scale<int,0,0>` ➙ `integer_sequence<int,0>`
* `scale<int,2,-1,2,0,1,5>` ➙ `integer_sequence<int,-2,4,0,2,10>`
* `using A = integer_sequence<int,-1,2,4>`
* `scale<A,3>` ➙ `integer_sequence<int,-3,6,12>`

#### Header `tao/seq/at_index.hpp`

Returns the `I`-th type from a list of types `Ts...`.

* `at_index_t< std::size_t I, typename... Ts >`

Examples:

* `at_index<0,bool,int,void,char*>` ➙ `bool`
* `at_index<2,bool,int,void,char*>` ➙ `void`

#### Header `tao/seq/reverse.hpp`

Reverses a sequence.

Examples:

* `reverse_t<int,1,4,0,3,2>` ➙ `integer_sequence<int,2,3,0,4,1>`
* `reverse_t<index_sequence<1,4,0,3,2>>` ➙ `index_sequence<int,2,3,0,4,1>`

#### Header `tao/seq/sort.hpp`

Sort a sequence according to a given predicate.

* `sort_t< typename OP, typename T, T... Ns >`
* `sort_t< typename OP, typename S >`

Examples:

Given a predicate `less`...

    struct less
    {
       template< typename T, T A, T B >
       using apply = std::integral_constant< bool, ( A < B ) >;
    };

* `sort_t<less,int,7,-2,3,0,4>` ➙ `integer_sequence<int,-2,0,3,4,7>`
* `using S = index_sequence<39,10,2,4,10,2>`
* `sort_t<less,S>` ➙ `index_sequence<2,2,4,10,10,39>`

## Changelog

### 2.0.1

Released 2019-11-09

* Fixed Conan upload.

### 2.0.0

Released 2019-11-07

* Generalized `exclusive_scan` and `inclusive_scan`.
* Split `fold` into `accumulate` and `reduce`.
* Added `first`, `reverse`, `prod`, `partial_prod`, `multiplies`, `difference`, and `sort`.
* Improved compile-times for `at_index`.
* Added `make_index_of_sequence`, `permutate`, and `sort_index` to contrib (unofficial).

### 1.0.2

Released 2018-07-22

* Added documentation for the remaining headers.

### 1.0.1

Released 2018-07-21

* Removed `type_by_index`, use `at_index` instead.

### 1.0.0

Released 2018-06-29

* Initial release.

## License

The Art of C++ is certified [Open Source](http://www.opensource.org/docs/definition.html) software. It may be used for any purpose, including commercial purposes, at absolutely no cost. It is distributed under the terms of the [MIT license](http://www.opensource.org/licenses/mit-license.html) reproduced here.

> Copyright (c) 2015-2019 Daniel Frey
>
> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
