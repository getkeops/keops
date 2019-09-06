#pragma once

namespace keops {

/////////////////////////////////////////////////////
// get the value at position N from a pack of ints //
/////////////////////////////////////////////////////

template< class P, int N >
struct PackVal {
  using type = typename PackVal< typename P::NEXT, N - 1 >::type;
};

template< class P >
struct PackVal< P, 0 > {
  using type = PackVal< P, 0 >;
  static const int Val = P::FIRST;
};

}