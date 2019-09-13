#pragma once

namespace keops {

//////////////////////////////////////////////////////////
// create pack of arbitrary size filled with zero value //
//////////////////////////////////////////////////////////

template < int N >
struct ZeroPack_Alias {
  using type = typename ZeroPack_Alias< N - 1 >::type::template PUTLEFT< 0 >;
};

template < >
struct ZeroPack_Alias< 0 > {
  using type = pack<>;
};

template < int N >
using ZeroPack = typename ZeroPack_Alias< N >::type;



}