#pragma once

namespace keops {


// Remove an element from a univpack

template < class C, class PACK >
struct RemoveFromPack_Alias;

template < class C >
struct RemoveFromPack_Alias<C,univpack<>> { // RemoveFrom( C, [] )
  using type = univpack<>;    // []
};

template < class C, class D, typename... Args >
struct RemoveFromPack_Alias<C,univpack<D,Args...>> { // RemoveFrom( C, [D, ...] )
using tmp = typename RemoveFromPack_Alias<C,univpack<Args...>>::type;
using type = typename tmp::template PUTLEFT<D>;     // = [D] + RemoveFrom( C, [...] )
};

template < class C, typename... Args >
struct RemoveFromPack_Alias<C,univpack<C,Args...>> { // RemoveFrom( C, [C, ...] )
using type = typename RemoveFromPack_Alias<C,univpack<Args...>>::type;
};

//template < class C, class PACK >
//using RemoveFromPack = typename RemoveFromPack_Alias<C,PACK>::type;


}