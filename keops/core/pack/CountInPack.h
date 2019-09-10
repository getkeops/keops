#pragma once

namespace keops {


// count number of occurrences of a type in a univpack

template < class C, class PACK >
struct CountInPack_Alias {
  static const int N = 0;
};

template < class C, typename... Args >
struct CountInPack_Alias<C,univpack<C,Args...>> { // CountIn( C, [C, ...] )
static const int N = 1+CountInPack_Alias<C,univpack<Args...>>::N;
};

template < class C, class D, typename... Args >
struct CountInPack_Alias<C,univpack<D,Args...>> { // CountIn( C, [D, ...] )
static const int N = CountInPack_Alias<C,univpack<Args...>>::N;
};

template < class C >
struct CountInPack_Alias<C,univpack<>> {        // CountIn( C, [] )
  static const int N = 0;
};

//template < class C, class PACK >
//static const int CountInPack() { return CountInPack_Alias<C,PACK>::N; }

}
