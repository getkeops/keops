#pragma once

namespace keops {

// Conditional type, a templating emulator of a conditional statement. --------------------------
// This convoluted syntax allows us to write
// CondType<A,B,1> = A,  CondType<A,B,0> = B
template < class A, class B, bool TEST >
struct CondType_Alias;

template < class A, class B >
struct CondType_Alias<A,B,true> {
  using type = A;
};

template < class A, class B >
struct CondType_Alias<A,B,false> {
  using type = B;
};

template < class A, class B, bool TEST >
using CondType = typename CondType_Alias<A,B,TEST>::type;


}