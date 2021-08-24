#pragma once

namespace keops {


// IsSameType<A,B> = false and IsSameType<A,A> = true
template < class A, class B >
struct IsSameType_Alias {
  static const bool val = false;
};

template < class A >
struct IsSameType_Alias<A,A> {
  static const bool val = true;
};

template < class A, class B >
struct IsSameType {
  static const bool val = IsSameType_Alias<A,B>::val;
};

}
