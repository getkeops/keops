#pragma once

#include "core/formulas/Factorize.h"

namespace keops {

// helper for counting the number of occurrences of a subformula in a formula

template < class F, class G >
struct CountIn_ {
  static const int val = 0;
};

template < class F >
struct CountIn_< F, F > {
  static const int val = 1;
};

template < class F, class G >
struct CountIn {
  static const int val = CountIn_< F, G >::val;
};

template < template < class, int... > class OP, class F, class G, int... NS >
struct CountIn< OP< F, NS... >, G > {
  static const int val = CountIn_< OP< F, NS... >, G >::val + CountIn< F, G >::val;
};

template < template < class, class > class OP, class FA, class FB, class G >
struct CountIn< OP< FA, FB >, G > {
  static const int val = CountIn_< OP< FA, FB >, G >::val + CountIn< FA, G >::val + CountIn< FB, G >::val;
};

// specializing CountIn
template < class F, class G >
struct Factorize_Impl;

template < class F, class G, class H >
struct CountIn< Factorize_Impl< F, G >, H > {
  static const int val =
      CountIn_< Factorize_Impl< F, G >, H >::val + CountIn< F, H >::val - CountIn< G, H >::val * CountIn< F, G >::val
          + CountIn< G, H >::val;
};

}