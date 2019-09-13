#pragma once

#include "core/pack/PackVal.h"
#include "core/pack/ZeroPack.h"
#include "core/pack/ReplaceInPack.h"

namespace keops {

////////////////////////////////////////////////////////
// Check that all values in a pack of ints are unique //
////////////////////////////////////////////////////////

// here we count the number of times each value appears, then
// test if the sum is > 1 (which is not an optimal algorithm, it could be improved...)
template<class P, class TAB = ZeroPack < P::MAX + 1> >
struct CheckAllDistinct_BuildTab {
  static const int VAL = PackVal< TAB, P::FIRST >::type::Val;
  using NEWTAB = ReplaceInPack< TAB, VAL + 1, P::FIRST >;
  using type = typename CheckAllDistinct_BuildTab< typename P::NEXT, NEWTAB >::type;
};

template<class TAB>
struct CheckAllDistinct_BuildTab< pack<>, TAB > {
  using type = TAB;
};

template<class P>
struct CheckAllDistinct {
  using TAB = typename CheckAllDistinct_BuildTab< P >::type;
  static const bool val = TAB::MAX < 2;
};

}