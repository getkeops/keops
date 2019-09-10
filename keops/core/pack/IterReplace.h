#pragma once

#include "core/pack/UnivPack.h"

namespace keops {

// iterate replace operator

template<class F, class G, class PACK>
struct IterReplace_Impl {
  using CURR = typename F::template Replace<G, typename PACK::FIRST>;
  using type = typename IterReplace_Impl<F, G, typename PACK::NEXT>::type::template PUTLEFT<CURR>;
};

template<class F, class G>
struct IterReplace_Impl<F, G, univpack<>> {
  using type = univpack<>;
};

template<class F, class G, class PACK>
using IterReplace = typename IterReplace_Impl<F, G, PACK>::type;


}