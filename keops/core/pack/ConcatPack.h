#pragma once

#include "core/pack/UnivPack.h"


namespace keops {

// OPERATIONS ON PACKS AND UNIVPACKS ============================================================
// Once again, a convoluted syntax to write the "concatenation" of two lists. -------------------
// ConcatPacks<[...],[...]> = [..., ...]  (for packs or univpacks)
template < class PACK1, class PACK2 >
struct ConcatPacks_Alias {
  using type = int;
}; // default dummy type

template < int... IS, int... JS >
struct ConcatPacks_Alias< pack< IS... >, pack< JS... > > {
using type = pack< IS..., JS... >;
};

template < typename... Args1, typename... Args2 >
struct ConcatPacks_Alias< univpack< Args1... >, univpack< Args2... > > {
using type = univpack< Args1..., Args2... >;
};

template < class PACK1, class PACK2 >
using ConcatPacks = typename ConcatPacks_Alias< PACK1, PACK2 >::type;

}