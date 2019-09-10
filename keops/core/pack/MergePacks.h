#pragma once

#include "core/pack/UnivPack.h"
#include "core/pack/ConcatPack.h"
#include "core/pack/RemoveFromPack.h"

namespace keops {


// Merge operation for univpacks. ---------------------------------------------------------------
// MergePacks<[...],[...]> = {...} : a "merged" list, without preservation of ordering
//                                   and uniqueness of elements
// Basically, this operator concatenates two LISTS and sees the result as a SET.
// (Jean :) the syntax becomes *really* convoluted here. I may have made a mistake when commenting.

template < class PACK1, class PACK2 >
struct MergePacks_Alias;

template < class C, typename... Args1, typename... Args2 >
struct MergePacks_Alias< univpack<Args1... >,univpack< C, Args2... > > {         // Merge([...], [C,...])
using tmp = typename RemoveFromPack_Alias< C, univpack< Args1... > > ::type;
using type = typename MergePacks_Alias< ConcatPacks< tmp, univpack< C > >,univpack< Args2... > >::type;
};

template < typename... Args1 >
struct MergePacks_Alias< univpack< Args1... >, univpack<> > {                   // Merge( [], [...])
using type = univpack< Args1... >;
};                                   // = [...]

template < class PACK1, class PACK2 >
using MergePacks = typename MergePacks_Alias< PACK1, PACK2 >::type;

}