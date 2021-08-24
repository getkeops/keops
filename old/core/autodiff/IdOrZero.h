#pragma once

#include "core/pre_headers.h"
#include "core/formulas/constants/Zero.h"

namespace keops {

//////////////////////////////////////////////////////////////
////                        ZEROS                         ////
//////////////////////////////////////////////////////////////

//template < int DIM > struct Zero; // Declare Zero in the header, for IdOrZero_Alias. _Implementation below.

// IdOrZero( Vref, V, Fun ) = FUN                   if Vref == V
//                            Zero (of size V::DIM) if Vref != V
template < class Vref, class V, class FUN >
struct IdOrZero_Alias {
  using type = Zero< V::DIM >;
};

template < class V, class FUN >
struct IdOrZero_Alias< V, V, FUN > {
  using type = FUN;
};

template < class Vref, class V, class FUN >
using IdOrZero = typename IdOrZero_Alias< Vref, V, FUN >::type;

}
