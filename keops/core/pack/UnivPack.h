#pragma once

#include <sstream>
#include <utility>

#include "core/pack/Pack.h"

namespace keops {


// "univpack" is a minimal "templating list", defined recursively. ------------------------------
// It allows us to work with "lists" of variables in a formula, at compilation time.

template < int... NS > struct pack; // "pack" is the m

// The empty "univpack", an empty list "< > = []"
template < typename... Args >
struct univpack {
  using FIRST = void;         // [].head() = void
  static const int SIZE = 0;  // len([])   = 0

  // helpers to print the univpack to the standard output
  static void PrintAll(std::ostream& str) {}
  static void PrintComma(std::ostream& str) {}
  static void PrintId(std::ostream& str) {
    str << "univpack< >";
  }

  static void PrintAllIndexSequence(std::ostream& str) {}

  template < class D >        // [].append_first(D) = [D]
  using PUTLEFT = univpack<D>;

  using NEXT = void;          // [].tail() = void
};



// An helper class to convert index_sequence to Pack
template<typename> struct packFromIndSeq{};

template<size_t... Is> struct packFromIndSeq<std::index_sequence<Is...>> {
using type = pack<Is...>;
};

    template<size_t... Is> struct packFromIndSeq<const std::index_sequence<Is...>> {
        using type = pack<Is...>;
    };

// A non-empty univpack, defined recursively as [C] + univpack( Args )
template < class C, typename... Args >
struct univpack<C,Args...> {
  using FIRST = C;             // [C, ...].head() = C
  static const int SIZE = 1+sizeof...(Args); // len([C, ...]) = 1 + len([...])

  // helpers to print the univpack to the standard output
  static void PrintComma(std::ostream& str) {
    str << " ," << std::endl;
  }

  static void PrintAll(std::ostream& str) {
    FIRST::PrintId(str);
    NEXT::PrintComma(str);
    NEXT::PrintAll(str);
  }
  // This function prints binaryOp with template...
  static void PrintAllIndexSequence(std::ostream& str) {
    str << ", {";
    packFromIndSeq<FIRST>::type::PrintAll(str);
    str << "}";
    NEXT::PrintAllIndexSequence(str);
  }

  static void PrintId(std::ostream& str) {
    str << "univpack< " << std::endl;
    PrintAll(str);
    str << " >";
  }

  template < class D >         // [C, ...].append_first(D) = [D, C, ...]
  using PUTLEFT = univpack<D, C, Args...>;

  using NEXT = univpack<Args...>; // [C, ...].tail() = [...]

};

}
