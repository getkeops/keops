#pragma once

#include "core/pack/UnivPack.h"

namespace keops {

// shared specifications between kernels.

struct Kernel {

  template < int DIMCHK >
  using CHUNKED_VERSION = void;

  static const bool IS_CHUNKABLE = false;

  template < int DIMCHK >
  using CHUNKED_FORMULAS = univpack<>;

  static const int NUM_CHUNKED_FORMULAS = 0;

  template < int IND >
  using POST_CHUNK_FORMULA = void;
  
  template < int CAT >
  using CHUNKED_VARS = univpack<>;

  template < int CAT >
  using NOTCHUNKED_VARS = univpack<>;
  
};

}
