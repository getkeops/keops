#pragma once



//////////////////////////////////////////////
//    templated plus and minus infinity     //
//////////////////////////////////////////////

#ifdef __CUDACC__
  #include <npp.h>
  #define INFINITY_FLOAT NPP_MAXABS_32F
  #define INFINITY_DOUBLE NPP_MAXABS_64F
#else
  #include <limits>
  #define INFINITY_FLOAT std::numeric_limits< float >::infinity()
  #define INFINITY_DOUBLE std::numeric_limits< double >::infinity()
#endif

namespace keops {

template < typename TYPE >
struct PLUS_INFINITY;

template <>
struct PLUS_INFINITY< float > {
  static constexpr float value = INFINITY_FLOAT;
};

template <>
struct PLUS_INFINITY< double > {
  static constexpr double value = INFINITY_DOUBLE;
};

template < typename TYPE >
struct NEG_INFINITY;

template <>
struct NEG_INFINITY< float > {
  static constexpr float value = -INFINITY_FLOAT;
};

template <>
struct NEG_INFINITY< double > {
  static constexpr double value = -INFINITY_DOUBLE;
};

}
