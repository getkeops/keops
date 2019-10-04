#ifndef RKEOPS_DATA_TYPE_H
#define RKEOPS_DATA_TYPE_H

namespace rkeops {

// data type 
#if USE_DOUBLE
using type_t = double;
#else
using type_t = float;
#endif

}

#endif // RKEOPS_DATA_TYPE_H