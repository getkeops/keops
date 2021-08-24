#pragma once

/**************************************************************/
/*  This is the generic header to be included in the binders  */
/*  in order to call keops routines                           */
/**************************************************************/

#define __INDEX__ int32_t // use int instead of double

#if USE_HALF && USE_CUDA
  #include <cuda_fp16.h>
#endif

#include "utils.h"
#include "checks.h"
#include "switch.h"
