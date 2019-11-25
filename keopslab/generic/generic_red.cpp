#include <mex.h>

// #include "formula.h" made in cmake
#include "binders/utils.h"
#include "binders/checks.h"
#include "binders/switch.h"


using namespace keops;

template<>
size_t keops_binders::get_ndim(const mxArray &pm) {
  return mxGetNumberOfDimensions(&pm);
}

template<>
size_t keops_binders::get_size(const mxArray &pm, size_t l) {
  const mwSize *d = mxGetDimensions(&pm);
  return d[l];
}

template<>
mxArray *keops_binders::allocate_result_array(const size_t *dimout, const size_t a) {
  return mxCreateNumericArray((int) 2, dimout, mxDOUBLE_CLASS, mxREAL);
//  return mxCreateDoubleMatrix(dimout[0], dimout[1], mxREAL);
}

template<>
mxArray* keops_binders::allocate_result_array_gpu(const size_t *dimout, const size_t a) {
  mexErrMsgTxt("[keOpsLab] does not yet support array on GPU.");
}

template< typename _T >
_T* keops_binders::get_data(const mxArray &pm) {
  return mxGetPr(&pm);
}

void keops_binders::keops_error(std::string msg) {
  mexErrMsgTxt(msg.c_str());
}

template<>
bool keops_binders::is_contiguous(const mxArray &pm) {
  return true;
};


////////////////////////////////////////////////////////////////////////////////
// Helper function to cast mxArray (which is double by default) to __TYPE__   //
////////////////////////////////////////////////////////////////////////////////

template< typename output_T >
output_T* castedFun(const mxArray *dd) {
  /*  get the dimensions */
#if  USE_DOUBLE
  // return keops_binders::get_data< double >(*dd);
  return mxGetPr(dd);
#else
  int n = mxGetNumberOfElements(dd);
  // double *double_ptr = keops_binders::get_data<const mxArray &, double >(*dd);
  double *double_ptr = mxGetPr(dd);
  
  output_T *float_ptr = new output_T[n];
  std::copy(double_ptr, double_ptr + n, float_ptr);
  
  return float_ptr;
#endif
};

void ExitFcn(void) {
//#ifdef USE_CUDA && USE_CUDA==1
  //cudaDeviceReset();
//#endif
}



//////////////////////////////////////////////////////////////////
///////////////// MEX ENTRY POINT ////////////////////////////////
//////////////////////////////////////////////////////////////////

/* the gateway function */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  
  // register an exit function to prevent crash at matlab exit or recompiling
  mexAtExit(ExitFcn);
  
  if (nlhs != 1)
    mexErrMsgTxt("One output required.");
  
  // in case function is called without any input, we output an array containing information
  // about the formula. Currently only the minimal number of arguments is returned
  if (nrhs == 0) {
    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    double *info = mxGetPr(plhs[0]);
    info[0] = keops::NARGS;
    return;
  }
  
  //////////////////////////////////////////////////////////////
  // Input arguments                                          //
  //////////////////////////////////////////////////////////////
  
  int argu = 0;
  
  //----- the next input arguments: tagCpuGpu--------------//
  if (mxGetM(prhs[argu]) != 1 || mxGetN(prhs[argu]) != 1)
    mexErrMsgTxt("[KeOps]: third arg should be scalar tagCpuGpu");
  int tagCpuGpu = *mxGetPr(prhs[argu]);
  keops_binders::check_tag(tagCpuGpu, "CpuGpu");
  
  
  argu++;
  //----- the next input arguments: tag1D2D--------------//
  if (mxGetM(prhs[argu]) != 1 || mxGetN(prhs[argu]) != 1)
    mexErrMsgTxt("[KeOps]: fourth arg should be scalar tag1D2D");
  int tag1D2D = *mxGetPr(prhs[argu]);
  keops_binders::check_tag(tag1D2D, "1D2D");
  
  //----- GpuArray are not currently supported--------------//
  int tagHostDevice = 0;
  keops_binders::check_tag(tagHostDevice, "HostDevice");
  
  argu++;
  //----- the next input arguments: device_id--------------//
  if (mxGetM(prhs[argu]) != 1 || mxGetN(prhs[argu]) != 1)
    mexErrMsgTxt("[KeOps]: fifth arg should be scalar device_id");
  short int Device_Id_s = keops_binders::cast_Device_Id(*mxGetPr(prhs[argu]));
  
  argu++;
  //----- the next input arguments: args--------------//
  //  create pointers to the input vectors
  const mxArray *args[keops::NARGS];
  __TYPE__ *castedargs[keops::NARGS];
  
  printf("---- %d\n", keops::NARGS);
  printf("---- %d\n", nrhs - 3);
  for (int k = 0; k < nrhs - 3; k++) {
    //  input sources
    args[k] = prhs[argu + k];
    castedargs[k] = castedFun< __TYPE__ >(prhs[argu + k]);
  }

  // number of input arrays of the matlab function.
  // The "-3" is because there are 3 parameter inputs before the list
  // of arrays : tagCpuGpu, tagID2D, device_id
  std::tuple< int, int, int, int * > sizes = keops_binders::check_ranges(nrhs - 3, args);
  
  int nx = std::get< 0 >(sizes);
  int ny = std::get< 1 >(sizes);

//////////////////////////////////////////////////////////////
// Output arguments
//////////////////////////////////////////////////////////////
  
  // set the output pointer to the output result(vector)
  plhs[0] = keops_binders::create_result_array< mxArray * >(nx, ny);
  
  //create a C pointer to a copy of the output result(vector)
  double *gamma = mxGetPr(plhs[0]);
  __TYPE__ *castedgamma = castedFun< __TYPE__ >(plhs[0]);

//////////////////////////////////////////////////////////////
// Call Cuda codes
//////////////////////////////////////////////////////////////
  
  keops_binders::launch_keops(tag1D2D, tagCpuGpu, tagHostDevice,
                              Device_Id_s,
                              nx, ny,
                              castedgamma,
                              castedargs);

#if not USE_DOUBLE
  // copy the casted results in double
  int ngamma = mxGetNumberOfElements(plhs[0]);
  std::copy(castedgamma, castedgamma + ngamma, gamma);

  delete[] castedgamma;
  for(int k=0; k<nrhs-3; k++)
    delete[] castedargs[k];
#endif

}


