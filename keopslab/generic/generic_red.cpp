#include <mex.h>

// #include "formula.h" made in cmake
#include "binders/utils.h"
#include "binders/checks.h"
#include "binders/switch.h"

using namespace keops;

template <>
size_t keops_binders::get_ndim(const mxArray &pm) {
  return mxGetNumberOfDimensions(&pm);
}

template <>
size_t keops_binders::get_size(const mxArray &pm, size_t l) {
  const mwSize *d = mxGetDimensions(&pm);
  return d[l];
}

template <>
mxArray *keops_binders::allocate_result_array(const size_t *dimout, const size_t a) {
  return mxCreateNumericArray((int) 2, dimout, mxDOUBLE_CLASS, mxREAL);
//  return mxCreateDoubleMatrix(dimout[0], dimout[1], mxREAL);
}

template <>
mxArray *keops_binders::allocate_result_array_gpu(const size_t *dimout, const size_t a) {
  mexErrMsgTxt("[keOpsLab] does not yet support array on GPU.");
}

template < typename _T >
_T *keops_binders::get_data(const mxArray &pm) {
  return mxGetPr(&pm);
}

void keops_binders::keops_error(std::string msg) {
  mexErrMsgTxt(msg.c_str());
}

template <>
bool keops_binders::is_contiguous(const mxArray &pm) {
  return true;
};


////////////////////////////////////////////////////////////////////////////////
// Helper function to cast mxArray (which is double by default) to __TYPE__   //
////////////////////////////////////////////////////////////////////////////////

template < typename input_T, typename output_T >
output_T *castedFun(input_T *double_ptr, const mxArray *dd) {
  /*  get the dimensions */
  int n = mxGetNumberOfElements(dd);
#if  USE_DOUBLE
  return double_ptr;
#else
  output_T *__type__ptr = new output_T[n];
  std::copy(double_ptr, double_ptr + n, __type__ptr);

  return __type__ptr;
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

  //----- the first input arguments: nx--------------//
  if (mxGetM(prhs[argu]) != 1 || mxGetN(prhs[argu]) != 1)
    mexErrMsgTxt("[KeOps]: first arg should be scalar nx");
  int nx = *mxGetPr(prhs[argu]);
  argu++;

  //----- the second input arguments: ny--------------//
  if (mxGetM(prhs[argu]) != 1 || mxGetN(prhs[argu]) != 1)
    mexErrMsgTxt("[KeOps]: second arg should be scalar ny");
  int ny = *mxGetPr(prhs[argu]);
  argu++;

  //----- the next input arguments: tagCpuGpu--------------//
  if (mxGetM(prhs[argu]) != 1 || mxGetN(prhs[argu]) != 1)
    mexErrMsgTxt("[KeOps]: third arg should be scalar tagCpuGpu");
  int tagCpuGpu = *mxGetPr(prhs[argu]);
  argu++;

  //----- the next input arguments: tag1D2D--------------//
  if (mxGetM(prhs[argu]) != 1 || mxGetN(prhs[argu]) != 1)
    mexErrMsgTxt("[KeOps]: fourth arg should be scalar tag1D2D");
  int tag1D2D = *mxGetPr(prhs[argu]);
  argu++;

  //----- GpuArray are not currently supported--------------//
  int tagHostDevice = 0;

  keops_binders::check_tag(tag1D2D, "1D2D");
  keops_binders::check_tag(tagCpuGpu, "CpuGpu");
  keops_binders::check_tag(tagHostDevice, "HostDevice");

  //----- the next input arguments: device_id--------------//
  if (mxGetM(prhs[argu]) != 1 || mxGetN(prhs[argu]) != 1)
    mexErrMsgTxt("[KeOps]: fifth arg should be scalar device_id");
  int device_id = *mxGetPr(prhs[argu]);
  argu++;

  short int Device_Id_s = keops_binders::cast_Device_Id(device_id);

  //----- the next input arguments: args--------------//
  //  create pointers to the input vectors
  double *args[keops::NARGS];
  __TYPE__ *castedargs[keops::NARGS];

  for (int k = 0; k < keops::NARGS; k++) {
    //  input sources
    args[k] = keops_binders::get_data< double >(*prhs[argu + k]);
    castedargs[k] = castedFun< double, __TYPE__ >(args[k], prhs[argu + k]);
  }

  // number of input arrays of the matlab function.
  // The "-5" is because there are 4 parameter inputs before the list
  // of arrays : nx, ny, tagCpuGpu, tagID2D, device_id
  size_t nargs = nrhs - 5;
  keops_binders::check_narg(nargs); // TODO: put check_args

//////////////////////////////////////////////////////////////
// Output arguments
//////////////////////////////////////////////////////////////

  // set the output pointer to the output result(vector)
  plhs[0] = keops_binders::create_result_array< mxArray* >(nx, ny);

  //create a C pointer to a copy of the output result(vector)
  double *gamma = mxGetPr(plhs[0]);
  __TYPE__ *castedgamma = castedFun< double, __TYPE__ >(gamma, plhs[0]);

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
  for(int k=0; k<NARGS; k++)
    delete[] castedargs[k];
#endif

}


