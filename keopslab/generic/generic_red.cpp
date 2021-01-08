#include <mex.h>

// #include "formula.h" made in cmake

// keops_binders import
#include "binders/include.h"

using namespace keops;

template<>
int keops_binders::get_ndim< const mxArray* >(const mxArray* pm) {
  return mxGetNumberOfDimensions(pm);
}

template<>
int keops_binders::get_size< const mxArray* >(const mxArray* pm, int l) {
  const mwSize *d = mxGetDimensions(pm);
  return d[l];
}

template<>
int keops_binders::get_size< mxArray* >(mxArray* pm, int l) {
  const mwSize *d = mxGetDimensions(pm);
  return d[l];
}

template<>
double* keops_binders::get_data< mxArray*, double >(mxArray* pm) {
  return static_cast< double* >(mxGetData(pm));
}

template<>
float* keops_binders::get_data< mxArray*, float >(mxArray* pm) {
  return static_cast< float* >(mxGetData(pm));
}

template<>
double* keops_binders::get_data< const mxArray*, double >(const mxArray* pm) {
  return static_cast< double* >(mxGetData(pm));
}

template<>
float* keops_binders::get_data< const mxArray*, float >(const mxArray* pm) {
  return static_cast< float* >(mxGetData(pm));
}

template<>
__INDEX__* keops_binders::get_rangedata(mxArray* pm) {
  return static_cast< __INDEX__ * >(mxGetData(pm));
}

template<>
__INDEX__* keops_binders::get_rangedata(const mxArray* pm) {
  return static_cast< __INDEX__ * >(mxGetData(pm));
}

template<>
mxArray* keops_binders::allocate_result_array< mxArray*, double >(int* dimout, int nbatchdims) {
  // mxArray constructor only accepts "size_t" to specify the shape of a new tensor:
  int ndimout = nbatchdims + 2;
  size_t dimout_size_t[ndimout];
  std::copy(dimout, dimout + ndimout, dimout_size_t);
  
  return mxCreateNumericArray(ndimout, dimout_size_t, mxDOUBLE_CLASS, mxREAL);
}

template<>
mxArray* keops_binders::allocate_result_array< mxArray*, float >(int* dimout, int nbatchdims) {
  // mxArray constructor only accepts "size_t" to specify the shape of a new tensor:
  int ndimout = nbatchdims + 2;
  size_t dimout_size_t[ndimout];
  std::copy(dimout, dimout + ndimout, dimout_size_t);
  
  return mxCreateNumericArray(ndimout, dimout_size_t, mxSINGLE_CLASS, mxREAL);
}

template<>
mxArray*
keops_binders::allocate_result_array_gpu< mxArray*, __TYPE__ >(int* dimout, int nbatchdims, short int device_id) {
  mexErrMsgTxt("[keOpsLab] does not yet support array on GPU.");
  throw std::runtime_error("[KeOps] this line is just here to avoid a warning.");
}

void keops_binders::keops_error(std::string msg) {
  mexErrMsgTxt(msg.c_str());
}

template<>
bool keops_binders::is_contiguous(const mxArray* pm) {
  return true;
};


////////////////////////////////////////////////////////////////////////////////
// Helper functions to cast mxArray                                           //
////////////////////////////////////////////////////////////////////////////////

const mxArray* castedFun(const mxArray* dd) {

#if  USE_DOUBLE
  //return keops_binders::get_data< double >(dd);
  return dd;
#else
  mxArray* df = mxCreateNumericArray(mxGetNumberOfDimensions(dd), mxGetDimensions(dd), mxSINGLE_CLASS, mxREAL);
  float* float_ptr = (float*) mxGetData(df);
  
  double* double_ptr = (double*) mxGetData(dd);
  int n = mxGetNumberOfElements(dd);
  std::copy(double_ptr, double_ptr + n, float_ptr);
  
  return df;
#endif
};

mxArray* icastedFun(mxArray* df) {

#if  USE_DOUBLE
  //return keops_binders::get_data< double >(dd);
  return dd;
#else
  mxArray* dd = mxCreateNumericArray(mxGetNumberOfDimensions(df), mxGetDimensions(df), mxDOUBLE_CLASS, mxREAL);
  double* double_ptr = (double*) mxGetData(dd);
  
  float* float_ptr = (float*) mxGetData(df);
  int n = mxGetNumberOfElements(df);
  std::copy(float_ptr, float_ptr + n, double_ptr);
  
  return dd;
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
  
  
  argu++;
  //----- the next input arguments: tag1D2D--------------//
  if (mxGetM(prhs[argu]) != 1 || mxGetN(prhs[argu]) != 1)
    mexErrMsgTxt("[KeOps]: fourth arg should be scalar tag1D2D");
  int tag1D2D = *mxGetPr(prhs[argu]);

  
  //----- GpuArray are not currently supported--------------//
  int tagHostDevice = 0;
  
  argu++;
  //----- the next input arguments: device_id--------------//
  if (mxGetM(prhs[argu]) != 1 || mxGetN(prhs[argu]) != 1)
    mexErrMsgTxt("[KeOps]: fifth arg should be scalar device_id");
  int Device_Id_s = *mxGetPr(prhs[argu]);
  
  argu++;
  //----- the next input arguments: device_id--------------//
  if (mxGetM(prhs[argu]) != 1 || mxGetN(prhs[argu]) != 1)
    mexErrMsgTxt("[KeOps]: sixth arg should be scalar nx");
  int nx = *mxGetPr(prhs[argu]);
  
  argu++;
  //----- the next input arguments: device_id--------------//
  if (mxGetM(prhs[argu]) != 1 || mxGetN(prhs[argu]) != 1)
    mexErrMsgTxt("[KeOps]: seventh arg should be scalar ny");
  int ny = *mxGetPr(prhs[argu]);
  
  argu++;
  //----- the next input arguments: args--------------//
  //  create pointers to the input vectors
  const mxArray *castedargs[keops::NARGS];
  for (int k = 0; k < nrhs - 5; k++) {
    castedargs[k] = castedFun(prhs[argu + k]);
  }
  
//////////////////////////////////////////////////////////////
// Call Cuda codes                                          //
//////////////////////////////////////////////////////////////
  
  plhs[0] = icastedFun(keops_binders::launch_keops< const mxArray*, mxArray*, mxArray* >(
          tag1D2D,
          tagCpuGpu,
          tagHostDevice,
          Device_Id_s,
		  nx,
		  ny,
          nrhs - 5,
          castedargs));
}


