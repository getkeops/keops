#include <mex.h>

#include "formula.h"
#include "core/Pack.h"

extern "C" int CpuConv(int, int, __TYPE__*, __TYPE__**);
extern "C" int CpuTransConv(int, int, __TYPE__*, __TYPE__**);

#ifdef USE_CUDA
extern "C" int GpuConv1D(int, int, __TYPE__*, __TYPE__**);
extern "C" int GpuConv1D_FromDevice(int, int, __TYPE__*, __TYPE__**);
extern "C" int GpuConv2D(int, int, __TYPE__*, __TYPE__**);
extern "C" int GpuConv2D_FromDevice(int, int, __TYPE__*, __TYPE__**);
extern "C" int GpuTransConv1D(int, int, __TYPE__*, __TYPE__**);
extern "C" int GpuTransConv1D_FromDevice(int, int, __TYPE__*, __TYPE__**);
extern "C" int GpuTransConv2D(int, int, __TYPE__*, __TYPE__**);
extern "C" int GpuTransConv2D_FromDevice(int, int, __TYPE__*, __TYPE__**);
#endif

using namespace keops;

void ExitFcn(void) {
//#ifdef USE_CUDA
    //cudaDeviceReset();
//#endif
}

// uncomment the following block of code to redirect stdout to matlab console
/*
   class mystream : public std::streambuf
   {
   protected:
   virtual std::streamsize xsputn(const char *s, std::streamsize n) { mexPrintf("%.*s", n, s); return n; }
   virtual int overflow(int c=EOF) { if (c != EOF) { mexPrintf("%.1s", &c); } return 1; }
   };
   class scoped_redirect_cout
   {
   public:
   scoped_redirect_cout() { old_buf = std::cout.rdbuf(); std::cout.rdbuf(&mout); }
   ~scoped_redirect_cout() { std::cout.rdbuf(old_buf); }
   private:
   mystream mout;
   std::streambuf *old_buf;
   };
   static scoped_redirect_cout mycout_redirect;
   */

//////////////////////////////////////////////////////////////////
///////////////// MEX ENTRY POINT ////////////////////////////////
//////////////////////////////////////////////////////////////////

/* the gateway function */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // register an exit function to prevent crash at matlab exit or recompiling
    mexAtExit(ExitFcn);

    using VARSI = typename F::template VARS<0>;	// list variables of type I used in formula F
    using VARSJ = typename F::template VARS<1>; // list variables of type J used in formula F
    using VARSP = typename F::template VARS<2>; // list variables of type parameter used in formula F

    using DIMSX = GetDims<VARSI>;
    using DIMSY = GetDims<VARSJ>;
    using DIMSP = GetDims<VARSP>;

    using INDSI = GetInds<VARSI>;
    using INDSJ = GetInds<VARSJ>;
    using INDSP = GetInds<VARSP>;

    using INDS = ConcatPacks<ConcatPacks<INDSI,INDSJ>,INDSP>;

    const int NARGSI = VARSI::SIZE; // number of I variables used in formula F
    const int NARGSJ = VARSJ::SIZE; // number of J variables used in formula F
    const int NARGSP = VARSP::SIZE; // number of parameters variables used in formula F

    int NARGS = nrhs-5;

    if(nlhs != 1)
        mexErrMsgTxt("One output required.");


    //////////////////////////////////////////////////////////////
    // Helper function to cast mxArray (which is double by default) to __TYPE__
    //////////////////////////////////////////////////////////////

    auto castedFun = [] (double* double_ptr, const mxArray *dd)  -> __TYPE__* {
        /*  get the dimensions */
        int n = mxGetNumberOfElements(dd); //nrows
#if  USE_DOUBLE
        return double_ptr;
#else
        __TYPE__ *__type__ptr = new __TYPE__[n];
        std::copy(double_ptr,double_ptr+n,__type__ptr);

        return __type__ptr;
#endif
    };


    //////////////////////////////////////////////////////////////
    // Input arguments
    //////////////////////////////////////////////////////////////

    int argu = 0;
    int n[3];
    //----- the first input arguments: nx--------------//
    if(mxGetM(prhs[argu])!=1 || mxGetN(prhs[argu])!=1)
        mexErrMsgTxt("first arg should be scalar nx");
    n[0] = *mxGetPr(prhs[argu]);
    argu++;

    //----- the second input arguments: ny--------------//
    if(mxGetM(prhs[argu])!=1 || mxGetN(prhs[argu])!=1)
        mexErrMsgTxt("second arg should be scalar ny");
    n[1] = *mxGetPr(prhs[argu]);
    argu++;

    n[2] = 1;

    //----- the next input arguments: tagIJ--------------//
    if(mxGetM(prhs[argu])!=1 || mxGetN(prhs[argu])!=1)
        mexErrMsgTxt("third arg should be scalar tagIJ");
    int tagIJ = *mxGetPr(prhs[argu]);
    argu++;

    //----- the next input arguments: tagCpuGpu--------------//
    if(mxGetM(prhs[argu])!=1 || mxGetN(prhs[argu])!=1)
        mexErrMsgTxt("fourth arg should be scalar tagCpuGpu");
    int tagCpuGpu = *mxGetPr(prhs[argu]);
    argu++;

    //----- the next input arguments: tag1D2D--------------//
    if(mxGetM(prhs[argu])!=1 || mxGetN(prhs[argu])!=1)
        mexErrMsgTxt("fifth arg should be scalar tag1D2D");
    int tag1D2D = *mxGetPr(prhs[argu]);
    argu++;

    int *typeargs = new int[NARGS];
    int *dimargs = new int[NARGS];

    for(int k=0; k<NARGS; k++) {
        typeargs[k] = -1;
        dimargs[k] = -1;
    }
    for(int k=0; k<NARGSI; k++) {
        typeargs[INDSI::VAL(k)] = 0;
        dimargs[INDSI::VAL(k)] = DIMSX::VAL(k);
    }
    for(int k=0; k<NARGSJ; k++) {
        typeargs[INDSJ::VAL(k)] = 1;
        dimargs[INDSJ::VAL(k)] = DIMSY::VAL(k);
    }
    for(int k=0; k<NARGSP; k++) {
        typeargs[INDSP::VAL(k)] = 2;
        dimargs[INDSP::VAL(k)] = DIMSP::VAL(k);
    }

    //----- the next input arguments: args--------------//
    /*  create pointers to the input vectors */
    double **args = new double*[NARGS];
    __TYPE__ **castedargs = new __TYPE__*[NARGS];
    for(int k=0; k<NARGS; k++) {
        /*  input sources */
        args[k] = mxGetPr(prhs[argu+k]);
        castedargs[k] = castedFun(args[k],prhs[argu+k]);

        // checking dimensions
        if(dimargs[k]!=-1) { // we care only if the current variable is used in formula
            int dimk = mxGetM(prhs[argu+k]);
            // we check nx and ny here from the formula
            int nk = mxGetN(prhs[argu+k]);
            int typek = typeargs[k];
            // we check dimension here from the formula
            if(dimk!=dimargs[k]) {
                mexPrintf("For argument #%d : dimension (=number of columns) is %d but should be %d.\n",k,dimk,dimargs[k]);
                mexErrMsgTxt("Wrong dimension for input argument.");
            }
            if(n[typek]!=nk) {
                mexPrintf("For argument #%d : size (=number of rows) is %d but should be %d.\n",k,nk,n[typek]);
                mexErrMsgTxt("inconsistent input sizes");
            }
        }
    }
    argu += NARGS;


    //////////////////////////////////////////////////////////////
    // Output arguments
    //////////////////////////////////////////////////////////////

    /*  set the output pointer to the output result(vector) */
    int dimout = F::DIM;
    int nout = n[tagIJ];
    plhs[0] = mxCreateDoubleMatrix(dimout,nout,mxREAL);

    /*  create a C pointer to a copy of the output result(vector)*/
    double *gamma = mxGetPr(plhs[0]);
    __TYPE__*castedgamma = castedFun(gamma,plhs[0]);

    //////////////////////////////////////////////////////////////
    // Call Cuda codes
    //////////////////////////////////////////////////////////////

    // tagCpuGpu=0 means convolution on Cpu, tagCpuGpu=1 means convolution on Gpu, tagCpuGpu=2 means convolution on Gpu from device data
    // tagIJ=0 means sum over j, tagIJ=1 means sum over j
    // tag1D2D=0 means 1D Gpu scheme, tag1D2D=1 means 2D Gpu scheme

    if(tagCpuGpu==0) {
        if(tagIJ==0){
            CpuConv( n[0], n[1], castedgamma, castedargs);
        } else{
            CpuTransConv( n[0], n[1], castedgamma, castedargs);
        } 
    }
#ifdef USE_CUDA
    else if(tagCpuGpu==1) {
        if(tagIJ==0) {
            if(tag1D2D==0) {
                GpuConv1D( n[0], n[1], castedgamma, castedargs);
            } else {
                GpuConv2D( n[0], n[1], castedgamma, castedargs);
            }
        } else {
            if(tag1D2D==0){
                GpuTransConv1D( n[0], n[1], castedgamma, castedargs);
            } else {
                GpuTransConv2D( n[0], n[1], castedgamma, castedargs);
            }
        }
    } 
    else {
        if(tagIJ==0) {
            if(tag1D2D==0){
                GpuConv1D_FromDevice( n[0], n[1], castedgamma, castedargs);
            } else {
                GpuConv2D_FromDevice( n[0], n[1], castedgamma, castedargs);
            }
        } else {
            if(tag1D2D==0){
                GpuTransConv1D_FromDevice( n[0], n[1], castedgamma, castedargs);
            } else{
                GpuTransConv2D_FromDevice( n[0], n[1], castedgamma, castedargs);
            }
        }
    }
#endif

#if not USE_DOUBLE
    // copyt the casted results in double 
    int ngamma =mxGetNumberOfElements(plhs[0]);
    std::copy(castedgamma,castedgamma+ngamma,gamma);

    delete[] castedgamma;
    for(int k=0; k<NARGS; k++)
       delete[] castedargs[k];
    delete[] castedargs;
#endif

    delete[] args;
    delete[] typeargs;
    delete[] dimargs;

}
