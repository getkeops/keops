#include <mex.h>

// #include "formula.h" made in cmake
#include "core/pack/GetInds.h"
#include "core/pack/GetDims.h"
#include "core/pack/ConcatPack.h"

extern "C" int CpuReduc(int, int, __TYPE__*, __TYPE__**);

#if USE_CUDA 
extern "C" int GpuReduc1D_FromHost(int, int, __TYPE__*, __TYPE__**, int);
extern "C" int GpuReduc1D_FromDevice(int, int, __TYPE__*, __TYPE__**, int);
extern "C" int GpuReduc2D_FromHost(int, int, __TYPE__*, __TYPE__**, int);
extern "C" int GpuReduc2D_FromDevice(int, int, __TYPE__*, __TYPE__**, int);
#endif

using namespace keops;

void ExitFcn(void) {
//#ifdef USE_CUDA && USE_CUDA==1
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
    
    using FF = F::F; // F::F is formula inside reduction (ex if F is Sum_Reduction<Form> then F::F is Form)

    using VARSI = typename FF::template VARS<0>;    // list variables of type I used in formula F
    using VARSJ = typename FF::template VARS<1>;    // list variables of type J used in formula F
    using VARSP = typename FF::template VARS<2>;    // list variables of type parameter used in formula F

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

    // minimal number of input arrays required for the formula to be evaluated :
    const int NMINARGS = F::NMINARGS;	

    if(nlhs != 1)
        mexErrMsgTxt("One output required.");

// in case function is called without any input, we output an array containing information about the formula. Currently only the minimal number of arguments is returned
if(nrhs==0)
{
plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
double *info = mxGetPr(plhs[0]);
info[0] = NMINARGS;
}
else
{
    // number of input arrays of the matlab function. 
    // The "-5" is because there are 4 parameter inputs before the list of arrays : nx, ny, tagCpuGpu, tagID2D, device_id
    int nargs = nrhs-5;	

    if(nargs<NMINARGS)
        mexErrMsgTxt("Formula requires more input arrays to be evaluated");

    const int tagIJ = F::tagI;



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

    //----- the next input arguments: tagCpuGpu--------------//
    if(mxGetM(prhs[argu])!=1 || mxGetN(prhs[argu])!=1)
        mexErrMsgTxt("third arg should be scalar tagCpuGpu");
    int tagCpuGpu = *mxGetPr(prhs[argu]);
    argu++;

    //----- the next input arguments: tag1D2D--------------//
    if(mxGetM(prhs[argu])!=1 || mxGetN(prhs[argu])!=1)
        mexErrMsgTxt("fourth arg should be scalar tag1D2D");
    int tag1D2D = *mxGetPr(prhs[argu]);
    argu++;

    //----- the next input arguments: device_id--------------//
    if(mxGetM(prhs[argu])!=1 || mxGetN(prhs[argu])!=1)
        mexErrMsgTxt("fifth arg should be scalar device_id");
    int device_id = *mxGetPr(prhs[argu]);
    argu++;

    int *typeargs = new int[NMINARGS];
    int *dimargs = new int[NMINARGS];

    for(int k=0; k<NMINARGS; k++) {
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
    double **args = new double*[NMINARGS];
    __TYPE__ **castedargs = new __TYPE__*[NMINARGS];
    for(int k=0; k<NMINARGS; k++) {
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
    argu += NMINARGS;


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

    // tagIJ=0 means reduction over j, tagIJ=1 means sum over j
    // tagCpuGpu=0 means reduction on Cpu, tagCpuGpu=1 means reduction on Gpu from host data, tagCpuGpu=2 means reduction on Gpu from device data
    // tag1D2D=0 means 1D Gpu scheme, tag1D2D=1 means 2D Gpu scheme

#if USE_CUDA
    if(tagCpuGpu==0) 
        CpuReduc( n[0], n[1], castedgamma, castedargs);
    else if(tagCpuGpu==1) 
        if(tag1D2D==0)
            GpuReduc1D_FromHost( n[0], n[1], castedgamma, castedargs, device_id);
        else
            GpuReduc2D_FromHost( n[0], n[1], castedgamma, castedargs, device_id);
    else if(tagCpuGpu==2)
        if(tag1D2D==0)
            GpuReduc1D_FromDevice( n[0], n[1], castedgamma, castedargs, device_id);
        else
            GpuReduc2D_FromDevice( n[0], n[1], castedgamma, castedargs, device_id);
#else
    if(tagCpuGpu != 0)
        mexWarnMsgTxt("CPU Routine are used. To suppress this warning set tagCpuGpu to 0.");
    CpuReduc( n[0], n[1], castedgamma, castedargs);
#endif

#if not USE_DOUBLE
    // copy the casted results in double 
    int ngamma =mxGetNumberOfElements(plhs[0]);
    std::copy(castedgamma,castedgamma+ngamma,gamma);

    delete[] castedgamma;
    for(int k=0; k<NMINARGS; k++)
       delete[] castedargs[k];
    delete[] castedargs;
#endif

    delete[] args;
    delete[] typeargs;
    delete[] dimargs;
}
}
