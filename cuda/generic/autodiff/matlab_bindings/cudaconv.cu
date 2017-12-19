
// see compile_mex file for compiling

#include <mex.h>
#include "core/GpuConv2D.cu"
#include "core/autodiff.h"

// FORMULA and __TYPE__ are supposed to be set via #define macros in the compilation command

using FUNC = Generic<FORMULA>;
using VARSI = FUNC::sEval::VARSI;
using VARSJ = FUNC::sEval::VARSJ;
using DIMSX = FUNC::sEval::DIMSX;
using DIMSY = FUNC::sEval::DIMSY;

const int NARGSI = DIMSX::SIZE;
const int NARGSJ = DIMSY::SIZE;
const int NARGS = NARGSI+NARGSJ;
const int DIMPARAM = FUNC::sEval::DIMPARAM;

using INDSI = FUNC::sEval::INDSI;
using INDSJ = FUNC::sEval::INDSJ;

using INDS = FUNC::sEval::INDS;



//////////////////////////////////////////////////////////////////
///////////////// MEX ENTRY POINT ////////////////////////////////
//////////////////////////////////////////////////////////////////




template < int TAG >
void mexFunction_template( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    //plhs: double *gamma
    //prhs: double *params, double *arg0, double *arg1, ...

    //////////////////////////////////////////////////////////////
    // Input arguments
    //////////////////////////////////////////////////////////////

    //----- the first input argument: params--------------//
    int argu = 0;
    /*  create a pointer to the input vector */
    double *params = mxGetPr(prhs[argu]);
    /*  get the dimensions of the input targets */
    int mp = mxGetM(prhs[argu]); //nrows
    int np = mxGetN(prhs[argu]); //ncols
    /* check to make sure the array is 1D */
    if( min(mp,np)!=1 )
        mexErrMsgTxt("Input params must be a 1D array.");
    np = max(mp,np);
    if(np!=DIMPARAM)
        mexErrMsgTxt("wrong dimension for input");

    int typeargs[NARGS];
    using CATSI = GetCats<VARSI>;
    for(int k=0; k<NARGSI; k++)
        typeargs[INDSI::VAL(k)] = CATSI::VAL(k);
    using CATSJ = GetCats<VARSJ>;
    for(int k=0; k<NARGSJ; k++)
        typeargs[INDSJ::VAL(k)] = CATSJ::VAL(k);

    //----- the next input arguments: args--------------//
    int n[2] = {-1,-1}; // n[0] will be nx, n[1] will be ny;
    argu++;
    /*  create pointers to the input vectors */
    double **args = new double*[NARGS];    
    for(int k=0; k<NARGS; k++)
    {
    	/*  input sources */
    	args[k] = mxGetPr(prhs[argu+k]);
    	int dimk = mxGetM(prhs[argu+k]);
    	// we check dimension here from the formula
            mexErrMsgTxt("wrong dimension for input");
        // we get/check nx and ny here from the formula
    	int nk = mxGetN(prhs[argu+k]);
        int typek = typeargs[k];
        if(n[typek]==-1)
            n[typek] = typeargs[k];
            else if(n[typek]!=nk)
                mexErrMsgTxt("inconsistent input sizes");
    }
    argu += NARGS;

    //////////////////////////////////////////////////////////////
    // Output arguments
    //////////////////////////////////////////////////////////////

    /*  set the output pointer to the output result(vector) */
    int dimout = FORMULA::DIM;
    int nout = n[TAG];
    plhs[0] = mxCreateDoubleMatrix(dimout,nout,mxREAL);

    /*  create a C pointer to a copy of the output result(vector)*/
    double *gamma = mxGetPr(plhs[0]);

    //////////////////////////////////////////////////////////////
    // Call Cuda codes
    //////////////////////////////////////////////////////////////
    
    GpuConv2D(FUNC::sEval(), params, n[TAG], n[1-TAG], gamma, args);

}




/* the gateway function */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    // register an exit function to prevent crash at matlab exit or recompiling
    //mexAtExit(ExitFcn);

    /*  check for proper number of arguments */
    if(nrhs < 1+NARGS || nrhs>2+NARGS) // params, args..., tag with tag optional
        mexErrMsgTxt("Wrong number of inputs.");
    if(nlhs != 1) 
        mexErrMsgTxt("One output required.");

    int tag;
    // last optional argument is tag (default 0 if summation over j, 1 if summation over i)
    if(nrhs==2+NARGS)
    {
        int argu = 1+NARGS;
        /*  create a pointer to the input vector */
        double *tag_f = mxGetPr(prhs[argu]);
        /*  get the dimensions of the input targets */
        int mp = mxGetM(prhs[argu]); //nrows
        int np = mxGetN(prhs[argu]); //ncols
        if(mp!=1||np!=1||*tag_f!=0||*tag_f!=1)
            mexErrMsgTxt("last input tag should be a scalar value : 0 or 1");
        if(mp!=1||np!=1)
            mexErrMsgTxt("last input tag should be a scalar value : 0 or 1");
        tag = (int)*tag_f;
    }
    else
        tag = 0;

    if(tag==0)
   	mexFunction_template<0>(nlhs,plhs,nrhs,prhs);
    else
        mexFunction_template<1>(nlhs,plhs,nrhs,prhs);

}
