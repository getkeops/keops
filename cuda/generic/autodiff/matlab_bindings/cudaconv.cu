#include <mex.h>
#include "link_autodiff.cu"

// FORMULA and __TYPE__ are supposed to be set via #define macros in the compilation command

using FUN = Generic<FORMULA>;
using VARSI = FUN::VARSI;
using VARSJ = FUN::VARSJ;
using DIMSX = FUN::DIMSX;
using DIMSY = FUN::DIMSY;

using NARGSI = VARSI::SIZE;
using NARGSJ = VARSJ::SIZE;
using NARGS = NARGSI+NARGSJ;

using INDSI = FUN::INDSI;
using INDSJ = FUN::INDSJ;

using INDS = FUN::INDS;



//////////////////////////////////////////////////////////////////
///////////////// MEX ENTRY POINT ////////////////////////////////
//////////////////////////////////////////////////////////////////


/* the gateway function */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    //plhs: double *gamma
    //prhs: double *params, double *arg0, double *arg1, ..., int tag

    // register an exit function to prevent crash at matlab exit or recompiling
    mexAtExit(ExitFcn);

    /*  check for proper number of arguments */
    if(nrhs != 2+NARGS) // params, args..., tag
        mexErrMsgTxt("Wrong number of inputs.");
    if(nlhs != 1) 
        mexErrMsgTxt("One output required.");

    //////////////////////////////////////////////////////////////
    // Input arguments
    //////////////////////////////////////////////////////////////

    //----- the first input argument: params--------------//
    argu++;
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
    for(int k=0; k<NARGSI; k++)
        typeargs[VARSI::val(k)::N] = VARSI::val(k)::CAT;
    for(int k=0; k<NARGSJ; k++)
        typeargs[VARSJ::val(k)::N] = VARSJ::val(k)::CAT;

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
	if(dimk!=DIMS::val(k))
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

    // last argument is tag (0 if summation over j, 1 if summation over i)
    /*  create a pointer to the input vector */
    double *tag = mxGetPr(prhs[argu]);
    /*  get the dimensions of the input targets */
    int mp = mxGetM(prhs[argu]); //nrows
    int np = mxGetN(prhs[argu]); //ncols

    //////////////////////////////////////////////////////////////
    // Output arguments
    //////////////////////////////////////////////////////////////

    /*  set the output pointer to the output result(vector) */
    int dimout = DIMOUT
    int nout = ?nx:ny;
    plhs[0] = mxCreateDoubleMatrix(dimout,nout,mxREAL);

    /*  create a C pointer to a copy of the output result(vector)*/
    double *gamma = mxGetPr(plhs[0]);

    //////////////////////////////////////////////////////////////
    // Call Cuda codes
    //////////////////////////////////////////////////////////////
    
    GpuConv2D(FUN::sEval(), params, nx, ny, gamma, args);


    return;

}
