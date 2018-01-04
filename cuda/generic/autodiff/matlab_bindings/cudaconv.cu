
// see compile_mex file for compiling

#include <mex.h>
#include "core/GpuConv2D.cu"
#include "core/autodiff.h"
#include "core/newsyntax.h"

// FORMULA and __TYPE__ are supposed to be set via #define macros in the compilation command

using F = decltype(FORMULA);

void ExitFcn(void) {
    cudaDeviceReset();
}

//////////////////////////////////////////////////////////////////
///////////////// MEX ENTRY POINT ////////////////////////////////
//////////////////////////////////////////////////////////////////


/* the gateway function */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    // register an exit function to prevent crash at matlab exit or recompiling
    mexAtExit(ExitFcn);
    
    const int TAG = 0; // only summation over j index...

	using VARSI = typename F::template VARS<TAG>;
	using VARSJ = typename F::template VARS<1-TAG>;
	
	using DIMSX = GetDims<VARSI>;
	using DIMSY = GetDims<VARSJ>;
	
    using PARAMS = typename F::VARS<2>;
    static const int DIMPARAM = PARAMS::SIZE;

	using INDSI = GetInds<VARSI>;
	using INDSJ = GetInds<VARSJ>;

	using INDS = ConcatPacks<INDSI,INDSJ>;
	
	const int NARGSI = VARSI::SIZE;
	const int NARGSJ = VARSJ::SIZE;
	const int NARGS = NARGSI+NARGSJ;

	
    
    /*  check for proper number of arguments */
    if(nrhs != (DIMPARAM?1:0)+NARGS) // args..., params or args... if no parameter in formula
        mexErrMsgTxt("Wrong number of inputs.");
    if(nlhs != 1) 
        mexErrMsgTxt("One output required.");


    //////////////////////////////////////////////////////////////
    // Input arguments
    //////////////////////////////////////////////////////////////


    int typeargs[NARGS], dimargs[NARGS];
    for(int k=0; k<NARGSI; k++)
    {
        typeargs[INDSI::VAL(k)] = TAG;
        dimargs[INDSI::VAL(k)] = DIMSX::VAL(k);
    }
    for(int k=0; k<NARGSJ; k++)
    {
        typeargs[INDSJ::VAL(k)] = 1-TAG;
        dimargs[INDSJ::VAL(k)] = DIMSY::VAL(k);
    }

    int argu = 0;
    //----- the first input arguments: args--------------//
    int n[2] = {-1,-1}; // n[0] will be nx, n[1] will be ny;
    /*  create pointers to the input vectors */
    double **args = new double*[NARGS];    
    for(int k=0; k<NARGS; k++)
    {
    	/*  input sources */
    	args[k] = mxGetPr(prhs[argu+k]);
    	int dimk = mxGetM(prhs[argu+k]);
        // we get/check nx and ny here from the formula
    	int nk = mxGetN(prhs[argu+k]);
        int typek = typeargs[k];
        cout << "k=" << k << endl;
        cout << "nk=" << nk << endl;
        cout << "n[typek]=" << n[typek] << endl;
        cout << "dimk=" << dimk << endl;
        cout << "dimargs[k]=" << dimargs[k] << endl;
    	// we check dimension here from the formula
    	if(dimk!=dimargs[k])
            mexErrMsgTxt("wrong dimension for input");
        if(n[typek]==-1)
            n[typek] = nk;
            else if(n[typek]!=nk)
            {
                mexErrMsgTxt("inconsistent input sizes");
            }
    }

	double *params;
	if(DIMPARAM) {
		//----- the last input argument: params--------------//
		argu++;
		/*  create a pointer to the input vector */
		params = mxGetPr(prhs[argu]);
		/*  get the dimensions of the input targets */
		int mp = mxGetM(prhs[argu]); //nrows
		int np = mxGetN(prhs[argu]); //ncols
		/* check to make sure the array is 1D */
		if( min(mp,np)!=1 )
			mexErrMsgTxt("Input params must be a 1D array.");
		np = max(mp,np);
		if(np!=DIMPARAM)
			mexErrMsgTxt("wrong dimension for input");
	}
	
    //////////////////////////////////////////////////////////////
    // Output arguments
    //////////////////////////////////////////////////////////////

    /*  set the output pointer to the output result(vector) */
    int dimout = F::DIM;
    int nout = n[TAG];
    plhs[0] = mxCreateDoubleMatrix(dimout,nout,mxREAL);

    /*  create a C pointer to a copy of the output result(vector)*/
    double *gamma = mxGetPr(plhs[0]);

    //////////////////////////////////////////////////////////////
    // Call Cuda codes
    //////////////////////////////////////////////////////////////
    
    GpuConv2D(Generic<F,TAG>::sEval(), params, n[TAG], n[1-TAG], gamma, args);

    

}
