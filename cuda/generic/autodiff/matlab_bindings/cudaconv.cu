#include <mex.h>
#include "link_autodiff.cu"

using FUN = Generic<FORMULA>;
using VARSI = FUN::VARSI;
using VARSJ = FUN::VARSJ;
using DIMSX = FUN::DIMSX;
using DIMSY = FUN::DIMSY;

using INDSI = FUN::INDSI;
using INDSJ = FUN::INDSJ;

using INDS = FUN::INDS;


template < typename TYPE >
void CallCuda<float>(double *params, int nx, int ny, int nargs, double **args) { }

template < > 
void CallCuda<float>(double *params, int nx, int ny, int nargs, double **args)
{
	GpuConv2D(FUN::sEval(), params, nx, ny, gamma, args);
}

template < > 
void CallCuda<double>(double *params, int nx, int ny, int nargs, double **args)
{
    // convert to float
    float **args_f = new float*[nargs];
    for(int i=0; i<nargs; i++)
    {
    	int ni = *** // from formula
    	int dimi = ***
    	
    	/////////
    	
    	
    	args_f[i] = new float[ni*dimi];
	    for(int i=0; i<nx*dimpoint; i++)
        	x_f[i] = x[i];
    for(int i=0; i<ny*dimpoint; i++)
        y_f[i] = y[i];
    for(int i=0; i<ny*dimvect; i++)
        beta_f[i] = beta[i];


    // function calls;
    GpuConv2D(Generic<FORMULA>::sEval(), params_f, nx, ny, gamma_f, args_f);

    for(int i=0; i<nx*dimvect; i++)
        gamma[i] = gamma_f[i];

    delete [] x_f;
    delete [] y_f;
    delete [] beta_f;
    delete [] gamma_f;
}



//////////////////////////////////////////////////////////////////
///////////////// MEX ENTRY POINT ////////////////////////////////
//////////////////////////////////////////////////////////////////


/* the gateway function */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    //plhs: double *gamma
    //prhs: double *params, double *arg0, double *arg1, ...

    // register an exit function to prevent crash at matlab exit or recompiling
    mexAtExit(ExitFcn);
    
    /*  check for proper number of arguments */
    if(nrhs != 1+) 
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
    if( min(mp,np)!=1 ) {
        mexErrMsgTxt("Input params must be a 1D array.");
    }
    np = max(mp,np);

	//----- the next input arguments: args--------------//
    argu++;
    int nargs = nrhs-argu;
	/*  create pointers to the input vectors */
    double **args = new double*[nargs];    
    for(int i=0; i<nargs; i++)
    {
    	/*  input sources */
    	args[i] = mxGetPr(prhs[argu+i]);
    	int mx = mxGetM(prhs[argu+i]); //mrows
    	int nx = mxGetN(prhs[argu+i]); //ncols
    	// we should check dimensions here from the formula
		// ...
		// ...
	}

    //////////////////////////////////////////////////////////////
    // Output arguments
    //////////////////////////////////////////////////////////////

    /*  set the output pointer to the output result(vector) */
    int dimout = *** // infer from formula
    int nout = *** // infer from formula
    plhs[0] = mxCreateDoubleMatrix(dimout,nout,mxREAL);

    /*  create a C pointer to a copy of the output result(vector)*/
    double *gamma = mxGetPr(plhs[0]);

    //////////////////////////////////////////////////////////////
    // Call Cuda codes
    //////////////////////////////////////////////////////////////
    
    CallCuda<__TYPE__>(params, nargs, args)


    return;

}
