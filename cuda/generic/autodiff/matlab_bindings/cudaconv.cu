#include <mex.h>
#include "core/CpuConv.cpp"
#include "core/GpuConv1D.cu"
#include "core/GpuConv2D.cu"
#include "core/autodiff.h"

#include "core/newsyntax.h"
// see compile_mex file for compiling

// F and __TYPE__ are supposed to be set via "using" or "#define" 

void ExitFcn(void) {
	#ifdef __CUDACC__
	    cudaDeviceReset();
	#endif
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
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    // register an exit function to prevent crash at matlab exit or recompiling
    mexAtExit(ExitFcn);
        
	using VARSI = typename FORMULA::template VARS<0>;	// list variables of type I used in formula F
	using VARSJ = typename FORMULA::template VARS<1>; // list variables of type J used in formula F
	
	using DIMSX = GetDims<VARSI>;
	using DIMSY = GetDims<VARSJ>;
	
    using PARAMS = typename FORMULA::VARS<2>;
    static const int DIMPARAM = PARAMS::SIZE;

	using INDSI = GetInds<VARSI>;
	using INDSJ = GetInds<VARSJ>;

	using INDS = ConcatPacks<INDSI,INDSJ>;
	
	const int NARGSI = VARSI::SIZE; // number of I variables used in formula F
	const int NARGSJ = VARSJ::SIZE; // number of J variables used in formula F

	int NARGS = nrhs-5-(DIMPARAM?1:0);
	
    if(nlhs != 1) 
        mexErrMsgTxt("One output required.");

    //////////////////////////////////////////////////////////////
    // Input arguments
    //////////////////////////////////////////////////////////////

    int argu = 0;
    int n[2];
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
		mexErrMsgTxt("fifth arg should be scalar tagID2D");
	int tag1D2D = *mxGetPr(prhs[argu]);
	argu++;
	
    int *typeargs = new int[NARGS];
    int *dimargs = new int[NARGS];
    for(int k=0; k<NARGS; k++)
    {
        typeargs[k] = -1;
        dimargs[k] = -1;
    }
    for(int k=0; k<NARGSI; k++)
    {
        typeargs[INDSI::VAL(k)] = 0;
        dimargs[INDSI::VAL(k)] = DIMSX::VAL(k);
    }
    for(int k=0; k<NARGSJ; k++)
    {
        typeargs[INDSJ::VAL(k)] = 1;
        dimargs[INDSJ::VAL(k)] = DIMSY::VAL(k);
    }

    //----- the next input arguments: args--------------//
    /*  create pointers to the input vectors */
    double **args = new double*[NARGS];    
    for(int k=0; k<NARGS; k++)
    {
    	/*  input sources */
    	args[k] = mxGetPr(prhs[argu+k]);
    	// checking dimensions
    	if(dimargs[k]!=-1) // we care only if the current variable is used in formula
    	{
			int dimk = mxGetM(prhs[argu+k]);
			// we check nx and ny here from the formula
			int nk = mxGetN(prhs[argu+k]);
			int typek = typeargs[k];
			// we check dimension here from the formula
			if(dimk!=dimargs[k])
				mexErrMsgTxt("wrong dimension for input");
			if(n[typek]!=nk)
			{
				mexErrMsgTxt("inconsistent input sizes");
			}
		}
    }
    argu += NARGS;
    
	double *params;
	if(DIMPARAM) {
		//----- the next input argument: params--------------//
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
		argu++;
	}

    //////////////////////////////////////////////////////////////
    // Output arguments
    //////////////////////////////////////////////////////////////

    /*  set the output pointer to the output result(vector) */
    int dimout = FORMULA::DIM;
    int nout = n[tagIJ];
    plhs[0] = mxCreateDoubleMatrix(dimout,nout,mxREAL);

    /*  create a C pointer to a copy of the output result(vector)*/
    double *gamma = mxGetPr(plhs[0]);

    //////////////////////////////////////////////////////////////
    // Call Cuda codes
    //////////////////////////////////////////////////////////////
    
    // tagCpuGpu=0 means convolution on Cpu, tagCpuGpu=1 means convolution on Gpu, tagCpuGpu=2 means convolution on Gpu from device data
    // tagIJ=0 means sum over j, tagIJ=1 means sum over j
    // tag1D2D=0 means 1D Gpu scheme, tag1D2D=1 means 2D Gpu scheme
    
    if(tagCpuGpu==0)
    {
    	if(tagIJ==0)
		    CpuConv(Generic<FORMULA,0>::sEval(), params, n[0], n[1], gamma, args);
		else
			CpuConv(Generic<FORMULA,1>::sEval(), params, n[1], n[0], gamma, args);
	}
#ifdef __CUDACC__
	else if(tagCpuGpu==1)
    {
    	if(tagIJ==0)
    	{
    		if(tag1D2D==0)
			    GpuConv1D(Generic<FORMULA,0>::sEval(), params, n[0], n[1], gamma, args);
			else
			    GpuConv2D(Generic<FORMULA,0>::sEval(), params, n[0], n[1], gamma, args);
		}
		else
    	{
    		if(tag1D2D==0)
			    GpuConv1D(Generic<FORMULA,1>::sEval(), params, n[1], n[0], gamma, args);
			else
			    GpuConv2D(Generic<FORMULA,1>::sEval(), params, n[1], n[0], gamma, args);
		}		
	}
	else
    {
    	if(tagIJ==0)
    	{
    		if(tag1D2D==0)
			    GpuConv1D_FromDevice(Generic<FORMULA,0>::sEval(), params, n[0], n[1-0], gamma, args);
			else
			    GpuConv2D_FromDevice(Generic<FORMULA,0>::sEval(), params, n[0], n[1-0], gamma, args);
		}
		else
    	{
    		if(tag1D2D==0)
			    GpuConv1D_FromDevice(Generic<FORMULA,1>::sEval(), params, n[1], n[0], gamma, args);
			else
			    GpuConv2D_FromDevice(Generic<FORMULA,1>::sEval(), params, n[1], n[0], gamma, args);
		}		
	}
#endif

    delete[] args;
    delete[] typeargs;
    delete[] dimargs;
    

}
