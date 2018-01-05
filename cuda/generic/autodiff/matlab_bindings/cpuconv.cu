
// see compile_mex_cpu file for compiling

// F and __TYPE__ are supposed to be set via "using" or "#define" 

using FORMULA = decltype(F);

void ExitFcn(void) {
    cudaDeviceReset();
}

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


//////////////////////////////////////////////////////////////////
///////////////// MEX ENTRY POINT ////////////////////////////////
//////////////////////////////////////////////////////////////////


/* the gateway function */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    // register an exit function to prevent crash at matlab exit or recompiling
    mexAtExit(ExitFcn);
    
    const int TAG = 0;

	using VARSI = typename FORMULA::template VARS<TAG>;	// list variables of type I used in formula F
	using VARSJ = typename FORMULA::template VARS<1-TAG>; // list variables of type J used in formula F
	
	using DIMSX = GetDims<VARSI>;
	using DIMSY = GetDims<VARSJ>;
	
    using PARAMS = typename FORMULA::VARS<2>;
    static const int DIMPARAM = PARAMS::SIZE;

	using INDSI = GetInds<VARSI>;
	using INDSJ = GetInds<VARSJ>;

	using INDS = ConcatPacks<INDSI,INDSJ>;
	
	const int NARGSI = VARSI::SIZE; // number of I variables used in formula F
	const int NARGSJ = VARSJ::SIZE; // number of J variables used in formula F

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
	
	int NARGS = nrhs-2-(DIMPARAM?1:0);
	
    if(nlhs != 1) 
        mexErrMsgTxt("One output required.");


    //////////////////////////////////////////////////////////////
    // Input arguments
    //////////////////////////////////////////////////////////////


    int *typeargs = new int[NARGS];
    int *dimargs = new int[NARGS];
    for(int k=0; k<NARGS; k++)
    {
        typeargs[k] = -1;
        dimargs[k] = -1;
    }
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
    
	double *params;
	if(DIMPARAM) {
		//----- the last input argument: params--------------//
		argu+=NARGS;
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
    int dimout = FORMULA::DIM;
    int nout = n[TAG];
    plhs[0] = mxCreateDoubleMatrix(dimout,nout,mxREAL);

    /*  create a C pointer to a copy of the output result(vector)*/
    double *gamma = mxGetPr(plhs[0]);

    //////////////////////////////////////////////////////////////
    // Call Cuda codes
    //////////////////////////////////////////////////////////////
    
    CpuConv(Generic<FORMULA,TAG>::sEval(), params, n[TAG], n[1-TAG], gamma, args);

    delete[] args;
    delete[] typeargs;
    delete[] dimargs;
    

}
