#include <mex.h>


extern "C" int GaussGpuEval  (__TYPE__, __TYPE__*, __TYPE__*, __TYPE__*, __TYPE__*, __TYPE__*, int, int, int, int);
extern "C" int LaplaceGpuEval(__TYPE__, __TYPE__*, __TYPE__*, __TYPE__*, __TYPE__*, __TYPE__*, int, int, int, int);
extern "C" int InverseMultiquadricGpuEval(__TYPE__, __TYPE__*, __TYPE__*, __TYPE__*, __TYPE__*, __TYPE__*, int, int, int, int);
extern "C" int CauchyGpuEval(__TYPE__, __TYPE__*, __TYPE__*, __TYPE__*, __TYPE__*, __TYPE__*, int, int, int, int);

//////////////////////////////////////////////////////////////////
///////////////// MEX ENTRY POINT ////////////////////////////////
//////////////////////////////////////////////////////////////////


/* the gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[]){
    //plhs: double *gamma
    //prhs: double *alpha, double *x, double *y, double *beta, double sigma, char kernel_type
 
    // register an exit function to prevent crash at matlab exit or recompiling
    mexAtExit(ExitFcn);

    const char *kernel_type = "gaussian";

    /*  check for proper number of arguments */
    if(nrhs <= 4) 
        mexErrMsgTxt("6 inputs required.");
    if(nrhs >= 7) 
        mexErrMsgTxt("6 inputs required.");
    if(nlhs < 1 | nlhs > 1) 
        mexErrMsgTxt("One output required.");

    //////////////////////////////////////////////////////////////
    // Input arguments
    //////////////////////////////////////////////////////////////

    int argu = -1;

    //------ the first input argument: alpha---------------//
    argu++;
    /*  create a pointer to the input vectors wts */
    double *alpha = mxGetPr(prhs[argu]);
    /*  get the dimensions of the input weights */
    int dimvect = mxGetM(prhs[argu]);
    int nx = mxGetN(prhs[argu]); //ncols

    //----- the second input argument: x--------------//
    argu++;
    /*  create a pointer to the input vectors srcs */
    double *x = mxGetPr(prhs[argu]);
    /*  input sources */
    int dimpoint = mxGetM(prhs[argu]); //mrows
    /* check to make sure the number of columns is nx */
    if( mxGetN(prhs[argu])!=nx ) {
        mexErrMsgTxt("Input x must have same number of columns as alpha.");
    }

    //----- the third input argument: y--------------//
    argu++;
    /*  create a pointer to the input vectors tgts */
    double *y = mxGetPr(prhs[argu]);
    /*  input sources */
    int ny = mxGetN(prhs[argu]); //ncols
    /* check to make sure the number of rows is dimpoint */
    if( mxGetM(prhs[argu])!=dimpoint )
        mexErrMsgTxt("Input y must have same number of rows as x.");

    //------ the fourth input argument: beta---------------//
    argu++;
    /*  create a pointer to the input vectors wts */
    double *beta = mxGetPr(prhs[argu]);
    /* check to make sure the number of rows is dimvect */
    if( mxGetM(prhs[argu])!=dimvect )
        mexErrMsgTxt("Input y must have same number of rows as alpha.");
    /* check to make sure the number of columns is ny */
    if( mxGetN(prhs[argu])!=ny )
        mexErrMsgTxt("Input beta must have same number of columns as y.");

    //----- the fifth input argument: sigma-------------//
    argu++;
    /* check to make sure the input argument is a scalar */
    if( !mxIsDouble(prhs[argu]) || mxIsComplex(prhs[argu]) ||
            mxGetN(prhs[argu])*mxGetM(prhs[argu])!=1 ) {
        mexErrMsgTxt("Input sigma must be a scalar.");
    }
    /*  get the scalar input sigma */
    double sigma = mxGetScalar(prhs[argu]);
    if (sigma <= 0.0)
        mexErrMsgTxt("Input sigma must be a positive number.");
    double oosigma2 = 1.0f/(sigma*sigma);

    //----- the sixtht input argument: kernel_type-------------//
    if (nrhs == 6) {
        argu++;
        /* check to make sure the input argument is a scalar */
        if( !mxIsChar(prhs[argu]) ) {
            mexErrMsgTxt("Input kernel_type must be a string.");
        }  

        /*  get the scalar input sigma */
        kernel_type = mxArrayToString(prhs[argu]);
    }

    //////////////////////////////////////////////////////////////
    // Output arguments
    //////////////////////////////////////////////////////////////
    /*  set the output pointer to the output result(vector) */
    plhs[0] = mxCreateDoubleMatrix(dimpoint,nx,mxREAL);

    /*  create a C pointer to a copy of the output result(vector)*/
    double *gamma = mxGetPr(plhs[0]);

#if USE_DOUBLE
    if (strcmp(kernel_type,"gaussian") == 0)
        GaussGpuEval(oosigma2,alpha,x,y,beta,gamma,dimpoint,dimvect,nx,ny);  
    else if (strcmp(kernel_type,"laplacian") == 0)
        LaplaceGpuEval(oosigma2,alpha,x,y,beta,gamma,dimpoint,dimvect,nx,ny);
    else if (strcmp(kernel_type,"cauchy") == 0)
        CauchyGpuEval(oosigma2,alpha,x,y,beta,gamma,dimpoint,dimvect,nx,ny);
    else if (strcmp(kernel_type,"inverse_multiquadric") == 0)
        InverseMultiquadricGpuEval(oosigma2,alpha,x,y,beta,gamma,dimpoint,dimvect,nx,ny);
    else
        mexErrMsgTxt("kernel_type is not implemented");
#else
    // convert to float
    float *alpha_f = new float[nx*dimvect];
    float *x_f = new float[nx*dimpoint];
    float *y_f = new float[ny*dimpoint];
    float *beta_f = new float[ny*dimvect];
    float *gamma_f = new float[nx*dimpoint];
    for(int i=0; i<nx*dimvect; i++)
        alpha_f[i] = alpha[i];
    for(int i=0; i<nx*dimpoint; i++)
        x_f[i] = x[i];
    for(int i=0; i<ny*dimpoint; i++)
        y_f[i] = y[i];
    for(int i=0; i<ny*dimvect; i++)
        beta_f[i] = beta[i];

    // function calls;
    if (strcmp(kernel_type,"gaussian") == 0)
        GaussGpuEval(oosigma2,alpha_f,x_f,y_f,beta_f,gamma_f,dimpoint,dimvect,nx,ny);
    else if (strcmp(kernel_type,"laplacian") == 0)
        LaplaceGpuEval(oosigma2,alpha_f,x_f,y_f,beta_f,gamma_f,dimpoint,dimvect,nx,ny);
    else if (strcmp(kernel_type,"cauchy") == 0)
        CauchyGpuEval(oosigma2,alpha_f,x_f,y_f,beta_f,gamma_f,dimpoint,dimvect,nx,ny);
    else if (strcmp(kernel_type,"inverse_multiquadric") == 0)
        InverseMultiquadricGpuEval(oosigma2,alpha_f,x_f,y_f,beta_f,gamma_f,dimpoint,dimvect,nx,ny);
    else
        mexErrMsgTxt("kernel_type is not implemented");

    for(int i=0; i<nx*dimpoint; i++)
        gamma[i] = gamma_f[i];

    delete [] alpha_f;
    delete [] x_f;
    delete [] y_f;
    delete [] beta_f;
    delete [] gamma_f;
#endif

    return;

}
