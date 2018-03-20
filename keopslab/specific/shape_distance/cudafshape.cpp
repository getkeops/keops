#include <mex.h>

extern "C" int cudafshape(__TYPE__, __TYPE__, __TYPE__ , __TYPE__*, __TYPE__*, __TYPE__*, __TYPE__*, __TYPE__*, __TYPE__*, __TYPE__*, int, int, int, int, int);

//////////////////////////////////////////////////////////////////
///////////////// MEX ENTRY POINT ////////////////////////////////
//////////////////////////////////////////////////////////////////
void ExitFcn(void) {}
 
 /* the gateway function */
 void mexFunction( int nlhs, mxArray *plhs[],
                   int nrhs, const mxArray *prhs[]) { 
 //plhs: double *gamma
 //prhs: double *x, double *y, double* f, double* g, double *alpha, double *beta, double sigmax, double sigmaf, double sigmaXi

   // register an exit function to prevent crash at matlab exit or recompiling
   mexAtExit(ExitFcn);

   /*  check for proper number of arguments */
   if(nrhs != 9) 
     mexErrMsgTxt("9 inputs required.");
   if(nlhs < 1 | nlhs > 1) 
     mexErrMsgTxt("One output required.");
 
   //////////////////////////////////////////////////////////////
   // Input arguments
   //////////////////////////////////////////////////////////////
   
   int argu = -1;
 
   //----- the first input argument: x--------------//
   argu++;
   /*  create a pointer to the input vectors srcs */
   double *x = mxGetPr(prhs[argu]);
   /*  input sources */
   int dimpoint = mxGetM(prhs[argu]); //mrows
   int nx = mxGetN(prhs[argu]); //ncols
 
   //----- the second input argument: y--------------//
   argu++;
   /*  create a pointer to the input vectors trgs */
   double *y = mxGetPr(prhs[argu]);
   /*  get the dimensions of the input targets */
   int ny = mxGetN(prhs[argu]); //ncols
   /* check to make sure the first dimension is dimpoint */
   if( mxGetM(prhs[argu])!=dimpoint ) {
     mexErrMsgTxt("Input y must have same number of rows as x.");
   }

   //----- the third input argument: f--------------//
   argu++;
   /*  create a pointer to the input vectors srcs */
   double *f = mxGetPr(prhs[argu]);
   /*  get dimension of the signal */
   int dimsig = mxGetM(prhs[argu]); //mrows
   /* check to make sure the second dimension is nx and fist dim is 1*/
   if( mxGetM(prhs[argu])*mxGetN(prhs[argu])!=nx ) {
     mexErrMsgTxt("Input f must be a vector with the same number of columns as x.");
   }
 
   //----- the fourth input argument: g--------------//
   argu++;
   /*  create a pointer to the input vectors trgs */
   double *g = mxGetPr(prhs[argu]);
   /* check to make sure the second dimension is ny and first dim is 1 */
   if( mxGetM(prhs[argu])*mxGetN(prhs[argu])!=ny ) {
     mexErrMsgTxt("Input g must be a vector with the same number of columns as y.");
   }
    
  //------ the fifth input argument: alpha---------------//
   argu++;
   /*  create a pointer to the input vectors wts */
   double *alpha = mxGetPr(prhs[argu]);
   /*  get the dimensions of the input weights */
   int dimvect = mxGetM(prhs[argu]);
   /* check to make sure the second dimension is nx */
   if( mxGetN(prhs[argu])!=nx ) {
     mexErrMsgTxt("Input alpha must have same number of columns as x.");
   }

  //------ the sixth input argument: beta---------------//
   argu++;
   /*  create a pointer to the input vectors wts */
   double *beta = mxGetPr(prhs[argu]);
   /*  get the dimensions of the input weights */
   if (dimvect != mxGetM(prhs[argu])){
	   mexErrMsgTxt("Input beta must have the same number of row as alpha");}
   /* check to make sure the second dimension is ny */
   if( mxGetN(prhs[argu])!=ny ) {
     mexErrMsgTxt("Input beta must have same number of columns as y.");
   }
 
   //----- the seventh input argument: sigmax-------------//
   argu++;
   /* check to make sure the input argument is a scalar */
   if( !mxIsDouble(prhs[argu]) || mxIsComplex(prhs[argu]) ||
       mxGetN(prhs[argu])*mxGetM(prhs[argu])!=1 ) {
     mexErrMsgTxt("Input sigmax must be a scalar.");
   }
   /*  get the input sigma */
   double sigmax = mxGetScalar(prhs[argu]);
   if (sigmax <= 0.0)
 	  mexErrMsgTxt("Input sigma must be a positive number.");
   double oosigmax2 = 1.0f/(sigmax*sigmax);
   
   //----- the eighth input argument: sigmaf-------------//
   argu++;
   /* check to make sure the input argument is a scalar */
    if( !mxIsDouble(prhs[argu]) || mxIsComplex(prhs[argu]) ||
       mxGetN(prhs[argu])*mxGetM(prhs[argu])!=1 ) {
     mexErrMsgTxt("Input sigmaf must be a scalar.");
   }
   /*  get the input sigma */
   double sigmaf = mxGetScalar(prhs[argu]);
   if (sigmaf <= 0.0){
	  mexErrMsgTxt("Input sigmaf must be a positive number.");
   }
   double oosigmaf2=1.0f/(sigmaf*sigmaf);

   //----- the ninth input argument: sigmaXi-------------//
   argu++;
   /* check to make sure the input argument is a scalar */
    if( !mxIsDouble(prhs[argu]) || mxIsComplex(prhs[argu]) ||
       mxGetN(prhs[argu])*mxGetM(prhs[argu])!=1 ) {
     mexErrMsgTxt("Input sigmaxi must be a scalar.");
   }
   /*  get the input sigma */
   double sigmaXi = mxGetScalar(prhs[argu]);
   if (sigmaXi <= 0.0){
	  mexErrMsgTxt("Input sigmaxi must be a positive number.");
   }
   double oosigmaXi2=1.0f/(sigmaXi*sigmaXi);


   //////////////////////////////////////////////////////////////
   // Output arguments
   //////////////////////////////////////////////////////////////
   /*  set the output pointer to the output result(vector) */
   plhs[0] = mxCreateDoubleMatrix(1,nx,mxREAL);
   
   /*  create a C pointer to a copy of the output result(vector)*/
   double *gamma = mxGetPr(plhs[0]);
   
#if  USE_DOUBLE
   cudafshape(oosigmax2,oosigmaf2,oosigmaXi2,x,y,f,g,alpha,beta,gamma,dimpoint,dimsig,dimvect,nx,ny);
#else
   // convert to float
   
   float *x_f = new float[nx*dimpoint];
   for(int i=0; i<nx*dimpoint; i++)
     x_f[i] = x[i];

   float *y_f = new float[ny*dimpoint];
   for(int i=0; i<ny*dimpoint; i++)
     y_f[i] = y[i];

   float *f_f = new float[nx*dimsig];
   for(int i=0; i<nx*dimsig; i++)
     f_f[i] = f[i];
   
   float *g_f = new float[ny*dimsig];
   for(int i=0; i<ny*dimsig; i++)
     g_f[i] = g[i];

   float *alpha_f = new float[nx*dimvect];
   for(int i=0; i<nx*dimvect; i++)
     alpha_f[i] = alpha[i];

   float *beta_f = new float[ny*dimvect];
   for(int i=0; i<ny*dimvect; i++)
     beta_f[i] = beta[i];
 
   
   // function calls;
   float *gamma_f = new float[nx];
   cudafshape(oosigmax2,oosigmaf2,oosigmaXi2,x_f,y_f,f_f,g_f,alpha_f,beta_f,gamma_f,dimpoint,dimsig,dimvect,nx,ny);
 
   for(int i=0; i<nx; i++)
       gamma[i] = gamma_f[i];

   delete [] x_f;
   delete [] y_f;
   delete [] f_f;
   delete [] g_f;
   delete [] alpha_f;
   delete [] beta_f;
   delete [] gamma_f;
#endif
   
   return;
   
 }

