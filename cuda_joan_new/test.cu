
// test convolution
// compile with 
//		nvcc -std=c++11 -o test test.cu

 
#define __TYPE__ double
#define __DIMPOINT__ 3 
#define __DIMVECT__ 3 
#define KERNEL SCALARRADIAL 	// type of kernel. Others are VARSURF (varifolds surfaces) and NCSURF (for normal cycles surfaces)
#define EVAL sEval 		// type of convolution for the type of kernel. For scalar radial kernels, others are sGrad1, sGrad, sHess, sDiff 
#define RADIALFUN CauchyFunction 	// Others are GaussFunction, Sum4GaussFunction, Sum4CauchyFunction



#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <vector>

#include "GpuConv2D.cu"

int main()
{
			
    int Nx=10000, Ny=10000;

	
    typedef typename KER::EVAL::DIMSX DIMSX;   
    typedef typename KER::EVAL::DIMSY DIMSY;
    const int SIZEX = DIMSX::SIZE;
    const int SIZEY = DIMSY::SIZE;        

    __TYPE__ *x[SIZEX];

    __TYPE__ *y[SIZEY];

	vector<__TYPE__>* tmp;
	
	// create random inputs
	for(int k=0; k<SIZEX; k++)
	{
		tmp = new vector<__TYPE__>(Nx*DIMSX::VAL(k));	
		x[k] = tmp->data();	
		for(int i=0; i<Nx*DIMSX::VAL(k); i++)
			x[k][i] = ((__TYPE__)rand())/RAND_MAX;
	}			
	
	for(int k=0; k<SIZEY; k++)
	{
		tmp = new vector<__TYPE__>(Ny*DIMSY::VAL(k));
		y[k] = tmp->data();
		for(int i=0; i<Ny*DIMSY::VAL(k); i++)
			y[k][i] = ((__TYPE__)rand())/RAND_MAX;
	}

	// set GPU device number
    //int deviceID = 0;
	//cudaSetDevice(deviceID);

	struct KER::EVAL funeval;
	typedef RADIALFUN<__TYPE__> RadialFun;

	// compute
	//GpuConv2D(KER(RadialFun()),funeval,Nx,Ny,x,y);
	GpuConv2D(KER(RadialFun()),funeval,Nx,Ny,x[0],x[1],y[0],y[1]);
	
	// display sum of output values
	__TYPE__ s = 0;
	for(int i=0; i<Nx*DIMSX::VAL(0); i++)
		s += x[0][i];
	cout << "sum=" << s << endl;

}

