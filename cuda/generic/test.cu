// test convolution
// compile with 
//		nvcc -std=c++11 -o test test.cu

 
#define __TYPE__ double
#define __DIMPOINT__ 3
#define __DIMVECT__ 3
#define KERNEL SCALARRADIAL 	// type of kernel. Others are VARSURF (varifolds surfaces) and NCSURF (for normal cycles surfaces)
#define EVAL sEval 	// type of convolution for the type of kernel. For scalar radial kernels, others are sGrad1, sGrad, sHess, sDiff 
#define RADIALFUN GaussFunction 	// Others are CauchyFunction, Sum4GaussFunction, Sum4CauchyFunction



#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <vector>
#include <ctime>

#include "GpuConv2D.cu"
#include "CudaScalarRadialKernels.h"
#include "CudaNCSurfKernels.h"
#include "CudaVarSurfKernels.h"

int main()
{
			
    int Nx=5000, Ny=5000;

	
    typedef typename KER::EVAL::DIMSX DIMSX;   
    typedef typename KER::EVAL::DIMSY DIMSY;
    const int SIZEX = DIMSX::SIZE;
    const int SIZEY = DIMSY::SIZE;        

    __TYPE__ *x[SIZEX];

    __TYPE__ *y[SIZEY];

	vector< vector<__TYPE__> > vx(SIZEX), vy(SIZEY);
	vector<__TYPE__> resgpu(Nx*DIMSX::VAL(0)), rescpu(Nx*DIMSX::VAL(0));
	
	// create random inputs
	for(int k=0; k<SIZEX; k++)
	{
		vx[k].resize(Nx*DIMSX::VAL(k));	
		x[k] = vx[k].data();	
		for(int i=0; i<Nx*DIMSX::VAL(k); i++)
			x[k][i] = ((__TYPE__)rand())/RAND_MAX-.5;
	}			
	
	for(int k=0; k<SIZEY; k++)
	{
		vy[k].resize(Ny*DIMSY::VAL(k));
		y[k] = vy[k].data();
		for(int i=0; i<Ny*DIMSY::VAL(k); i++)
			y[k][i] = ((__TYPE__)rand())/RAND_MAX-.5;
	}

	// set GPU device number
    int deviceID = 0;
    
    clock_t begin, end;
    __TYPE__ s;
    
    begin = clock();
	cudaSetDevice(deviceID);
	end = clock();
	cout << "time for GPU initialization : " << double(end - begin) / CLOCKS_PER_SEC << endl;
	
	begin = clock();
	struct KER::EVAL funeval;
	typedef RADIALFUN<__TYPE__> RadialFun;
	
	// compute
	begin = clock();
	GpuConv2D(KER(RadialFun()),funeval,Nx,Ny,x,y);
	end = clock();
	cout << "time for GPU computation (first run) : " << double(end - begin) / CLOCKS_PER_SEC << endl;
		
	begin = clock();
	GpuConv2D(KER(RadialFun()),funeval,Nx,Ny,x,y);
	end = clock();
	cout << "time for GPU computation (second run) : " << double(end - begin) / CLOCKS_PER_SEC << endl;
	
	resgpu = vx[0];
	
	begin = clock();
	CpuConv(KER(RadialFun()),funeval,Nx,Ny,x,y);
	end = clock();
	cout << "time for CPU computation : " << double(end - begin) / CLOCKS_PER_SEC << endl;

	rescpu = vx[0];
	
	// display sum of errors
	s = 0;
	for(int i=0; i<Nx*DIMSX::VAL(0); i++)
		s += abs(resgpu[i]-rescpu[i]);
	cout << "mean abs error =" << s/Nx << endl;

}

