// test convolution with autodiff
// compile with 
//		nvcc -std=c++11 -O2 -c test_link.cu
//		./compile "Scal<Square<Scalprod<X<2,4>,Y<3,4>>>,GaussKernel<P<0>,X<0,3>,Y<1,3>,Y<4,3>>>"
// 		nvcc -o test_link test_link.o "build/Scal<Square<Scalprod<X<2,4>,Y<3,4>>>,GaussKernel<P<0>,X<0,3>,Y<1,3>,Y<4,3>>>.so"


#include <iostream>
#include <assert.h>
#include <cuda.h>
#include <vector>
#include <ctime>
#include <algorithm>

using namespace std;


extern "C" int GpuConv(float*, int, int, float*, float**);
extern "C" int CpuConv(float*, int, int, float*, float**);

float floatrand() { return ((float)rand())/RAND_MAX-.5; } // random value between -.5 and .5

template < class V > void fillrandom(V& v) { generate(v.begin(), v.end(), floatrand); } // fills vector with random values

int main()
{
	
	int Nx=5000, Ny=2000;

	vector<float> vf(Nx*3); fillrandom(vf); float *f = vf.data();
	vector<float> vx(Nx*3); fillrandom(vx); float *x = vx.data();
	vector<float> vy(Ny*3); fillrandom(vy); float *y = vy.data();
	vector<float> vu(Nx*4); fillrandom(vu); float *u = vu.data();
	vector<float> vv(Ny*4); fillrandom(vv); float *v = vv.data();
	vector<float> vb(Ny*3); fillrandom(vb); float *b = vb.data();

	vector<float*> vargs(5); vargs[0] = x; vargs[1]=y; vargs[2]=u; vargs[3]=v; vargs[4]=b; float **args = vargs.data();
	
	vector<float> resgpu(Nx*3), rescpu(Nx*3);

	float params[1];
	float Sigma = 1;
	params[0] = 1.0/(Sigma*Sigma);
	
	clock_t begin, end;
	
	begin = clock();
	int deviceID = 0;
	cudaSetDevice(deviceID);
	end = clock();
	cout << "time for GPU initialization : " << double(end - begin) / CLOCKS_PER_SEC << endl;
	
	cout << "testing function" << endl;
	begin = clock();
	GpuConv(params, Nx, Ny, f, args);
	end = clock();
	cout << "time for GPU computation (first run) : " << double(end - begin) / CLOCKS_PER_SEC << endl;
	
	begin = clock();
	GpuConv(params, Nx, Ny, f, args);
	end = clock();
	cout << "time for GPU computation (second run) : " << double(end - begin) / CLOCKS_PER_SEC << endl;
	

	resgpu = vf;
		
	begin = clock();
	CpuConv(params, Nx, Ny, f, args);
	end = clock();
	cout << "time for CPU computation : " << double(end - begin) / CLOCKS_PER_SEC << endl;
	
	rescpu = vf;
	
	// display mean of errors
	float s = 0;
	for(int i=0; i<Nx*3; i++)
		s += abs(resgpu[i]-rescpu[i]);
	cout << "mean abs error =" << s/Nx << endl;


	



}



