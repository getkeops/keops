// test convolution with autodiff
// compile with
//		nvcc -std=c++11 -O2 -c test_link_grady.cu
//		./compile "Grad<Scal<Square<Scalprod<X<2,4>,Y<3,4>>>,GaussKernel<P<0>,X<0,3>,Y<1,3>,Y<4,3>>>,Y<1,3>,X<5,3>>"
// 		nvcc -o test_link_grady test_link_grady.o "build/Grad<Scal<Square<Scalprod<X<2,4>,Y<3,4>>>,GaussKernel<P<0>,X<0,3>,Y<1,3>,Y<4,3>>>,Y<1,3>,X<5,3>>.so"


#include <iostream>
#include <assert.h>
#include <cuda.h>
#include <vector>
#include <ctime>
#include <algorithm>

using namespace std;


extern "C" int GpuTransConv1D(float*, int, int, float*, float**);
extern "C" int GpuTransConv2D(float*, int, int, float*, float**);
extern "C" int CpuTransConv(float*, int, int, float*, float**);

float floatrand() {
    return ((float) std::rand())/RAND_MAX-.5; // random value between -.5 and .5
}

template < class V > void fillrandom(V& v) {
    generate(v.begin(), v.end(), floatrand); // fills vector with random values
}

int main() {

    int Nx=5000, Ny=2000;

    vector<float> vf(Nx*3); fillrandom(vf); float *f = vf.data();
    vector<float> vx(Nx*3); fillrandom(vx); float *x = vx.data();
    vector<float> vy(Ny*3); fillrandom(vy); float *y = vy.data();
    vector<float> vu(Nx*4); fillrandom(vu); float *u = vu.data();
    vector<float> vv(Ny*4); fillrandom(vv); float *v = vv.data();
    vector<float> vb(Ny*3); fillrandom(vb); float *b = vb.data();
    vector<float> vex(Nx*3); fillrandom(vex); float *ex = vex.data();

    vector<float*> vargs(6);
    vargs[0] = x;
    vargs[1]=y;
    vargs[2]=u;
    vargs[3]=v;
    vargs[4]=b;
    vargs[5]=ex;
    float **args = vargs.data();

    vector<float> resgpu2D(Ny*3), resgpu1D(Ny*3), rescpu(Ny*3);

    float params[1];
    float Sigma = 1;
    params[0] = 1.0/(Sigma*Sigma);

    clock_t begin, end;

    begin = clock();
    int deviceID = 0;
    cudaSetDevice(deviceID);
    end = clock();
    cout << "time for GPU initialization : " << double(end - begin) / CLOCKS_PER_SEC << endl;

    cout << "testing gradient wrt y" << endl;
    begin = clock();
    GpuTransConv2D(params, Nx, Ny, f, args);
    end = clock();
    cout << "time for GPU computation (2D) : " << double(end - begin) / CLOCKS_PER_SEC << endl;

    resgpu2D = vf;

    begin = clock();
    GpuTransConv1D(params, Nx, Ny, f, args);
    end = clock();
    cout << "time for GPU computation (1D) : " << double(end - begin) / CLOCKS_PER_SEC << endl;


    resgpu1D = vf;

    begin = clock();
    CpuTransConv(params, Nx, Ny, f, args);
    end = clock();
    cout << "time for CPU computation : " << double(end - begin) / CLOCKS_PER_SEC << endl;

    rescpu = vf;

    // display mean of errors
    float s = 0;
    for(int i=0; i<Ny*3; i++)
        s += abs(resgpu2D[i]-rescpu[i]);
    cout << "mean abs error 2D=" << s/Ny << endl;


    s = 0;
    for(int i=0; i<Ny*3; i++)
        s += abs(resgpu1D[i]-rescpu[i]);
    cout << "mean abs error 1D =" << s/Ny << endl;


}



