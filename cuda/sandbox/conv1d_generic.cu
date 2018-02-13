#include <iostream>
#include <assert.h>
#include <cuda.h>
#include <vector>
#include <algorithm>

// load GPUConv1D
using namespace std;

// Some convenient functions
float floatrand() {
    return ((float)rand())/RAND_MAX-.5;    // random value between -.5 and .5
}

template < class V > void fillrandom(V& v) {
    generate(v.begin(), v.end(), floatrand);    // fills vector with random values
}




// Signature of the generic function:
extern "C" int GpuConv1D(float*, int, int, float*, float**);



int main() {

    int Nx =150000 ;
    int Ny= 100000 ;

    int dimPoint = 3;
    int dimVect = 3;

    vector<float> vf(Nx*dimPoint);
    fillrandom(vf);
    float *f = vf.data();

    vector<float> vx(Nx*dimPoint);
    fillrandom(vx);
    float *x = vx.data();

    vector<float> vy(Ny*dimPoint);
    fillrandom(vy);
    float *y = vy.data();

    vector<float> vv(Ny*dimVect);
    fillrandom(vv);
    float *v = vv.data();

    // wrap variables
    vector<float*> vargs(3);
    vargs[0]=x;
    vargs[1]=y;
    vargs[2]=v;
    float **args = vargs.data();

    float params[1];
    float Sigma = 1;
    params[0] = 1.0/(Sigma*Sigma);

    GpuConv1D(params, Nx, Ny, f, args);

    return 0;
}



