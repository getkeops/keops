#include <iostream>
#include <assert.h>
#include <cuda.h>
#include <vector>
#include <algorithm>

using namespace std;


// Some convenient functions
float floatrand() {
    return ((float)rand())/RAND_MAX-.5;    // random value between -.5 and .5
}

template < class V > void fillrandom(V& v) {
    generate(v.begin(), v.end(), floatrand);    // fills vector with random values
}

extern "C" int GaussGpuEvalConv(float ooSigma2, float* x_h, float* y_h, float* beta_h, float* gamma_h, int dimPoint, int dimVect, int nx, int ny) ;

int main() {

    int Nx = 150000 ;
    int Ny= 100000 ;

    int dimPoint = 3;
    int dimVect = 3;

    vector<float> vf(Nx*dimVect);
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

    float Sigma =1;
    float ooSigma2 = 1.0/(Sigma*Sigma);

    GaussGpuEvalConv(ooSigma2, x, y, v, f, dimPoint, dimVect,Nx,Ny);

    return 0;
}
