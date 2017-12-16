#include <iostream>
#include <assert.h>
#include <cuda.h>
#include <vector>
#include <algorithm>
#include <benchmark/benchmark.h>


using namespace std;


float floatrand() {
    return ((float)rand())/RAND_MAX-.5;    // random value between -.5 and .5
}

template < class V > void fillrandom(V& v) {
    generate(v.begin(), v.end(), floatrand);    // fills vector with random values
}


extern "C" int GpuConv(float*, int, int, float*, float**);
extern "C" int CpuConv(float*, int, int, float*, float**);

void main_generic() {

    int Nx=5000, Ny=2000;

    vector<float> vf(Nx*3);
    fillrandom(vf);
    float *f = vf.data();
    vector<float> vx(Nx*3);
    fillrandom(vx);
    float *x = vx.data();
    vector<float> vy(Ny*3);
    fillrandom(vy);
    float *y = vy.data();
    vector<float> vu(Nx*4);
    fillrandom(vu);
    float *u = vu.data();
    vector<float> vv(Ny*4);
    fillrandom(vv);
    float *v = vv.data();
    vector<float> vb(Ny*3);
    fillrandom(vb);
    float *b = vb.data();

    vector<float*> vargs(5);
    vargs[0]=x;
    vargs[1]=y;
    vargs[2]=u;
    vargs[3]=v;
    vargs[4]=b;
    float **args = vargs.data();

    vector<float> resgpu(Nx*3), rescpu(Nx*3);

    float params[1];
    float Sigma = 1;
    params[0] = 1.0/(Sigma*Sigma);

    GpuConv(params, Nx, Ny, f, args);


}
extern "C" int GaussGpuGrad1Conv(float ooSigma2, float* alpha_h, float* x_h, float* y_h, float* beta_h, float* gamma_h, int dimPoint, int dimVect, int nx, int ny) ;

void main_specific() {

    int Nx=5000, Ny=2000;

    vector<float> vf(Nx*3);
    fillrandom(vf);
    float *f = vf.data();
    vector<float> vx(Nx*3);
    fillrandom(vx);
    float *x = vx.data();
    vector<float> vy(Ny*3);
    fillrandom(vy);
    float *y = vy.data();
    vector<float> vu(Nx*4);
    fillrandom(vu);
    float *u = vu.data();
    vector<float> vv(Ny*4);
    fillrandom(vv);
    float *v = vv.data();
    vector<float> vb(Ny*3);
    fillrandom(vb);
    float *b = vb.data();

    vector<float*> vargs(5);
    vargs[0]=x;
    vargs[1]=y;
    vargs[2]=u;
    vargs[3]=v;
    vargs[4]=b;
    float **args = vargs.data();

    vector<float> resgpu(Nx*3), rescpu(Nx*3);

    float params[1];
    float Sigma = 1;
    params[0] = 1.0/(Sigma*Sigma);

    GaussGpuGrad1Conv(params[0], u, x, y, v, f, 3,3,Nx,Ny);

}

static void BM_dummy(benchmark::State& state) {
    for (auto _ : state)
        main_generic();
}
// Register the function as a benchmark
BENCHMARK(BM_dummy);

// Define another benchmark
static void BM_specific(benchmark::State& state) {
    for (auto _ : state)
        main_specific();
}
BENCHMARK(BM_specific);

static void BM_generic(benchmark::State& state) {
    for (auto _ : state)
        main_generic();
}
// Register the function as a benchmark
BENCHMARK(BM_generic);


BENCHMARK_MAIN();
