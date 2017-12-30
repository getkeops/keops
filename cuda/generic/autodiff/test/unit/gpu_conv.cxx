
#include <iostream>
#include <assert.h>
#include <vector>
#include <ctime>
#include <algorithm>

#include "gtest/gtest.h"


extern "C" int GpuConv1D(float*, int, int, float*, float**);
extern "C" int GpuConv2D(float*, int, int, float*, float**);
extern "C" int CpuConv(float*, int, int, float*, float**);

float floatrand() {
    return ((float)rand())/RAND_MAX-.5;    // random value between -.5 and .5
}

template < class V > void fillrandom(V& v) {
    generate(v.begin(), v.end(), floatrand);    // fills vector with random values
}

namespace {
TEST(GpuConv, medium){
    int Nx=50000, Ny=2000;

    std::vector<float> vf(Nx*3);
    fillrandom(vf);
    float *f = vf.data();
    std::vector<float> vx(Nx*3);
    fillrandom(vx);
    float *x = vx.data();
    std::vector<float> vy(Ny*3);
    fillrandom(vy);
    float *y = vy.data();
    std::vector<float> vu(Nx*4);
    fillrandom(vu);
    float *u = vu.data();
    std::vector<float> vv(Ny*4);
    fillrandom(vv);
    float *v = vv.data();
    std::vector<float> vb(Ny*3);
    fillrandom(vb);
    float *b = vb.data();

    std::vector<float*> vargs(5);
    vargs[0] = x;
    vargs[1]=y;
    vargs[2]=u;
    vargs[3]=v;
    vargs[4]=b;
    float **args = vargs.data();

    std::vector<float> resgpu2D(Nx*3), resgpu1D(Nx*3), rescpu(Nx*3);

    float params[1];
    float Sigma = 1;
    params[0] = 1.0/(Sigma*Sigma);

    GpuConv2D(params, Nx, Ny, f, args);
    resgpu2D = vf;

    GpuConv1D(params, Nx, Ny, f, args);
    resgpu1D = vf;

    CpuConv(params, Nx, Ny, f, args);
    rescpu = vf;

    float s2d = 0;
    for(int i=0; i<Nx*3; i++)
        s2d += abs(resgpu2D[i]-rescpu[i]);

    float s1d = 0;
    for(int i=0; i<Nx*3; i++)
        s1d += abs(resgpu1D[i]-rescpu[i]);

    EXPECT_LE(s1d,5e-5);
    EXPECT_LE(s2d,5e-5);
}

TEST(GpuConv, big){
    int Nx=5000000, Ny=20000;

    std::vector<float> vf(Nx*3);
    fillrandom(vf);
    float *f = vf.data();
    std::vector<float> vx(Nx*3);
    fillrandom(vx);
    float *x = vx.data();
    std::vector<float> vy(Ny*3);
    fillrandom(vy);
    float *y = vy.data();
    std::vector<float> vu(Nx*4);
    fillrandom(vu);
    float *u = vu.data();
    std::vector<float> vv(Ny*4);
    fillrandom(vv);
    float *v = vv.data();
    std::vector<float> vb(Ny*3);
    fillrandom(vb);
    float *b = vb.data();

    std::vector<float*> vargs(5);
    vargs[0] = x;
    vargs[1]=y;
    vargs[2]=u;
    vargs[3]=v;
    vargs[4]=b;
    float **args = vargs.data();

    std::vector<float> resgpu2D(Nx*3), resgpu1D(Nx*3), rescpu(Nx*3);

    float params[1];
    float Sigma = 1;
    params[0] = 1.0/(Sigma*Sigma);

    GpuConv2D(params, Nx, Ny, f, args);
    resgpu2D = vf;

    GpuConv1D(params, Nx, Ny, f, args);
    resgpu1D = vf;

    CpuConv(params, Nx, Ny, f, args);
    rescpu = vf;

    float s2d = 0;
    for(int i=0; i<Nx*3; i++)
        s2d += abs(resgpu2D[i]-rescpu[i]);

    float s1d = 0;
    for(int i=0; i<Nx*3; i++)
        s1d += abs(resgpu1D[i]-rescpu[i]);

    EXPECT_LE(s1d,5e-5);
    EXPECT_LE(s2d,5e-5);
}

}  // namespace



GTEST_API_ int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
