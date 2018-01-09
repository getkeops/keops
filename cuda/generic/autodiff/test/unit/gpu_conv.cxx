
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

float one() {
    return ((float) 1.0); 
}

template < class V > void fillones(V& v) {
    generate(v.begin(), v.end(), one);    // fills vector with random values
}

struct vuple{
    std::vector<float> rescpu;
    std::vector<float> resgpu1D;
    std::vector<float> resgpu2D;
};

vuple compute_convs(int Nx, int Ny){
    std::vector<float> vf(Nx*3);
    fillrandom(vf);
    float *f = vf.data();
    std::vector<float> vx(Nx*3);
    fillrandom(vx);
    float *x = vx.data();
    std::vector<float> vy(Ny*3);
    fillrandom(vy);
    float *y = vy.data();
    std::vector<float> vu(Nx*3);
    fillones(vu);
    float *u = vu.data();
    std::vector<float> vv(Ny*3);
    fillones(vv);
    float *v = vv.data();
    std::vector<float> vb(Ny*3);
    fillrandom(vb);
    float *b = vb.data();

    std::vector<float*> vargs(3);
    vargs[0] = x;
    vargs[1]=y;
    vargs[2]=u;
    /*vargs[3]=v;*/
    /*vargs[4]=b;*/
    float **args = vargs.data();

    std::vector<float> resgpu2D(Nx*3), resgpu1D(Nx*3), rescpu(Nx*3);

    float params[1];
    float Sigma = .0000000001;
    params[0] = 1.0/(Sigma*Sigma);

    GpuConv2D(params, Nx, Ny, f, args);
    resgpu2D = vf;

    fillrandom(vf);
    GpuConv1D(params, Nx, Ny, f, args);
    resgpu1D = vf;

    fillrandom(vf);
    CpuConv(params, Nx, Ny, f, args);
    rescpu = vf;

    vuple res = {rescpu,resgpu1D,resgpu2D};

    return res;
}

namespace {

TEST(GpuConv, medium){
    int Nx=50000, Ny=2000;
    vuple res_conv = compute_convs(Nx, Ny);

    float s2d = 0;
    for(int i=0; i<Nx*3; i++)
        s2d += abs(res_conv.resgpu2D[i]-res_conv.rescpu[i]);

    float s1d = 0;
    for(int i=0; i<Nx*3; i++)
        s1d += abs(res_conv.resgpu1D[i]-res_conv.rescpu[i]);

    EXPECT_LE(s1d,5e-5);
    EXPECT_LE(s2d,5e-5);
}

TEST(GpuConv, big){
    int Nx=500, Ny=200;
    vuple res_conv = compute_convs(Nx, Ny);

    for(int i=0; i<Nx*3; i++){
        std::cout << "Checks : " << res_conv.resgpu2D[i] << " " << res_conv.resgpu1D[i] << " " << res_conv.rescpu[i] << " : " << i << std::endl;
    }

    float s2d = 0;
    for(int i=0; i<Nx*3; i++){
        float t = abs(res_conv.resgpu2D[i]-res_conv.rescpu[i]);
        s2d += t;
        if (t > 5e-5) 
            std::cout << res_conv.resgpu2D[i] << " " << res_conv.rescpu[i] << " : " << i << std::endl;
    }

    float s1d = 0;
    for(int i=0; i<Nx*3; i++)
        s1d += abs(res_conv.resgpu1D[i]-res_conv.rescpu[i]);

    float s21d = 0;
    for(int i=0; i<Nx*3; i++)
        s21d += abs(res_conv.resgpu1D[i]-res_conv.resgpu2D[i]);

    EXPECT_LE(s1d,5e-5);
    EXPECT_LE(s2d,5e-5);
    EXPECT_LE(s21d,5e-5);
}

}  // namespace



GTEST_API_ int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
