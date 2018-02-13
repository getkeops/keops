#include <iostream>
#include <assert.h>
#include <vector>
#include <ctime>
#include <algorithm>

#include "gtest/gtest.h"


extern "C" int GpuConv1D(__TYPE__*, int, int, __TYPE__*, __TYPE__**);
extern "C" int GpuConv2D(__TYPE__*, int, int, __TYPE__*, __TYPE__**);
extern "C" int CpuConv(__TYPE__*, int, int, __TYPE__*, __TYPE__**);

__TYPE__ __TYPE__rand() {
    return ((__TYPE__)rand())/RAND_MAX-.5;    // random value between -.5 and .5
}

template < class V > void fillrandom(V& v) {
    generate(v.begin(), v.end(), __TYPE__rand);    // fills vector with random values
}

__TYPE__ one() {
    return ((__TYPE__) 1.0); 
}

template < class V > void fillones(V& v) {
    generate(v.begin(), v.end(), one);    // fills vector with ones
}

struct vuple{
    std::vector<__TYPE__> rescpu;
    std::vector<__TYPE__> resgpu1D;
    std::vector<__TYPE__> resgpu2D;
};

vuple compute_convs(int Nx, int Ny){
    std::vector<__TYPE__> vf(Nx*3);
    fillrandom(vf);
    __TYPE__ *f = vf.data();

    std::vector<__TYPE__> vx(Nx*3);
    fillrandom(vx);
    __TYPE__ *x = vx.data();
    
    std::vector<__TYPE__> vy(Ny*3);
    fillrandom(vy);
    __TYPE__ *y = vy.data();
    
    std::vector<__TYPE__> vu(Nx*3);
    fillrandom(vu);
    __TYPE__ *u = vu.data();
    
    std::vector<__TYPE__> vv(Ny*3);
    fillrandom(vv);
    __TYPE__ *v = vv.data();
    
    std::vector<__TYPE__> vb(Ny*3);
    fillrandom(vb);
    __TYPE__ *b = vb.data();

    std::vector<__TYPE__*> vargs(5);
    vargs[0]=x;
    vargs[1]=y;
    vargs[2]=u;
    vargs[3]=v;
    vargs[4]=b;
    __TYPE__ **args = vargs.data();

    std::vector<__TYPE__> resgpu2D(Nx*3), resgpu1D(Nx*3), rescpu(Nx*3);

    __TYPE__ params[1];
    __TYPE__ Sigma = .1;
    params[0] = 1.0/(Sigma*Sigma);

    GpuConv2D(params, Nx, Ny, f, args);
    resgpu2D = vf;

    fillones(vf);
    GpuConv1D(params, Nx, Ny, f, args);
    resgpu1D = vf;

    fillones(vf);
    CpuConv(params, Nx, Ny, f, args);
    rescpu = vf;

    vuple res = {rescpu,resgpu1D,resgpu2D};

    return res;
}

namespace {

TEST(GpuConv, medium){
    int Nx=500, Ny=2000;
    std::cout << "Checks : " << std::endl;
    vuple res_conv = compute_convs(Nx, Ny);

    __TYPE__ s2d = 0;
    for(int i=0; i<Nx*3; i++)
        s2d += abs(res_conv.resgpu2D[i]-res_conv.rescpu[i]);

    __TYPE__ s1d = 0;
    for(int i=0; i<Nx*3; i++)
        s1d += abs(res_conv.resgpu1D[i]-res_conv.rescpu[i]);

    EXPECT_LE(s1d,5e-5);
    EXPECT_LE(s2d,5e-5);
}

/*TEST(GpuConv, big){*/
    /*int Nx=500, Ny=200;*/
    /*vuple res_conv = compute_convs(Nx, Ny);*/

    /*for(int i=0; i<Nx*3; i++){*/
        /*std::cout << "Checks : " << res_conv.resgpu2D[i] << " " << res_conv.resgpu1D[i] << " " << res_conv.rescpu[i] << " : " << i << std::endl;*/
    /*}*/

    /*__TYPE__ s2d = 0;*/
    /*for(int i=0; i<Nx*3; i++){*/
        /*__TYPE__ t = abs(res_conv.resgpu2D[i]-res_conv.rescpu[i]);*/
        /*s2d += t;*/
        /*if (t > 5e-5) */
            /*std::cout << res_conv.resgpu2D[i] << " " << res_conv.rescpu[i] << " : " << i << std::endl;*/
    /*}*/

    /*__TYPE__ s1d = 0;*/
    /*for(int i=0; i<Nx*3; i++)*/
        /*s1d += abs(res_conv.resgpu1D[i]-res_conv.rescpu[i]);*/

    /*__TYPE__ s21d = 0;*/
    /*for(int i=0; i<Nx*3; i++)*/
        /*s21d += abs(res_conv.resgpu1D[i]-res_conv.resgpu2D[i]);*/

    /*EXPECT_LE(s1d,5e-5);*/
    /*EXPECT_LE(s2d,5e-5);*/
    /*EXPECT_LE(s21d,5e-5);*/
/*}*/

}  // namespace



GTEST_API_ int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
