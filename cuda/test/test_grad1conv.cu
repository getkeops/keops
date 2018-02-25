#include <iostream>
#include <gtest/gtest.h>

// use manual timing for GPU based functions
#include <chrono>
#include <ctime>

#include "bench/generate_data.h"

#include "core/GpuConv1D.cu"
#include "core/GpuConv2D.cu"
#include "core/CpuConv.cpp"

#include "core/formulas/constants.h"
#include "core/formulas/maths.h"
#include "core/formulas/kernels.h"
#include "core/formulas/norms.h"
#include "core/formulas/factorize.h"

#include "core/autodiff.h"

#define ATOL 1e-3
#define RTOL 1e-4

using namespace std;

template <typename T>
void EXPECT_AllCLOSE(const vector<T> X, const vector<T> Y, const T atol, const T rtol) {
    ASSERT_EQ(X.size(), Y.size());

    int count = 0;
    T l1norm = 0.0;
    for (int i = 0; i < X.size(); ++i) {
        if (std::abs(X[i] - Y[i])> atol + rtol * (std::abs(Y[i]) + std::abs(X[i])))
            count +=1;
        l1norm +=  std::abs(X[i] - Y[i]);
    }
        EXPECT_LE(count,0) << "number of non close values: " << count;
        EXPECT_LE(l1norm/X.size(), atol);
}


template <typename T>
void EXPECT_NONZEROS(const vector<T> X) {

    int nb_of_zeros = 0;
    for (int i = 0; i < X.size(); ++i) {
        if (std::abs(X[i]) == 0)
            nb_of_zeros +=1;
    }
        EXPECT_LT(nb_of_zeros,X.size());
}

/////////////////////////////////////////////////////////////////////////////////////
//                      The function to be benchmarked                            //
/////////////////////////////////////////////////////////////////////////////////////

#define F0 Grad<GaussKernel<_P<0>,_X<0,3>,_Y<1,3>,_Y<2,3>>,_X<0,3>,_X<3,3>>
using FUN0 = typename Generic<F0>::sEval;

extern "C" int GaussGpuGrad1Conv(__TYPE__ ooSigma2, __TYPE__* alpha_h, __TYPE__* x_h, __TYPE__* y_h, __TYPE__* beta_h, __TYPE__* gamma_h, int dimPoint, int dimVect, int nx, int ny) ;

//typedef template < typename TYPE, class FUN, class PARAM >int(*OP)(FUN fun, PARAM param, int nx, int ny, TYPE* x1_h, TYPE** args);
typedef int(*OP)(FUN0,__TYPE__*, int, int, __TYPE__*, __TYPE__*, __TYPE__*, __TYPE__*, __TYPE__*);
template <typename T, OP op>
class test_grad1conv {

    public:
    test_grad1conv(int);

    vector<T> vresgpu, vresgrad;
    T *resgpu, *resgrad;
    data<T> data1;
};

template <typename T, OP op>
test_grad1conv<T,op>::test_grad1conv(int Nx){

        data<T> data3(Nx); data1 = data3;

        vresgpu.resize(Nx*data1.dimPoint);  resgpu = vresgpu.data(); 
        vresgrad.resize(Nx*data1.dimPoint); resgrad = vresgrad.data(); 

        GaussGpuGrad1Conv(data1.params[0], data1.u, data1.x, data1.y, data1.v, resgpu, data1.dimPoint,data1.dimVect,data1.Nx,data1.Ny); 
        op(FUN0(), data1.params, data1.Nx, data1.Ny, resgrad, data1.x, data1.y, data1.v, data1.u);
}


namespace {
    TEST(grad1conv_1D, small){

        test_grad1conv<__TYPE__,GpuConv1D> test_small(2001);

        EXPECT_AllCLOSE<__TYPE__>(test_small.vresgrad,test_small.vresgpu, ATOL, RTOL);
        EXPECT_NONZEROS<__TYPE__>(test_small.vresgrad);
        EXPECT_NONZEROS<__TYPE__>(test_small.vresgpu);
    }

    TEST(grad1conv_1D, medium){

        test_grad1conv<__TYPE__,GpuConv1D> test_medium(20001);

        EXPECT_AllCLOSE<__TYPE__>(test_medium.vresgrad,test_medium.vresgpu, ATOL, RTOL);
        EXPECT_NONZEROS<__TYPE__>(test_medium.vresgrad);
        EXPECT_NONZEROS<__TYPE__>(test_medium.vresgpu);
    }


    TEST(grad1conv_1D, large){
        test_grad1conv<__TYPE__,GpuConv1D> test_large(200001);

        EXPECT_AllCLOSE<__TYPE__>(test_large.vresgrad,test_large.vresgpu, ATOL, RTOL);
        EXPECT_NONZEROS<__TYPE__>(test_large.vresgrad);
        EXPECT_NONZEROS<__TYPE__>(test_large.vresgpu);
    }


    TEST(grad1conv_1D, verylarge){
        test_grad1conv<__TYPE__,GpuConv1D> test_verylarge(700001);

        EXPECT_AllCLOSE<__TYPE__>(test_verylarge.vresgrad,test_verylarge.vresgpu, ATOL, RTOL);
        EXPECT_NONZEROS<__TYPE__>(test_verylarge.vresgrad);
        EXPECT_NONZEROS<__TYPE__>(test_verylarge.vresgpu);
    }

    TEST(grad1conv_2D, small){

        test_grad1conv<__TYPE__,GpuConv2D> test_small(2001);

        EXPECT_AllCLOSE<__TYPE__>(test_small.vresgrad,test_small.vresgpu, ATOL, RTOL);
        EXPECT_NONZEROS<__TYPE__>(test_small.vresgrad);
        EXPECT_NONZEROS<__TYPE__>(test_small.vresgpu);
    }

    TEST(grad1conv_2D, medium){

        test_grad1conv<__TYPE__,GpuConv2D> test_medium(20001);

        EXPECT_AllCLOSE<__TYPE__>(test_medium.vresgrad,test_medium.vresgpu, ATOL, RTOL);
        EXPECT_NONZEROS<__TYPE__>(test_medium.vresgrad);
        EXPECT_NONZEROS<__TYPE__>(test_medium.vresgpu);
    }


    TEST(grad1conv_2D, large){
        test_grad1conv<__TYPE__,GpuConv2D> test_large(200001);

        EXPECT_AllCLOSE<__TYPE__>(test_large.vresgrad,test_large.vresgpu, ATOL, RTOL);
        EXPECT_NONZEROS<__TYPE__>(test_large.vresgrad);
        EXPECT_NONZEROS<__TYPE__>(test_large.vresgpu);
    }


    TEST(grad1conv_2D, verylarge){
        test_grad1conv<__TYPE__,GpuConv2D> test_verylarge(700001);

        EXPECT_AllCLOSE<__TYPE__>(test_verylarge.vresgrad,test_verylarge.vresgpu, ATOL, RTOL);
        EXPECT_NONZEROS<__TYPE__>(test_verylarge.vresgrad);
        EXPECT_NONZEROS<__TYPE__>(test_verylarge.vresgpu);
    }

}  // namespace



GTEST_API_ int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
