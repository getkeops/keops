#include <iostream>
#include <gtest/gtest.h>

// use manual timing for GPU based functions
#include <chrono>
#include <ctime>

#include "bench/generate_data.h"
#include "keops_includes.h"

#define ATOL 1e-3
#define RTOL 1e-4

#define TEST_SIZE_SMALL 2001
#define TEST_SIZE_MEDIUM 20001
#define TEST_SIZE_LARGE 200001
#define TEST_SIZE_VERY_LARGE 700001 

using namespace keops;

template <typename T>
void EXPECT_AllCLOSE(const std::vector<T> X, const std::vector<T> Y, const T atol, const T rtol) {
    ASSERT_EQ(X.size(), Y.size());

    int count = 0;
    T l1norm = 0.0;
    for (unsigned int i = 0; i < X.size(); ++i) {
        if (std::abs(X[i] - Y[i])> atol + rtol * (std::abs(Y[i]) + std::abs(X[i])))
            count +=1;
        l1norm +=  std::abs(X[i] - Y[i]);
    }
        EXPECT_LE(count,0) << "number of non close values: " << count;
        EXPECT_LE(l1norm/X.size(), atol);
}


template <typename T>
void EXPECT_NONZEROS(const std::vector<T> X) {

    unsigned int nb_of_zeros = 0;
    for (unsigned int i = 0; i < X.size(); ++i) {
        if (std::abs(X[i]) == 0)
            nb_of_zeros +=1;
    }
        EXPECT_LT(nb_of_zeros,X.size());
}

/////////////////////////////////////////////////////////////////////////////////////
//                      The function to be benchmarked                            //
/////////////////////////////////////////////////////////////////////////////////////

auto formula0 = Grad(GaussKernel(Pm(0,1), Vi(1,3), Vj(2,3), Vj(3,3)), Vi(1,3), Vi(4,3));
using F0 = decltype(InvKeopsNS(formula0));

using FUN0 = Sum_Reduction<F0>;

extern "C" int GaussGpuEval(__TYPE__ ooSigma2, __TYPE__* alpha_h, __TYPE__* x_h, __TYPE__* y_h, __TYPE__* beta_h, __TYPE__* gamma_h, int dimPoint, int dimVect, int nx, int ny) ;

typedef int(*OP)(FUN0, int, int, __TYPE__*, __TYPE__*, __TYPE__*, __TYPE__*, __TYPE__*, __TYPE__*);

template <typename T, OP op>
class test_grad1conv {

    public:
    test_grad1conv(int);

    std::vector<T> vresgpu, vresgrad;
    T *resgpu, *resgrad;
    data<T> data1;
};

template <typename T, OP op>
test_grad1conv<T,op>::test_grad1conv(int Nx):data1(data<T>(Nx)){

        vresgpu.resize(Nx*data1.dimPoint);  resgpu = vresgpu.data(); 
        vresgrad.resize(Nx*data1.dimPoint); resgrad = vresgrad.data(); 

        GaussGpuEval(data1.params[0], data1.u, data1.x, data1.y, data1.v, resgpu, data1.dimPoint,data1.dimVect,data1.Nx,data1.Ny); 
        op(FUN0(), data1.Nx, data1.Ny, resgrad, data1.params, data1.x, data1.y, data1.v, data1.u);
}


namespace {

// ----- the following pragma suppress "function was declared but never referenced warning" : it is no longer needed with nvcc >=9.2
#pragma diag_suppress 177 // https://stackoverflow.com/questions/49836419/how-to-hide-nvccs-function-was-declared-but-never-referenced-warnings

    TEST(grad1conv_1D, small){

        test_grad1conv<__TYPE__, keops::GpuConv1D_FromHost::Eval> test_small(TEST_SIZE_SMALL);

        EXPECT_AllCLOSE<__TYPE__>(test_small.vresgrad,test_small.vresgpu, ATOL, RTOL);
        EXPECT_NONZEROS<__TYPE__>(test_small.vresgrad);
        EXPECT_NONZEROS<__TYPE__>(test_small.vresgpu);
    }

    TEST(grad1conv_1D, medium){

        test_grad1conv<__TYPE__, keops::GpuConv1D_FromHost::Eval> test_medium(TEST_SIZE_MEDIUM);

        EXPECT_AllCLOSE<__TYPE__>(test_medium.vresgrad,test_medium.vresgpu, ATOL, RTOL);
        EXPECT_NONZEROS<__TYPE__>(test_medium.vresgrad);
        EXPECT_NONZEROS<__TYPE__>(test_medium.vresgpu);
    }


    TEST(grad1conv_1D, large){
        test_grad1conv<__TYPE__, keops::GpuConv1D_FromHost::Eval> test_large(TEST_SIZE_LARGE);

        EXPECT_AllCLOSE<__TYPE__>(test_large.vresgrad,test_large.vresgpu, ATOL, RTOL);
        EXPECT_NONZEROS<__TYPE__>(test_large.vresgrad);
        EXPECT_NONZEROS<__TYPE__>(test_large.vresgpu);
    }


    TEST(grad1conv_1D, verylarge){
        test_grad1conv<__TYPE__, keops::GpuConv1D_FromHost::Eval> test_verylarge(TEST_SIZE_VERY_LARGE);

        EXPECT_AllCLOSE<__TYPE__>(test_verylarge.vresgrad,test_verylarge.vresgpu, ATOL, RTOL);
        EXPECT_NONZEROS<__TYPE__>(test_verylarge.vresgrad);
        EXPECT_NONZEROS<__TYPE__>(test_verylarge.vresgpu);
    }

    TEST(grad1conv_2D, small){

        test_grad1conv<__TYPE__, keops::GpuConv2D_FromHost::Eval> test_small(TEST_SIZE_SMALL);

        EXPECT_AllCLOSE<__TYPE__>(test_small.vresgrad,test_small.vresgpu, ATOL, RTOL);
        EXPECT_NONZEROS<__TYPE__>(test_small.vresgrad);
        EXPECT_NONZEROS<__TYPE__>(test_small.vresgpu);
    }

    TEST(grad1conv_2D, medium){

        test_grad1conv<__TYPE__, keops::GpuConv2D_FromHost::Eval> test_medium(TEST_SIZE_MEDIUM);

        EXPECT_AllCLOSE<__TYPE__>(test_medium.vresgrad,test_medium.vresgpu, ATOL, RTOL);
        EXPECT_NONZEROS<__TYPE__>(test_medium.vresgrad);
        EXPECT_NONZEROS<__TYPE__>(test_medium.vresgpu);
    }


    TEST(grad1conv_2D, large){
        test_grad1conv<__TYPE__, keops::GpuConv2D_FromHost::Eval> test_large(TEST_SIZE_LARGE);

        EXPECT_AllCLOSE<__TYPE__>(test_large.vresgrad,test_large.vresgpu, ATOL, RTOL);
        EXPECT_NONZEROS<__TYPE__>(test_large.vresgrad);
        EXPECT_NONZEROS<__TYPE__>(test_large.vresgpu);
    }


    TEST(grad1conv_2D, verylarge){
        test_grad1conv<__TYPE__, keops::GpuConv2D_FromHost::Eval> test_verylarge(TEST_SIZE_VERY_LARGE);

        EXPECT_AllCLOSE<__TYPE__>(test_verylarge.vresgrad,test_verylarge.vresgpu, ATOL, RTOL);
        EXPECT_NONZEROS<__TYPE__>(test_verylarge.vresgrad);
        EXPECT_NONZEROS<__TYPE__>(test_verylarge.vresgpu);
    }

}  // namespace



GTEST_API_ int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
