// test convolution using factorized formula
// compile with
//		nvcc -I.. -D__TYPE__=float -Wno-deprecated-gpu-targets -DCUDA_BLOCK_SIZE=192 -DMAXTHREADSPERBLOCK0=1024 -DSHAREDMEMPERBLOCK0=49152 -std=c++14 -O2 -o build/test_factorized test_factorized.cu

// we define an arbitrary function F,
// then use a factorized version FF of the same function and test

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <vector>
#include <ctime>
#include <algorithm>

#include "keops_includes.h"

using namespace keops;

__TYPE__ floatrand() {
    return ((__TYPE__) std::rand())/RAND_MAX-.5;    // random value between -.5 and .5
}

template < class V > void fillrandom(V& v) {
    generate(v.begin(), v.end(), floatrand);    // fills vector with random values
}

int main() {
	
	std::cout << std::endl << "N.B. AutoFactorize is deactivated in all code as of oct 2020 to speed up compile time." << std::endl << std::endl;
	
	/*
    // In this part we define the symbolic variables of the function
    using X = Var<1,3,0>; 	// X is the second variable and represents a 3D vector
    using Y = Var<2,3,1>; 	// Y is the third variable and represents a 3D vector
    using U = Var<3,4,0>; 	// U is the fourth variable and represents a 4D vector
    using V = Var<4,4,1>; 	// V is the fifth variable and represents a 4D vector
    using Beta = Var<5,3,1>;	// Beta is the sixth variable and represents a 3D vector
    using C = Param<0,1>;		// C is the first variable and is a scalar parameter

    // symbolic expression of the function ------------------------------------------------------

    // here we define F to be F0+F0+F0+F0+F0+F0+F0+F0 where F0 = 1/(C+|X-Y|^2) in usual notations
    // with the standard implementation it means we will compute 8 times F0 to evaluate F
    using F0 = decltype(InvKeopsNS(  Inv( IntCst(1) + KeopsNS<C>()* SqNorm2(KeopsNS<X>()-KeopsNS<Y>())))  );//Scal<Exp<Scal<C,Minus<SqNorm2<Subtract<X,Y>>>>>,Beta>;
    using F1 = Add<F0,F0>;
    using F = Grad<F0, X, Var<6,1,0>>;

    std::cout << std::endl << "Function F : " << std::endl;
    std::cout << PrintFormula<F>();
    std::cout << std::endl << std::endl;

    // now we factorize F0 from F : new formula FF computes the same as F but will evaluate first F0 once and then just does three vector additions
    using FF = AutoFactorize < F >;

    std::cout << "Function FF = factorized version of F :" << std::endl;
    std::cout << "Factor = " << std::endl;
    std::cout << PrintFormula<FF::Factor>();
    std::cout << std::endl << "Factorized Formula = " << std::endl;
    //using INDS = pack<0,1,2,3,4,5>; // just to print the formula we define a dummy INDS...
    using INDS = pack<0,1,2,3,4,5,6>; // just to print the formula we define a dummy INDS...
    std::cout << PrintFormula<FF::FactorizedFormula<INDS>>();
    std::cout << std::endl << std::endl;


    using FUNCONVF = Sum_Reduction<F>;


    // now we test ------------------------------------------------------------------------------

    std::cout << std::endl << "Testing F" << std::endl;

    int Nx=1511, Ny=1001;
    __TYPE__ s;

    std::vector<__TYPE__> vf(Nx*FUNCONVF::DIM);    fillrandom(vf); __TYPE__ *f = vf.data();
    std::vector<__TYPE__> vx(Nx*X::DIM);    fillrandom(vx); __TYPE__ *x = vx.data();
    std::vector<__TYPE__> vy(Ny*Y::DIM);    fillrandom(vy); __TYPE__ *y = vy.data();
    std::vector<__TYPE__> vu(Nx*U::DIM);    fillrandom(vu); __TYPE__ *u = vu.data();
    std::vector<__TYPE__> vv(Ny*V::DIM);    fillrandom(vv); __TYPE__ *v = vv.data();
    std::vector<__TYPE__> vb(Ny*Beta::DIM); fillrandom(vb); __TYPE__ *b = vb.data();

    std::vector<__TYPE__> resgpu1(Nx*FUNCONVF::DIM), resgpu2(Nx*FUNCONVF::DIM), rescpu(Nx*FUNCONVF::DIM);

    __TYPE__ params[1];
    __TYPE__ Sigma = 1;
    params[0] = 1.0/(Sigma*Sigma);

    clock_t begin, end;

    begin = clock();
    int deviceID = 0;
    cudaSetDevice(deviceID);
    end = clock();
    std::cout << "time for GPU initialization : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    begin = clock();
    Eval<FUNCONVF,GpuConv1D_FromHost>::Run(Nx, Ny, f, params, x, y, u, v, b, x);
    end = clock();
    std::cout << "time for GPU computation (first run) : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    resgpu1 = vf;
    fillrandom(vf);

    begin = clock();
    Eval<FUNCONVF,GpuConv2D_FromHost>::Run(Nx, Ny, f, params, x, y, u, v, b, x);
    end = clock();
    std::cout << "time for GPU computation (second run) : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    resgpu2 = vf;
    fillrandom(vf);
    
    if(Nx*Ny<1e8) {
        begin = clock();
        Eval<FUNCONVF,CpuConv>::Run(Nx, Ny, f, params, x, y, u, v, b, x);
        end = clock();
        std::cout << "time for CPU computation : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

        rescpu = vf;
        fillrandom(vf);

        // display values
        std::cout << std::endl << "resgpu1 = ";
        for(int i=0; i<5; i++)
                    std::cout << resgpu1[i] << " ";
        std::cout << std::endl << "resgpu2 = ";
        for(int i=0; i<5; i++)
                    std::cout << resgpu2[i] << " ";
        std::cout << std::endl << "rescpu  = ";
            for(int i=0; i<5; i++)
                    std::cout << rescpu[i] << " ";

        // display mean of errors
        s = 0;
        for(int i=0; i<Nx*FUNCONVF::DIM; i++)
            s += std::abs(resgpu1[i]-rescpu[i]);
        std::cout << std::endl << "mean abs error (cpu vs gpu1) =" << s/Nx << std::endl;
        s = 0;
        for(int i=0; i<Nx*FUNCONVF::DIM; i++)
            s += std::abs(resgpu2[i]-rescpu[i]);
        std::cout << "mean abs error (cpu vs gpu2) =" << s/Nx << std::endl;
        s = 0;
    }

    for(int i=0; i<Nx*FUNCONVF::DIM; i++)
        s += std::abs(resgpu1[i]-resgpu2[i]);
    std::cout << "mean abs error (gpu1 vs gpu2) =" << s/Nx << std::endl;



/// testing FF

    std::cout << std::endl << std::endl << "Testing FF" << std::endl;

    using FUNCONVFF = Sum_Reduction<FF>;

    begin = clock();
    Eval<FUNCONVFF,GpuConv1D_FromHost>::Run(Nx, Ny, f, params, x, y, u, v, b, x);
    end = clock();
    std::cout << "time for GPU computation (first run) : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    resgpu1 = vf;
    fillrandom(vf);

    begin = clock();
    Eval<FUNCONVFF,GpuConv2D_FromHost>::Run(Nx, Ny, f, params, x, y, u, v, b, x);
    end = clock();
    std::cout << "time for GPU computation (second run) : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    resgpu2 = vf;
    fillrandom(vf);

    if(Nx*Ny<1e8) {
        begin = clock();
        Eval<FUNCONVFF,CpuConv>::Run(Nx, Ny, f, params, x, y, u, v, b, x);
        end = clock();
        std::cout << "time for CPU computation : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

        rescpu = vf;
        fillrandom(vf);

        // display values
        std::cout << std::endl << "resgpu1 = ";
        for(int i=0; i<5; i++)
                    std::cout << resgpu1[i] << " ";
        std::cout << std::endl << "resgpu2 = ";
        for(int i=0; i<5; i++)
                    std::cout << resgpu2[i] << " ";
        std::cout << std::endl << "rescpu  = ";
            for(int i=0; i<5; i++)
                    std::cout << rescpu[i] << " ";


        // display mean of errors
        s = 0;
        for(int i=0; i<Nx*FUNCONVF::DIM; i++)
            s += std::abs(resgpu1[i]-rescpu[i]);
        std::cout << std::endl << "mean abs error (cpu vs gpu1) =" << s/Nx << std::endl;
        s = 0;
        for(int i=0; i<Nx*FUNCONVF::DIM; i++)
            s += std::abs(resgpu2[i]-rescpu[i]);
        std::cout << "mean abs error (cpu vs gpu2) =" << s/Nx << std::endl;
    }
    s = 0;
    for(int i=0; i<Nx*FUNCONVF::DIM; i++)
        s += std::abs(resgpu1[i]-resgpu2[i]);
    std::cout << "mean abs error (gpu1 vs gpu2) =" << s/Nx << std::endl;
	
	*/

    return 0 ;
}



