// test convolution using factorized formula
// compile with
//		nvcc -std=c++11 -O2 -o build/test_factorized test_factorized.cu

// we define an arbitrary function F,
// then use a factorized version FF of the same function and test

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <vector>
#include <ctime>
#include <algorithm>

#include "core/formulas/constants.h"
#include "core/formulas/maths.h"
#include "core/formulas/kernels.h"
#include "core/formulas/norms.h"
#include "core/formulas/factorize.h"

#include "../core/GpuConv1D.cu"
#include "../core/GpuConv2D.cu"
#include "../core/CpuConv.cpp"

#include "../core/autodiff.h"

using namespace std;



__TYPE__ floatrand() {
    return ((__TYPE__)rand())/RAND_MAX-.5;    // random value between -.5 and .5
}

template < class V > void fillrandom(V& v) {
    generate(v.begin(), v.end(), floatrand);    // fills vector with random values
}

int main() {

    // In this part we define the symbolic variables of the function
    using X = Var<0,3,0>; 	// X is the first variable and represents a 3D vector
    using Y = Var<1,3,1>; 	// Y is the second variable and represents a 3D vector
    using U = Var<2,4,0>; 	// U is the third variable and represents a 4D vector
    using V = Var<3,4,1>; 	// V is the fourth variable and represents a 4D vector
    using Beta = Var<4,3,1>;	// Beta is the fifth variable and represents a 3D vector
    using C = Param<0>;		// C is the first extra parameter

    // symbolic expression of the function ------------------------------------------------------

    // here we define F to be F0+F0+F0+F0+F0+F0+F0+F0 where F0 = <U,V>^2 * exp(-C*|X-Y|^2) * Beta in usual notations
    // with the standard implementation it means we will compute 8 times F0 to evaluate F
    using F0 = Scal<Exp<Scal<Constant<C>,Minus<SqNorm2<Subtract<X,Y>>>>>,Beta>;
    using F1 = Add<F0,F0>;
    using F = Add<F1,F1>;

    cout << endl << "Function F : " << endl;
    F::PrintId();
    cout << endl << endl;

    // now we factorize F0 from F : new formula FF computes the same as F but will evaluate first F0 once and then just does three vector additions
    using FF = Factorize < F, F0 >;

    cout << "Function FF = factorized version of F :" << endl;
    cout << "Factor = " << endl;
    FF::Factor::PrintId();
    cout << endl << "Factorized Formula = " << endl;
    using INDS = pack<0,1,2,3,4>; // just to print the formula we define a dummy INDS...
    FF::FactorizedFormula<INDS>::PrintId();
    cout << endl << endl;


    using FUNCONVF = typename Generic<F>::sEval;


    // now we test ------------------------------------------------------------------------------

    cout << endl << "Testing F" << endl;

    int Nx=211, Ny=201;
    __TYPE__ s;

    vector<__TYPE__> vf(Nx*F::DIM);    fillrandom(vf); __TYPE__ *f = vf.data();
    vector<__TYPE__> vx(Nx*X::DIM);    fillrandom(vx); __TYPE__ *x = vx.data();
    vector<__TYPE__> vy(Ny*Y::DIM);    fillrandom(vy); __TYPE__ *y = vy.data();
    vector<__TYPE__> vu(Nx*U::DIM);    fillrandom(vu); __TYPE__ *u = vu.data();
    vector<__TYPE__> vv(Ny*V::DIM);    fillrandom(vv); __TYPE__ *v = vv.data();
    vector<__TYPE__> vb(Ny*Beta::DIM); fillrandom(vb); __TYPE__ *b = vb.data();

    vector<__TYPE__> resgpu1(Nx*F::DIM), resgpu2(Nx*F::DIM), rescpu(Nx*F::DIM);

    __TYPE__ params[1];
    __TYPE__ Sigma = 1;
    params[0] = 1.0/(Sigma*Sigma);

    clock_t begin, end;

    begin = clock();
    int deviceID = 0;
    cudaSetDevice(deviceID);
    end = clock();
    cout << "time for GPU initialization : " << double(end - begin) / CLOCKS_PER_SEC << endl;

    begin = clock();
    GpuConv1D(FUNCONVF(), params, Nx, Ny, f, x, y, u, v, b);
    end = clock();
    cout << "time for GPU computation (first run) : " << double(end - begin) / CLOCKS_PER_SEC << endl;

    resgpu1 = vf;
    fillrandom(vf);

    begin = clock();
    GpuConv2D(FUNCONVF(), params, Nx, Ny, f, x, y, u, v, b);
    end = clock();
    cout << "time for GPU computation (second run) : " << double(end - begin) / CLOCKS_PER_SEC << endl;

    resgpu2 = vf;
    fillrandom(vf);
    
    if(Nx*Ny<1e8) {
        begin = clock();
        CpuConv(FUNCONVF(), params, Nx, Ny, f, x, y, u, v, b);
        end = clock();
        cout << "time for CPU computation : " << double(end - begin) / CLOCKS_PER_SEC << endl;

        rescpu = vf;
        fillrandom(vf);

        // display values
        cout << endl << "resgpu1 = ";
        for(int i=0; i<5; i++)
                    cout << resgpu1[i] << " ";
        cout << endl << "resgpu2 = ";
        for(int i=0; i<5; i++)
                    cout << resgpu2[i] << " ";
        cout << endl << "rescpu  = ";
            for(int i=0; i<5; i++)
                    cout << rescpu[i] << " ";

        // display mean of errors
        s = 0;
        for(int i=0; i<Nx*F::DIM; i++)
            s += abs(resgpu1[i]-rescpu[i]);
        cout << endl << "mean abs error (cpu vs gpu1) =" << s/Nx << endl;
        s = 0;
        for(int i=0; i<Nx*F::DIM; i++)
            s += abs(resgpu2[i]-rescpu[i]);
        cout << "mean abs error (cpu vs gpu2) =" << s/Nx << endl;
    }

    // display mean of errors
    s = 0;
    for(int i=0; i<Nx*F::DIM; i++)
        s += abs(resgpu1[i]-resgpu2[i]);
    cout << "mean abs error (gpu1 vs gpu2) =" << s/Nx << endl;



/// testing FF

    cout << endl << endl << "Testing FF" << endl;

    using FUNCONVFF = typename Generic<FF>::sEval;

    begin = clock();
    GpuConv1D(FUNCONVFF(), params, Nx, Ny, f, x, y, u, v, b);
    end = clock();
    cout << "time for GPU computation (first run) : " << double(end - begin) / CLOCKS_PER_SEC << endl;

    resgpu1 = vf;
    fillrandom(vf);

    begin = clock();
    GpuConv2D(FUNCONVFF(), params, Nx, Ny, f, x, y, u, v, b);
    end = clock();
    cout << "time for GPU computation (second run) : " << double(end - begin) / CLOCKS_PER_SEC << endl;

    resgpu2 = vf;
    fillrandom(vf);

    if(Nx*Ny<1e8) {
        begin = clock();
        CpuConv(FUNCONVFF(), params, Nx, Ny, f, x, y, u, v, b);
        end = clock();
        cout << "time for CPU computation : " << double(end - begin) / CLOCKS_PER_SEC << endl;

        rescpu = vf;
        fillrandom(vf);

        // display values
        cout << endl << "resgpu1 = ";
        for(int i=0; i<5; i++)
                    cout << resgpu1[i] << " ";
        cout << endl << "resgpu2 = ";
        for(int i=0; i<5; i++)
                    cout << resgpu2[i] << " ";
        cout << endl << "rescpu  = ";
            for(int i=0; i<5; i++)
                    cout << rescpu[i] << " ";


        // display mean of errors
        s = 0;
        for(int i=0; i<Nx*F::DIM; i++)
            s += abs(resgpu1[i]-rescpu[i]);
        cout << endl << "mean abs error (cpu vs gpu1) =" << s/Nx << endl;
        s = 0;
        for(int i=0; i<Nx*F::DIM; i++)
            s += abs(resgpu2[i]-rescpu[i]);
        cout << "mean abs error (cpu vs gpu2) =" << s/Nx << endl;
    }

    // display mean of errors
    s = 0;
    for(int i=0; i<Nx*F::DIM; i++)
        s += abs(resgpu1[i]-resgpu2[i]);
    cout << "mean abs error (gpu1 vs gpu2) =" << s/Nx << endl;

}



