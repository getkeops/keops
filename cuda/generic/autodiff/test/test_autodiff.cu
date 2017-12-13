// test convolution with autodiff
// compile with
//		nvcc -std=c++11 -O2 -o test_autodiff test_autodiff.cu

// we define an arbitrary function using available blocks,
// then test its convolution on the GPU, then get its gradient and test again the convolution

// Here we build the function F(x,y,u,v,beta) = <u,v>^2 * exp(-C*|x-y|^2) * beta
// where x, y, beta are 3D vectors, and u, v are 4D vectors
// and the convolution is gamma_i = sum_j F(x_i,y_j,u_i,v_j,beta_j)
// then we define G(x,y,u,v,beta,eta) = gradient of F with respect to x, with new input variable eta (3D)
// and the new convolution is gamma_i = sum_j G(x_i,y_j,u_i,v_j,beta_j,eta_i)

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <vector>
#include <ctime>
#include <algorithm>

#include "../core/GpuConv2D.cu"
#include "../core/autodiff.h"

using namespace std;



float floatrand() {
    return ((float)rand())/RAND_MAX-.5;    // random value between -.5 and .5
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

    // symbolic expression of the function
    
    // here we define F = <U,V>^2 * exp(-C*|X-Y|^2) * Beta in usual notations
    using F = Scal<Square<Scalprod<U,V>>, Scal<Exp<Scal<Constant<C>,Minus<SqNorm2<Subtract<X,Y>>>>>,Beta>>;

    using FUNCONVF = typename Generic<F>::sEval;

    // gradient with respect to X
    using Eta = Var<5,F::DIM,0>;	// new variable is in sixth position and is input of gradient
    using GX = Grad<F,X,Eta>;

    using FUNCONVGX = typename Generic<GX>::sEval;

    // gradient with respect to Y
    using GY = Grad<F,Y,Eta>;

    using FUNCONVGY = typename Generic<GY,1>::sEval;	// parameter 1 after GY means i and j variables must be swapped

    // now we test

    int Nx=5000, Ny=2000;

    vector<float> vf(Nx*F::DIM);
    fillrandom(vf);
    float *f = vf.data();
    vector<float> vx(Nx*X::DIM);
    fillrandom(vx);
    float *x = vx.data();
    vector<float> vy(Ny*Y::DIM);
    fillrandom(vy);
    float *y = vy.data();
    vector<float> vu(Nx*U::DIM);
    fillrandom(vu);
    float *u = vu.data();
    vector<float> vv(Ny*V::DIM);
    fillrandom(vv);
    float *v = vv.data();
    vector<float> vb(Ny*Beta::DIM);
    fillrandom(vb);
    float *b = vb.data();

    vector<float> resgpu(Nx*F::DIM), rescpu(Nx*F::DIM);

    float params[1];
    float Sigma = 1;
    params[0] = 1.0/(Sigma*Sigma);

    clock_t begin, end;

    begin = clock();
    int deviceID = 0;
    cudaSetDevice(deviceID);
    end = clock();
    cout << "time for GPU initialization : " << double(end - begin) / CLOCKS_PER_SEC << endl;

    cout << "testing function F" << endl;
    begin = clock();
    GpuConv2D(FUNCONVF(), params, Nx, Ny, f, x, y, u, v, b);
    end = clock();
    cout << "time for GPU computation (first run) : " << double(end - begin) / CLOCKS_PER_SEC << endl;

    begin = clock();
    GpuConv2D(FUNCONVF(), params, Nx, Ny, f, x, y, u, v, b);
    end = clock();
    cout << "time for GPU computation (second run) : " << double(end - begin) / CLOCKS_PER_SEC << endl;

    resgpu = vf;

    begin = clock();
    CpuConv(FUNCONVF(), params, Nx, Ny, f, x, y, u, v, b);
    end = clock();
    cout << "time for CPU computation : " << double(end - begin) / CLOCKS_PER_SEC << endl;

    rescpu = vf;

    // display mean of errors
    float s = 0;
    for(int i=0; i<Nx*F::DIM; i++)
        s += abs(resgpu[i]-rescpu[i]);
    cout << "mean abs error =" << s/Nx << endl;





    vector<float> ve(Nx*Eta::DIM);
    fillrandom(ve);
    float *e = ve.data();

    cout << "testing function GX" << endl;
    begin = clock();
    GpuConv2D(FUNCONVGX(), params, Nx, Ny, f, x, y, u, v, b, e);
    end = clock();
    cout << "time for GPU computation (first run) : " << double(end - begin) / CLOCKS_PER_SEC << endl;

    begin = clock();
    GpuConv2D(FUNCONVGX(), params, Nx, Ny, f, x, y, u, v, b, e);
    end = clock();
    cout << "time for GPU computation (second run) : " << double(end - begin) / CLOCKS_PER_SEC << endl;

    resgpu = vf;

    begin = clock();
    CpuConv(FUNCONVGX(), params, Nx, Ny, f, x, y, u, v, b, e);
    end = clock();
    cout << "time for CPU computation : " << double(end - begin) / CLOCKS_PER_SEC << endl;

    rescpu = vf;

    // display mean of errors
    s = 0;
    for(int i=0; i<Nx*GX::DIM; i++)
        s += abs(resgpu[i]-rescpu[i]);
    cout << "mean abs error =" << s/Nx << endl;



    // gradient wrt Y, which is a "j" variable.

    rescpu.resize(Ny*GY::DIM);
    resgpu.resize(Ny*GY::DIM);
    vf.resize(Ny*GY::DIM);
    f = vf.data();

    cout << "testing function GY" << endl;
    begin = clock();
    GpuConv2D(FUNCONVGY(), params, Ny, Nx, f, x, y, u, v, b, e);
    end = clock();
    cout << "time for GPU computation (first run) : " << double(end - begin) / CLOCKS_PER_SEC << endl;

    begin = clock();
    GpuConv2D(FUNCONVGY(), params, Ny, Nx, f, x, y, u, v, b, e);
    end = clock();
    cout << "time for GPU computation (second run) : " << double(end - begin) / CLOCKS_PER_SEC << endl;

    resgpu = vf;

    begin = clock();
    CpuConv(FUNCONVGY(), params, Ny, Nx, f, x, y, u, v, b, e);
    end = clock();
    cout << "time for CPU computation : " << double(end - begin) / CLOCKS_PER_SEC << endl;

    rescpu = vf;

    // display mean of errors
    s = 0;
    for(int i=0; i<Ny*GY::DIM; i++)
        s += abs(resgpu[i]-rescpu[i]);
    cout << "mean abs error =" << s/Ny << endl;



}



