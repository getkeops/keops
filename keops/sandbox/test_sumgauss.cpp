// test convolution
// compile with
//		g++ -I.. -D__TYPE__=float -std=c++11 -O2 -o build/test_sumgauss test_sumgauss.cpp
// 

#include <stdio.h>
#include <assert.h>
#include <vector>
#include <ctime>
#include <algorithm>
#include <iostream>

#include "core/formulas/constants.h"
#include "core/formulas/maths.h"
#include "core/formulas/kernels.h"
#include "core/formulas/norms.h"
#include "core/formulas/factorize.h"

#include "core/autodiff.h"

#include "core/CpuConv.cpp"

using namespace std;



__TYPE__ floatrand() {
    return ((__TYPE__)rand())/RAND_MAX-.5;    // random value between -.5 and .5
}

template < class V > void fillrandom(V& v) {
    generate(v.begin(), v.end(), floatrand);    // fills vector with random values
}

#define DIMPOINT 3
#define DIMVECT 3

int main() {

    // symbolic expression of the function : linear combination of 4 gaussians
    // F(Cs,Ws,X,Y,B) où Cs, Ws paramètres (vecteurs de taille 4)
    using F = SumGaussKernel<DIMPOINT,DIMVECT,4>;

    cout << endl << "Function F : " << endl;
    PrintFormula<F>();
    cout << endl << endl;

    using FUNCONVF = typename Generic<F>::sEval;

    // now we test ------------------------------------------------------------------------------

    cout << "Testing F" << endl;

    int Nx=5000, Ny=2000;
        
    vector<__TYPE__> vf(Nx*F::DIM);    fillrandom(vf); __TYPE__ *f = vf.data();
    vector<__TYPE__> vx(Nx*DIMPOINT);    fillrandom(vx); __TYPE__ *x = vx.data();
    vector<__TYPE__> vy(Ny*DIMPOINT);    fillrandom(vy); __TYPE__ *y = vy.data();
    vector<__TYPE__> vb(Ny*DIMVECT); fillrandom(vb); __TYPE__ *b = vb.data();

    vector<__TYPE__> rescpu1(Nx*F::DIM);

    __TYPE__ oos2s[4] = {.5,.25,.1,1.0};
    __TYPE__ weights[4] = {1.0,-2.0,-.5,3.2};

    clock_t begin, end;

    begin = clock();
    CpuConv(FUNCONVF(), Nx, Ny, f, oos2s, weights, x, y, b);
    end = clock();
    cout << "time for CPU computation : " << double(end - begin) / CLOCKS_PER_SEC << endl;

    rescpu1 = vf;


    // display values
    cout << "rescpu1 = ";
    for(int i=0; i<5; i++)
        cout << rescpu1[i] << " ";
    cout << "..." << endl << endl;
    
    // compare with combination of 4 convolutions
    cout << "Comparing with combination of 4 convolutions" << endl;
    using F0 = GaussKernel_<DIMPOINT,DIMVECT>;
    using FUNCONVF0 = typename Generic<F0>::sEval;
    vector<__TYPE__> vf0(Nx*F::DIM);    fillrandom(vf0); __TYPE__ *f0 = vf0.data();
    vector<__TYPE__> vf1(Nx*F::DIM);    fillrandom(vf1); __TYPE__ *f1 = vf1.data();
    vector<__TYPE__> vf2(Nx*F::DIM);    fillrandom(vf2); __TYPE__ *f2 = vf2.data();
    vector<__TYPE__> vf3(Nx*F::DIM);    fillrandom(vf3); __TYPE__ *f3 = vf3.data();
	vector<__TYPE__> rescpu2(Nx*F::DIM);
    begin = clock();
    CpuConv(FUNCONVF0(), Nx, Ny, f0, oos2s, x, y, b);
    CpuConv(FUNCONVF0(), Nx, Ny, f1, oos2s+1, x, y, b);
    CpuConv(FUNCONVF0(), Nx, Ny, f2, oos2s+2, x, y, b);
    CpuConv(FUNCONVF0(), Nx, Ny, f3, oos2s+3, x, y, b);
    for(int i=0; i<Nx*F::DIM; i++)
        f[i] = weights[0]*f0[i]+weights[1]*f1[i]+weights[2]*f2[i]+weights[3]*f3[i];
    end = clock();
    cout << "time for CPU computation : " << double(end - begin) / CLOCKS_PER_SEC << endl;
    rescpu2 = vf;

    // display values
    cout << "rescpu2 = ";
    for(int i=0; i<5; i++)
        cout << rescpu2[i] << " ";
    cout << "..." << endl;

    // display mean of errors
    __TYPE__ s = 0;
    for(int i=0; i<Nx*F::DIM; i++)
        s += abs(rescpu1[i]-rescpu2[i]);
    cout << endl << "mean abs error = " << s/Nx << endl << endl;

}



