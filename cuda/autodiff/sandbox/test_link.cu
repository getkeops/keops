// test convolution with autodiff
// compile with
//		nvcc -std=c++11 -O2 -c test_link.cu
//		./compile "Scal<Square<Scalprod<X<2,4>,Y<3,4>>>,GaussKernel<P<0>,X<0,3>,Y<1,3>,Y<4,3>>>"
// 		nvcc -o test_link test_link.o "build/Scal<Square<Scalprod<X<2,4>,Y<3,4>>>,GaussKernel<P<0>,X<0,3>,Y<1,3>,Y<4,3>>>___TYPE__.so"


#include <iostream>
#include <assert.h>
#include <cuda.h>
#include <vector>
#include <ctime>
#include <algorithm>

using namespace std;


extern "C" int GpuConv1D(__TYPE__*, int, int, __TYPE__*, __TYPE__**);
extern "C" int GpuConv2D(__TYPE__*, int, int, __TYPE__*, __TYPE__**);
extern "C" int CpuConv(__TYPE__*, int, int, __TYPE__*, __TYPE__**);

__TYPE__ __TYPE__rand() {
    return ((__TYPE__)rand())/RAND_MAX-.5;    // random value between -.5 and .5
}

template < class V > void fillrandom(V& v) {
    generate(v.begin(), v.end(), __TYPE__rand);    // fills vector with random values
}


__TYPE__ __TYPE__ones() {
    return (__TYPE__)1.0;
}
template < class V > void fillones(V& v) {
    generate(v.begin(), v.end(), __TYPE__ones );    // fills vector with random values
}

int main() {

    int Nx=5000, Ny=20000;

    vector<__TYPE__> vf(Nx*3);
    fillrandom(vf);
    __TYPE__ *f = vf.data();

    vector<__TYPE__> vx(Nx*3);
    fillrandom(vx);
    __TYPE__ *x = vx.data();
    
    vector<__TYPE__> vy(Ny*3);
    fillrandom(vy);
    __TYPE__ *y = vy.data();
    
    vector<__TYPE__> vu(Nx*4);
    fillrandom(vu);
    __TYPE__ *u = vu.data();
    
    vector<__TYPE__> vv(Ny*4);
    fillrandom(vv);
    __TYPE__ *v = vv.data();
    
    vector<__TYPE__> vb(Ny*3);
    fillones(vb);
    __TYPE__ *b = vb.data();

    vector<__TYPE__*> vargs(5);
    vargs[0]=x;
    vargs[1]=y;
    vargs[2]=u;
    vargs[3]=v;
    vargs[4]=b;
    __TYPE__ **args = vargs.data();

    vector<__TYPE__> resgpu2D(Nx*3), resgpu1D(Nx*3), rescpu(Nx*3);

    __TYPE__ params[1];
    __TYPE__ Sigma = 1;
    params[0] = 1.0/(Sigma*Sigma);

    clock_t begin, end;

    begin = clock();
    int deviceID = 0;
    cudaSetDevice(deviceID);
    end = clock();
    cout << "time for GPU initialization : " << double(end - begin) / CLOCKS_PER_SEC << endl;

    cout << endl;

    cout << "testing convolution with sizes :" << "nx=" << Nx <<" and ny=" << Ny <<endl << endl;

    begin = clock();
    GpuConv1D(params, Nx, Ny, f, args);
    end = clock();
    cout << "time for GPU computation (1D) : " << double(end - begin) / CLOCKS_PER_SEC << endl;

    resgpu1D = vf;
    fillones(vf);

    begin = clock();
    GpuConv2D(params, Nx, Ny, f, args);
    end = clock();
    cout << "time for GPU computation (2D) : " << double(end - begin) / CLOCKS_PER_SEC << endl;

    resgpu2D = vf;
    fillrandom(vf);

    begin = clock();
    CpuConv(params, Nx, Ny, f, args);
    end = clock();
    cout << "time for CPU computation : " << double(end - begin) / CLOCKS_PER_SEC << endl << endl;

    rescpu = vf;

    // display mean of errors
    __TYPE__ s = 0;
    for(int i=0; i<Nx*3; i++)
        s += abs(resgpu2D[i]-rescpu[i]);
    cout << "mean abs error 2D =" << s/Nx << endl;


    s = 0;
    for(int i=0; i<Nx*3; i++)
        s += abs(resgpu1D[i]-rescpu[i]);
    cout << "mean abs error 1D =" << s/Nx << endl;

    
    // display some values
    cout << endl << "Check visually the results : "<< endl;

    cout << "resgpu2D :" ;
    for (int i=0; i<10; i++)
        cout << resgpu2D[i] << " ";
    cout << "..." << endl;
    cout << "resgpu1D :" ;
    for (int i=0; i<10; i++)
        cout << resgpu1D[i] << " ";
    cout << "..." << endl;
    cout << "rescpu   :" ;
    for (int i=0; i<10; i++)
        cout << rescpu[i] << " ";
    cout << "..." << endl;



}



