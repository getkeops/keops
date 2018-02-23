#include <iostream>
#include "bench/generate_data.h"

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////
//                      The function to be benchmarked                            //
/////////////////////////////////////////////////////////////////////////////////////

// Signature of the generic function:
extern "C" int GpuConv2D(__TYPE__*, int, int, __TYPE__*, __TYPE__**);

extern "C" int GpuConv1D(__TYPE__*, int, int, __TYPE__*, __TYPE__**);

extern "C" int GaussGpuGrad1Conv(__TYPE__ ooSigma2, __TYPE__* alpha_h, __TYPE__* x_h, __TYPE__* y_h, __TYPE__* beta_h, __TYPE__* gamma_h, int dimPoint, int dimVect, int nx, int ny) ;




/////////////////////////////////////////////////////////////////////////////////////
//                          Call the benchmark                                     //
/////////////////////////////////////////////////////////////////////////////////////


// The zeroth benchmark : simply to avoid warm up the GPU...
int main() {
        int Nx =100;
        
        data<__TYPE__> data1(Nx);
        data<__TYPE__> data3(Nx);
        data<__TYPE__> data2(Nx);

        vector<__TYPE__>  vf(Nx*data1.dimPoint);  __TYPE__ *f = vf.data(); 
        vector<__TYPE__> vf2(Nx*data1.dimPoint);  __TYPE__ *f2 = vf2.data(); 
        vector<__TYPE__> vf3(Nx*data1.dimPoint);  __TYPE__ *f3 = vf3.data(); 

    cout << Nx << endl;
    cout << data1.Nx << endl;
    cout << data1.Ny << endl;
    cout << data1.dimVect << endl;
    cout << data1.dimPoint << endl;
    cout << data1.x << endl;
    cout << data1.y << endl;
    cout << data1.u << endl;
    cout << data1.v << endl;
    cout << data1.f << endl;

        cout << "entering GpuConv1D" << endl;
        GpuConv1D(data1.params, data1.Nx, data1.Ny, data1.f, data1.args);
        cout << "entering GpuConv2D" << endl;
        GpuConv2D(data2.params, data2.Nx, data2.Ny, data2.f, data2.args);
        cout << "entering GaussGpuGrad1Conv" << endl;
        GaussGpuGrad1Conv(data3.params[0], data3.u, data3.x, data3.y, data3.v, data3.f, data3.dimPoint,data3.dimVect,Nx,data3.Ny); 


        __TYPE__ e=0;
        for (int i=0; i<Nx*data1.dimPoint; i++){
            e+= abs(f2[i] - f[i]) ;
        }
        cout << "Erreur : " << e << endl;

        return 0;
}
