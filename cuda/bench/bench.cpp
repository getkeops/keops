#include <iostream>
#include <benchmark/benchmark.h>

#include "bench/generate_data.h"

// use manual timing for GPU based functions
#include <chrono>
#include <ctime>

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////
//                      The function to be benchmarked                            //
/////////////////////////////////////////////////////////////////////////////////////

// Signature of the generic function:
extern "C" int GpuConv2D(__TYPE__*, int, int, __TYPE__*, __TYPE__**);

void main_generic_2D(int Nx) {
    data<__TYPE__> data1(Nx);
    GpuConv2D(data1.params, data1.Nx, data1.Ny, data1.f, data1.args);
}

// Signature of the generic function:
extern "C" int GpuConv1D(__TYPE__*, int, int, __TYPE__*, __TYPE__**);

void main_generic_1D(int Nx) {
    data<__TYPE__> data1(Nx);
    GpuConv1D(data1.params, data1.Nx, data1.Ny, data1.f, data1.args);
}

extern "C" int GaussGpuGrad1Conv(__TYPE__ ooSigma2, __TYPE__* alpha_h, __TYPE__* x_h, __TYPE__* y_h, __TYPE__* beta_h, __TYPE__* gamma_h, int dimPoint, int dimVect, int nx, int ny) ;

void main_specific(int Nx) {
    data<__TYPE__> data1(Nx);
    GaussGpuGrad1Conv(data1.params[0], data1.u, data1.x, data1.y, data1.v, data1.f, data1.dimPoint,data1.dimVect,data1.Nx,data1.Ny); 
}



/////////////////////////////////////////////////////////////////////////////////////
//                          Call the benchmark                                     //
/////////////////////////////////////////////////////////////////////////////////////


// The zeroth benchmark : simply to avoid warm up the GPU...
static void BM_dummy(benchmark::State& state) {
    for (auto _ : state) {
        int Nx =100;
        
        data<__TYPE__> data1(Nx);

        vector<__TYPE__>  vf(Nx*data1.dimPoint);  __TYPE__ *f1 = vf.data(); 
        vector<__TYPE__> vf2(Nx*data1.dimPoint);  __TYPE__ *f2 = vf2.data(); 
        vector<__TYPE__> vf3(Nx*data1.dimPoint);  __TYPE__ *f3 = vf3.data(); 

        GaussGpuGrad1Conv(data1.params[0], data1.u, data1.x, data1.y, data1.v, f3, data1.dimPoint,data1.dimVect,Nx,data1.Ny); 
        GpuConv2D(data1.params, data1.Nx, data1.Ny, f2, data1.args);
        GpuConv1D(data1.params, data1.Nx, data1.Ny, f1, data1.args);


        __TYPE__ e=0;
        for (int i=0; i<Nx*data1.dimPoint; i++){
            e+= abs(f2[i] - f1[i]) ;
        }
        cout << "Erreur : " << e << endl;
    }
}
BENCHMARK(BM_dummy);// Register the function as a benchmark


// A first Benchmark:
static void cuda_specific(benchmark::State& state) {
    int Nx = state.range(0);

    for (auto _ : state) {
        auto start = chrono::high_resolution_clock::now();
        //----------- the function to be benchmarked ------------//
        main_specific(Nx);
        //------------------------------------------------------//
        auto end   = chrono::high_resolution_clock::now();

        auto elapsed_seconds = chrono::duration_cast<chrono::duration<double>>( end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
}
// set range of the parameter to be tested : [ 8, 64, 512, 4k, 8k ]
BENCHMARK(cuda_specific)->Range(8, 8<<10)->UseManualTime();// Register the function as a benchmark

// A second one:
static void cuda_generic_2D(benchmark::State& state) {
    int Nx = state.range(0);

    for (auto _ : state) {
        auto start = chrono::high_resolution_clock::now();
        //----------- the function to be benchmarked ------------//
        main_generic_2D(Nx);
        //------------------------------------------------------//
        auto end   = chrono::high_resolution_clock::now();

        auto elapsed_seconds = chrono::duration_cast<chrono::duration<double>>( end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
}
// set range of the parameter to be tested : [ 8, 64, 512, 4k, 8k ]
BENCHMARK(cuda_generic_2D)->Range(8, 8<<10)->UseManualTime();// Register the function as a benchmark

// A third one:
static void cuda_generic_1D(benchmark::State& state) {
    int Nx = state.range(0);

    for (auto _ : state) {
        auto start = chrono::high_resolution_clock::now();
        //----------- the function to be benchmarked ------------//
        main_generic_1D(Nx);
        //------------------------------------------------------//
        auto end   = chrono::high_resolution_clock::now();

        auto elapsed_seconds = chrono::duration_cast<chrono::duration<double>>( end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
}
// set range of the parameter to be tested : [ 8, 64, 512, 4k, 8k ]
BENCHMARK(cuda_generic_1D)->Range(8, 8<<10)->UseManualTime();// Register the function as a benchmark

BENCHMARK_MAIN();// generate the benchmarks
