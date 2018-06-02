#include <iostream>
#include <benchmark/benchmark.h>

// use manual timing for GPU based functions
#include <chrono>
#include <ctime>

#include "core/formulas/newsyntax.h"

#include "bench/generate_data.h"

#include "core/GpuConv1D.cu"
#include "core/GpuConv2D.cu"
#include "core/CpuConv.cpp"

using namespace keops;

/////////////////////////////////////////////////////////////////////////////////////
//                      The function to be benchmarked                             //
/////////////////////////////////////////////////////////////////////////////////////
auto formula0 = Grad(GaussKernel(Pm(0,1),Vx(1,3),Vy(2,3),Vy(3,3)),Vx(1,3),Vx(4,3));
using F0 = decltype(formula0);
using F1 = F0;

using FUN0 = typename Generic<F0>::sEval;
using FUN1 = typename Generic<F1>::sEval;

extern "C" int GaussGpuEval(__TYPE__ ooSigma2, __TYPE__* alpha_h, __TYPE__* x_h, __TYPE__* y_h, __TYPE__* beta_h, __TYPE__* gamma_h, int dimPoint, int dimVect, int nx, int ny) ;


/////////////////////////////////////////////////////////////////////////////////////
//                                The Bench                                        //
/////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
void main_grad_1D(int Nx) {
    data<__TYPE__> data1(Nx);
    GpuConv1D(FUN0(), data1.Nx, data1.Ny, data1.f, data1.params, data1.x, data1.y, data1.v, data1.u);
}

static void cuda_grad_1D(benchmark::State& state) {
    int Nx = state.range(0);

    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        //----------- the function to be benchmarked ------------//
        main_grad_1D(Nx);
        //------------------------------------------------------//
        auto end   = std::chrono::high_resolution_clock::now();

        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>( end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
}
// set range of the parameter to be tested : [ 8, 64, 512, 4k, 8k ]
BENCHMARK(cuda_grad_1D)->Range(8, 8<<10)->UseManualTime();// Register the function as a benchmark




/////////////////////////////////////////////////////////////////////////////////////////////////
void main_grad_2D(int Nx) {
    data<__TYPE__> data1(Nx);
    GpuConv2D(FUN0(), data1.Nx, data1.Ny, data1.f, data1.params, data1.x, data1.y, data1.v, data1.u);
}

static void cuda_grad_2D(benchmark::State& state) {
    int Nx = state.range(0);

    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        //----------- the function to be benchmarked ------------//
        main_grad_2D(Nx);
        //------------------------------------------------------//
        auto end   = std::chrono::high_resolution_clock::now();

        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>( end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
}
// set range of the parameter to be tested : [ 8, 64, 512, 4k, 8k ]
BENCHMARK(cuda_grad_2D)->Range(8, 8<<10)->UseManualTime();// Register the function as a benchmark




/////////////////////////////////////////////////////////////////////////////////////////////////
void main_generic_1D(int Nx) {
    data<__TYPE__> data1(Nx);
    GpuConv1D(FUN1(), data1.Nx, data1.Ny, data1.f, data1.params, data1.x, data1.y, data1.v, data1.u);
}

static void cuda_generic_1D(benchmark::State& state) {
    int Nx = state.range(0);

    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        //----------- the function to be benchmarked ------------//
        main_generic_1D(Nx);
        //------------------------------------------------------//
        auto end   = std::chrono::high_resolution_clock::now();

        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>( end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
}
// set range of the parameter to be tested : [ 8, 64, 512, 4k, 8k ]
BENCHMARK(cuda_generic_1D)->Range(8, 8<<10)->UseManualTime();// Register the function as a benchmark




/////////////////////////////////////////////////////////////////////////////////////////////////
void main_generic_2D(int Nx) {
    data<__TYPE__> data1(Nx);
    GpuConv2D(FUN1(), data1.Nx, data1.Ny, data1.f, data1.params, data1.x, data1.y, data1.v, data1.u);
}

static void cuda_generic_2D(benchmark::State& state) {
    int Nx = state.range(0);

    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        //----------- the function to be benchmarked ------------//
        main_generic_2D(Nx);
        //------------------------------------------------------//
        auto end   = std::chrono::high_resolution_clock::now();

        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>( end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
}
// set range of the parameter to be tested : [ 8, 64, 512, 4k, 8k ]
BENCHMARK(cuda_generic_2D)->Range(8, 8<<10)->UseManualTime();// Register the function as a benchmark




/////////////////////////////////////////////////////////////////////////////////////////////////

void main_specific(int Nx) {
    data<__TYPE__> data1(Nx);
    GaussGpuEval(data1.params[0], data1.u, data1.x, data1.y, data1.v, data1.f, data1.dimPoint,data1.dimVect,data1.Nx,data1.Ny); 
}

static void cuda_specific(benchmark::State& state) {
    int Nx = state.range(0);

    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        //----------- the function to be benchmarked ------------//
        main_specific(Nx);
        //------------------------------------------------------//
        auto end   = std::chrono::high_resolution_clock::now();

        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>( end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
}
// set range of the parameter to be tested : [ 8, 64, 512, 4k, 8k ]
BENCHMARK(cuda_specific)->Range(8, 8<<10)->UseManualTime();// Register the function as a benchmark


BENCHMARK_MAIN();// generate the benchmarks

