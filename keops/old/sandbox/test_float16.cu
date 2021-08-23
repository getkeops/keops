// test convolution
// compile with
//		nvcc -I.. --gpu-architecture=compute_61 --gpu-code=sm_61 -DCUDA_BLOCK_SIZE=192 -DMAXTHREADSPERBLOCK0=1024 -DSHAREDMEMPERBLOCK0=49152 --use_fast_math -Wno-deprecated-gpu-targets -std=c++14 -O3 -o build/test_float16 test_float16.cu

// testing float16 convolution

#include <algorithm>
#include <thrust/device_vector.h>

#define __TYPE__ __half2
#define __TYPESTO__ __half
#define USE_DOUBLE 0
#define USE_HALF 1

#include <keops_includes.h>

#define DIMPOINT 3
#define DIMVECT 1

__TYPESTO__ floatrand() {
    return 1.5;//((__TYPESTO__) std::rand())/RAND_MAX-.5;    // random value between -.5 and .5
}

template < class V > void fillrandom(V& v) {
    generate(v.begin(), v.end(), floatrand);    // fills vector with random values
}

/*
void DispValues(__TYPESTO__ *x, int N, int dim) {
  std::cout << std::endl;
  int k = 0;
  for(int i=0; i<N; i++) {
    for(int d=0; d<dim; d++) {
      std::cout << x[k] << " ";
      k++;
    }
    std::cout << std::endl;
  }
  for(int d=0; d<dim; d++)
    std::cout << "... ";
  std::cout << std::endl << std::endl;
}
*/

using namespace keops;

int main() {

    int deviceID = 0;
    cudaSetDevice(deviceID);

    // symbolic expression of the function : a gaussian kernel
    auto c = Pm(0,1);
    auto x = Vi(1,DIMPOINT);
    auto y = Vj(2,DIMPOINT);
    auto beta = Vj(3,DIMVECT);
    
    auto f = Exp(-SqNorm2(x-y)) * beta; 

    std::cout << std::endl << "Function f : " << std::endl;
    std::cout << PrintFormula(f);
    std::cout << std::endl << std::endl;

    auto Sum_f = Sum_Reduction(f,0);

    // now we test ------------------------------------------------------------------------------

    int Nx=100000, Ny=100000;

    int Ntest = 1;

    std::vector<__TYPESTO__> vx(Nx*x.DIM);    fillrandom(vx); __TYPESTO__ *px = vx.data();
    thrust::device_vector<__TYPESTO__> vx_d(vx);
    __TYPE__ *x_d = (half2*)thrust::raw_pointer_cast(vx_d.data());

    std::vector<__TYPESTO__> vy(Ny*DIMPOINT);    fillrandom(vy); __TYPESTO__ *py = vy.data();
    thrust::device_vector<__TYPESTO__> vy_d(vy);
    __TYPE__ *y_d = (half2*)thrust::raw_pointer_cast(vy_d.data());
   
    std::vector<__TYPESTO__> vb(Ny*DIMVECT);     fillrandom(vb); __TYPESTO__ *pb = vb.data();
    thrust::device_vector<__TYPESTO__> vb_d(vb);
    __TYPE__ *b_d = (half2*)thrust::raw_pointer_cast(vb_d.data());
   
    thrust::device_vector<__TYPESTO__> vres_d(Nx*Sum_f.DIM);
    __TYPE__ *res_d = (half2*)thrust::raw_pointer_cast(vres_d.data());
    
    __TYPESTO__ param = 0.5;
    thrust::device_vector<__TYPESTO__> vparam_d(c.DIM);
    vparam_d[0] = param;
    __TYPE__ *param_d = (half2*)thrust::raw_pointer_cast(vparam_d.data());
    

    clock_t begin, end;

    std::cout << "blank run 1" << std::endl;
    begin = clock();
    EvalRed<GpuConv1D_FromDevice>(Sum_f,Nx/2, Ny/2, res_d, param_d, x_d, y_d, b_d);
    end = clock();
    std::cout << "time for blank run 1 : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    std::cout << "blank run 2" << std::endl;
    begin = clock();
    EvalRed<GpuConv1D_FromDevice>(Sum_f,Nx/2, Ny/2, res_d, param_d, x_d, y_d, b_d);
    end = clock();
    std::cout << "time for blank run 2 : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;


    std::cout << "testing From_Device mode" << std::endl;
    begin = clock();
    for(int i=0; i<Ntest; i++)
        EvalRed<GpuConv1D_FromDevice>(Sum_f,Nx/2, Ny/2, res_d, param_d, x_d, y_d, b_d);
    end = clock();
    std::cout << "time for " << Ntest << " GPU computations (1D scheme) : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
    std::vector<__TYPESTO__> resgpu1D(Nx*Sum_f.DIM);
    cudaMemcpy(resgpu1D.data(), res_d, Nx*Sum_f.DIM*sizeof(__TYPESTO__), cudaMemcpyDeviceToHost);

}



