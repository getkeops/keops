// test convolution
// compile with
//		nvcc -I.. -DCUDA_BLOCK_SIZE=192 -DMAXTHREADSPERBLOCK0=1024 -DSHAREDMEMPERBLOCK0=49152 -Wno-deprecated-gpu-targets -std=c++14 --use_fast_math -O3 -o test_fromdevice test_fromdevice.cu

// testing "from device" convolution, i.e. convolution which is performed on the device
// directly from device data

#include <algorithm>
#include <thrust/device_vector.h>
#include <stdio.h>

#include <keops_includes.h>

#include <chrono>
typedef std::chrono::high_resolution_clock Clock;

#define DIMPOINT 3
#define DIMVECT 1

__TYPE__ floatrand() {
    return ((__TYPE__) std::rand())/RAND_MAX-.5;    // random value between -.5 and .5
}

template < class V > void fillrandom(V& v) {
    generate(v.begin(), v.end(), floatrand);    // fills vector with random values
}

void DispValues(__TYPE__ *x, int N, int dim) {
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

using namespace keops;

int main(int argc, char **argv) {

    int deviceID = 0;
    cudaSetDevice(deviceID);

    // symbolic expression of the function : a gaussian kernel
    auto x = Vi(0,DIMPOINT);
    auto y = Vj(1,DIMPOINT);
    auto beta = Vj(2,DIMVECT);
    
    auto f = Exp(-SqNorm2(x-y)) * beta; 

    std::cout << std::endl << "Function f : " << std::endl;
    std::cout << PrintFormula(f);
    std::cout << std::endl << std::endl;

    auto Sum_f = Sum_Reduction(f,0);

    // now we test ------------------------------------------------------------------------------

    int Nx;
    sscanf(argv[1], "%d", &Nx);
    
    std::vector<__TYPE__> vx(Nx*x.DIM);    fillrandom(vx); __TYPE__ *px = vx.data();
    thrust::device_vector<__TYPE__> vx_d(vx);
    __TYPE__ *x_d = thrust::raw_pointer_cast(vx_d.data());

    std::vector<__TYPE__> vy(Nx*DIMPOINT);    fillrandom(vy); __TYPE__ *py = vy.data();
    thrust::device_vector<__TYPE__> vy_d(vy);
    __TYPE__ *y_d = thrust::raw_pointer_cast(vy_d.data());
   
    std::vector<__TYPE__> vb(Nx*DIMVECT);     fillrandom(vb); __TYPE__ *pb = vb.data();
    thrust::device_vector<__TYPE__> vb_d(vb);
    __TYPE__ *b_d = thrust::raw_pointer_cast(vb_d.data());
   
    thrust::device_vector<__TYPE__> vres_d(Nx*Sum_f.DIM);
    __TYPE__ *res_d = thrust::raw_pointer_cast(vres_d.data());
    



    clock_t begin, end;

    std::cout << "blank run 1" << std::endl;
    begin = clock();
    EvalRed<GpuConv1D_FromDevice>(Sum_f,Nx, Nx, res_d, x_d, y_d, b_d);
    end = clock();
    std::cout << "time for blank run 1 : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    std::cout << "blank run 2" << std::endl;
    begin = clock();
    EvalRed<GpuConv1D_FromDevice>(Sum_f,Nx, Nx, res_d, x_d, y_d, b_d);
    end = clock();
    std::cout << "time for blank run 2 : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;




    int Ntest = 10;

    std::cout << "testing From_Device mode" << std::endl;

    //begin = clock();
    auto start = Clock::now();

    for(int i=0; i<Ntest; i++)
        EvalRed<GpuConv1D_FromDevice>(Sum_f,Nx, Nx, res_d, x_d, y_d, b_d);

    //end = clock();
    //std::cout << "time for "<< Ntest <<" GPU computations (1D scheme) : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
    auto stop = Clock::now();
    std::cout << "time = " 
                  << Ntest << "x "
                  << (float) std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count() / (float) (1000000 * Ntest)
                  << " milliseconds" << std::endl;

    std::vector<__TYPE__> resgpu1D(Nx*Sum_f.DIM);
    cudaMemcpy(resgpu1D.data(), res_d, Nx*Sum_f.DIM*sizeof(__TYPE__), cudaMemcpyDeviceToHost);





}



