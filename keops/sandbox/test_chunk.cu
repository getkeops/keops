// test convolution 
// compile with
//		nvcc -I.. -DMAXTHREADSPERBLOCK0=1024 -DSHAREDMEMPERBLOCK0=49152 -Wno-deprecated-gpu-targets -std=c++14 -O3 -arch=sm_61 -o build/test_chunk -DCUDA_BLOCK_SIZE=192 -Xptxas="-v" -DINDIM=1000 -DENABLECHUNK=1 test_chunk.cu

// we define an arbitrary function using available blocks,
// then test its convolution on the GPU

// Here we build the function f(x,y,u,v,beta) = <u,v>^2 * exp(-p*|x-y|^2) * beta
// where p is a scalar parameter, x, y, beta are 3D vectors, and u, v are 4D vectors
// and the convolution is res_i = sum_j f(x_i,y_j,u_i,v_j,beta_j)

#include <algorithm>
#include <keops_includes.h>

#include <chrono>
typedef std::chrono::high_resolution_clock Clock;

using namespace keops;

__TYPE__ floatrand() {
    return ((__TYPE__) std::rand())/RAND_MAX-.5;    // random value between -.5 and .5
}

template < class V > void fillrandom(V& v) {
    generate(v.begin(), v.end(), floatrand);    // fills vector with random values
}

// a function to display output of reduction
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

int main() {

    // In this part we define the symbolic variables of the function
    auto x = Vi(0,INDIM); 	 // x is the second variable and represents a 3D vector, "i"-indexed.
    auto y = Vj(1,INDIM); 	 // y is the third variable and represents a 3D vector, "j"-indexed.

    // symbolic expression of the function ------------------------------------------------------

    // here we define f = <u,v>^2 * exp(-p*|x-y|^2) * beta in usual notations
    auto f = Sum(x*y);
    
    // We define the reduction operation on f. Here a sum reduction, performed over the "j" index, and resulting in a "i"-indexed variable
    auto Sum_f = ArgKMin_Reduction(f,10,0);  // 0 means output of reduction will be "i"-indexed (0 means"i", 1 means "j")


    // now we test ------------------------------------------------------------------------------

    int Nx=10000, Ny=10000;

    // here we define actual data for all variables and feed it it with random values
    std::vector<__TYPE__> vx(Nx*x.DIM);    fillrandom(vx); __TYPE__ *px = vx.data();
    std::vector<__TYPE__> vy(Ny*y.DIM);    fillrandom(vy); __TYPE__ *py = vy.data();

    // also a vector for the output
    std::vector<__TYPE__> vres(Nx*Sum_f.DIM);    fillrandom(vres); __TYPE__ *pres = vres.data();

    std::cout << "testing Sum reduction" << std::endl;
clock_t begin, end;
begin = clock();
    EvalRed<GpuConv1D_FromHost>(Sum_f,Nx, Ny, pres, px, py);
end = clock();
std::cout << "time for run 1 : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
begin = clock();
    EvalRed<GpuConv1D_FromHost>(Sum_f,Nx, Ny, pres, px, py);
end = clock();
std::cout << "time for run 2 : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
begin = clock();
    EvalRed<GpuConv1D_FromHost>(Sum_f,Nx, Ny, pres, px, py);
end = clock();
std::cout << "time for run 3 : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
    std::cout << "output:" << std::endl;
    DispValues(pres,5,Sum_f.DIM);

}



