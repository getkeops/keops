// Example of reductions using KeOps C++ / Cuda routines.
// This code performs the following computations :
//       res_i = Red_j F(i,j)
// where : 
//    - F is a Gaussian kernel F(i,j) = exp(-c*|x_i-y_j|^2)*b_j 
//    - the x_i are Nx=10000 points in R^3 
//    - the y_j are Ny=15000 points in R^3
//    - c is a scalar parameter
//    - the b_j are Ny vectors in R^2 
//    - Red_j represents a reduction operation, computed for each index i over the j indices
//        This reduction is vectorized, meaning that reduction is applied separately to each coordinate
//        of F(i,j).
//        Reductions tested hare are the following :
//        * Sum : returns the sum of F(i,j) values,
//        * Min : returns the minimum of F(i,j) values,
//        * ArgMin : returns the indices of minimal F(i,j) values,
//        * MinArgMin : returns minimal values and corresponding indices,
//        * ArgKMin : returns the indices of K first minimal values of F(i,j) (here K=3), 
//        * KMinArgKMin : returns values and indices of K first minimal values of F(i,j) (here K=3), 
//
// This program runs on CPU and GPU ; see the file test_reductions.cpp for the equivalent program on CPU only.
//
// This example can be compiled with the command
//		nvcc -I.. -DCUDA_BLOCK_SIZE=192 -DMAXTHREADSPERBLOCK0=1024 -DSHAREDMEMPERBLOCK0=49152 -Wno-deprecated-gpu-targets -std=c++14 -O3 -o build/test_reductions test_reductions.cu
// 

#include <algorithm>
#include <keops_includes.h>

using namespace keops;

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

#define DIMPOINT 3
#define DIMVECT 2

template < class RED > void DoTest(RED red, std::string red_id, int Nx, int Ny, __TYPE__ *pc, __TYPE__ *px, __TYPE__ *py, __TYPE__ *pb) {
    std::cout << "Testing " << red_id << " reduction" << std::endl;

    std::vector<__TYPE__> vres(Nx*RED::DIM);    fillrandom(vres); __TYPE__ *pres = vres.data();

    clock_t begin, end;

    begin = clock();
    EvalRed<CpuConv>(red, Nx, Ny, pres, pc, px, py, pb);
    end = clock();

    std::cout << "time for CPU computation : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    std::vector<__TYPE__> rescpu(Nx*RED::DIM);
    rescpu = vres;
    std::cout << "output = ";
    DispValues(rescpu.data(),5,RED::DIM);

    EvalRed<GpuConv1D_FromHost>(red, Nx, Ny, pres, pc, px, py, pb);	// first dummy call to Gpu

    begin = clock();
    EvalRed<GpuConv1D_FromHost>(red, Nx, Ny, pres, pc, px, py, pb);
    end = clock();

    std::cout << "time for GPU computation (1D scheme) : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    std::vector<__TYPE__> resgpu1(Nx*RED::DIM);
    resgpu1 = vres;
    std::cout << "output = ";
    DispValues(resgpu1.data(),5,RED::DIM);

    begin = clock();
    EvalRed<GpuConv2D_FromHost>(red, Nx, Ny, pres, pc, px, py, pb);
    end = clock();

    std::cout << "time for GPU computation (2D scheme) : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    std::vector<__TYPE__> resgpu2(Nx*RED::DIM);
    resgpu2 = vres;
    std::cout << "output = ";
    DispValues(resgpu2.data(),5,RED::DIM);

}

int main() {

    // symbolic expression of the function : a gaussian kernel
    auto c = Pm(0,1);
    auto x = Vi(1,DIMPOINT);
    auto y = Vj(2,DIMPOINT);
    auto b = Vj(3,DIMVECT);
    
    auto f = Exp(-c*SqNorm2(x-y)) * b; // or equivalently we could use the alias : auto f = GaussKernel(c,x,y,b);

    std::cout << std::endl << "Function f : " << std::endl;
    std::cout << PrintFormula(f);
    std::cout << std::endl << std::endl;

    // now we test ------------------------------------------------------------------------------

    int Nx=10000, Ny=15000;
        
    // we create random vectors for each variable and get their pointers.
    std::vector<__TYPE__> vx(Nx*DIMPOINT);    fillrandom(vx); __TYPE__ *px = vx.data();
    std::vector<__TYPE__> vy(Ny*DIMPOINT);    fillrandom(vy); __TYPE__ *py = vy.data();
    std::vector<__TYPE__> vb(Ny*DIMVECT);     fillrandom(vb); __TYPE__ *pb = vb.data();
    __TYPE__ pc[1] = {.5};

    DoTest(Sum_Reduction(f,0), "Sum", Nx, Ny, pc, px, py, pb);
    DoTest(Min_Reduction(f,0), "Min", Nx, Ny, pc, px, py, pb);
    DoTest(ArgMin_Reduction(f,0), "ArgMin", Nx, Ny, pc, px, py, pb);
    DoTest(Min_ArgMin_Reduction(f,0), "Min_ArgMin", Nx, Ny, pc, px, py, pb);
    DoTest(Max_Reduction(f,0), "Max", Nx, Ny, pc, px, py, pb);
    DoTest(ArgMax_Reduction(f,0), "ArgMax", Nx, Ny, pc, px, py, pb);
    DoTest(Max_ArgMax_Reduction(f,0), "Max_ArgMax", Nx, Ny, pc, px, py, pb);
    DoTest(KMin_Reduction(f,3,0), "KMin", Nx, Ny, pc, px, py, pb);
    DoTest(ArgKMin_Reduction(f,3,0), "ArgKMin", Nx, Ny, pc, px, py, pb);
    DoTest(KMin_ArgKMin_Reduction(f,3,0), "KMin_ArgKMin", Nx, Ny, pc, px, py, pb);
    // last test with Max_SumShiftExp reduction, which requires formula to be of dimension 1, 
    // so we compute it on the first dimension of f only
    auto f0 = Elem(f,0);
    DoTest(Max_SumShiftExp_Reduction(f0,0), "Max_SumShiftExp", Nx, Ny, pc, px, py, pb);
}



