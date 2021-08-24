// Example of reductions using KeOps C++ routines.
// This code performs the following computations :
//       res_i = Red_j F(i,j)
// where : 
//    - F is a Gaussian kernel F(i,j) = exp(-c*|x_i-y_j|^2)*b_j 
//    - the x_i are Nx=5000 points in R^3 
//    - the y_j are Ny=2000 points in R^3
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
// This program runs on CPU ; see the file test_reductions.cu for the equivalent program on GPU.
//
// This example can be compiled with the command
//		g++ -I.. -D__TYPE__=float -std=c++14 -O3 -o build/test_reductions test_reductions.cpp
// 

#include <algorithm>
#include <iostream>

#include <keops_includes.h>


using namespace keops;

__TYPE__ floatrand() {
  return ((__TYPE__) std::rand()) / RAND_MAX - .5;    // random value between -.5 and .5
}

template< class V >
void fillrandom(V& v) {
  generate(v.begin(), v.end(), floatrand);    // fills vector with random values
}

void DispValues(__TYPE__* x, int N, int dim) {
  std::cout << std::endl;
  int k = 0;
  for (int i = 0; i < N; i++) {
    for (int d = 0; d < dim; d++) {
      std::cout << x[k] << " ";
      k++;
    }
    std::cout << std::endl;
  }
  for (int d = 0; d < dim; d++)
    std::cout << "... ";
  std::cout << std::endl << std::endl;
}

#define DIMPOINT 3
#define DIMVECT 2

template< class RED >
void DoTest(std::string red_id, int Nx, int Ny, __TYPE__* oos2, __TYPE__* x, __TYPE__* y, __TYPE__* b) {
  std::cout << "Testing " << red_id << " reduction" << std::endl;
  
  std::vector< __TYPE__ > vres(Nx * RED::DIM);
  fillrandom(vres);
  __TYPE__* res = vres.data();
  
  clock_t begin, end;
  
  begin = clock();
  Eval< RED, CpuConv >::Run(Nx, Ny, res, oos2, x, y, b);
  end = clock();
  
  std::cout << "time for CPU computation : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
  
  std::cout << "output = ";
  DispValues(res, 5, RED::DIM);
}


template< class RED >
void DoTest(RED red, std::string red_id, int Nx, int Ny, __TYPE__* pc, __TYPE__* px, __TYPE__* py, __TYPE__* pb) {
  std::cout << "Testing " << red_id << " reduction" << std::endl;
  
  std::vector< __TYPE__ > vres(Nx * RED::DIM);
  fillrandom(vres);
  __TYPE__* pres = vres.data();
  
  clock_t begin, end;
  
  begin = clock();
  EvalRed< CpuConv >(red, Nx, Ny, pres, pc, px, py, pb);
  end = clock();
  
  std::cout << "time for CPU computation : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
  
  std::vector< __TYPE__ > rescpu(Nx * RED::DIM);
  rescpu = vres;
  std::cout << "output = ";
  DispValues(rescpu.data(), 5, RED::DIM);
  
}

int main() {
  
  // symbolic expression of the function : a gaussian kernel
  auto c = Pm(0, 1);
  auto x = Vi(1, DIMPOINT);
  auto y = Vj(2, DIMPOINT);
  auto b = Vj(3, DIMVECT);
  
  auto f = Exp(-c * SqNorm2(x - y)) * b; // or equivalently we could use the alias : auto f = GaussKernel(c,x,y,b);
  
  std::cout << std::endl << "Function f : " << std::endl;
  std::cout << PrintFormula(f);
  std::cout << std::endl << std::endl;
  
  // now we test ------------------------------------------------------------------------------
  
  int Nx = 100, Ny = 150;
  
  // we create random vectors for each variable and get their pointers.
  std::vector< __TYPE__ > vx(Nx * DIMPOINT); fillrandom(vx); __TYPE__* px = vx.data();
  std::vector< __TYPE__ > vy(Ny * DIMPOINT); fillrandom(vy); __TYPE__* py = vy.data();
  std::vector< __TYPE__ > vb(Ny * DIMVECT);  fillrandom(vb); __TYPE__* pb = vb.data();
  __TYPE__ pc[1] = {.5};
  
  DoTest(Sum_Reduction(f, 0), "Sum", Nx, Ny, pc, px, py, pb);
  DoTest(Min_Reduction(f, 0), "Min", Nx, Ny, pc, px, py, pb);
  DoTest(ArgMin_Reduction(f, 0), "ArgMin", Nx, Ny, pc, px, py, pb);
  DoTest(Min_ArgMin_Reduction(f, 0), "Min_ArgMin", Nx, Ny, pc, px, py, pb);
  DoTest(Max_Reduction(f, 0), "Max", Nx, Ny, pc, px, py, pb);
  DoTest(ArgMax_Reduction(f, 0), "ArgMax", Nx, Ny, pc, px, py, pb);
  DoTest(Max_ArgMax_Reduction(f, 0), "Max_ArgMax", Nx, Ny, pc, px, py, pb);
  DoTest(KMin_Reduction(f, 3, 0), "KMin", Nx, Ny, pc, px, py, pb);
  DoTest(ArgKMin_Reduction(f, 3, 0), "ArgKMin", Nx, Ny, pc, px, py, pb);
  DoTest(KMin_ArgKMin_Reduction(f, 3, 0), "KMin_ArgKMin", Nx, Ny, pc, px, py, pb);
  // last test with Max_SumShiftExp reduction, which requires formula to be of dimension 1,
  // so we compute it on the first dimension of f only
  auto f0 = Elem(f, 0);
  DoTest(Max_SumShiftExp_Reduction(f0, 0), "Max_SumShiftExp", Nx, Ny, pc, px, py, pb);
}



