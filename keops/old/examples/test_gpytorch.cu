// test gpytorch compile with
//		nvcc -I.. -DCUDA_BLOCK_SIZE=192 -DMAXTHREADSPERBLOCK0=1024 -DSHAREDMEMPERBLOCK0=49152 -std=c++14 -O2 -o build/test_gpytorch ./test_gpytorch.cu

#include <algorithm>
#include <keops_includes.h>

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

  int Nx=2000, Ny=2000;

  // TEST with 101
  int N = 101; 
  auto Sum_f = Sum_Reduction(((((Var(0,1,2) * Sqrt(Sum(Square((Var(1,101,0) - Var(2,101,1)))))) + (IntCst(1) + (Var(3,1,2) * Square(Sqrt(Sum(Square((Var(1,101,0) - Var(2,101,1))))))))) * Exp((Var(4,1,2) * Sqrt(Sum(Square((Var(1,101,0) - Var(2,101,1)))))))) * Var(5,1,1)),0); 

  std::vector<__TYPE__> va(1);        fillrandom(va);   __TYPE__ *pa = va.data();
  std::vector<__TYPE__> vx(Nx*N);     fillrandom(vx);   __TYPE__ *px = vx.data();
  std::vector<__TYPE__> vy(Ny*N);     fillrandom(vy);   __TYPE__ *py = vy.data();
  std::vector<__TYPE__> vb(1);        fillrandom(vb);   __TYPE__ *pb = vb.data();
  std::vector<__TYPE__> vc(1);        fillrandom(vc);   __TYPE__ *pc = vc.data();
  std::vector<__TYPE__> vz(Nx*1);     fillrandom(vz);   __TYPE__ *pz = vz.data();

  std::vector<__TYPE__> vres(Nx*Sum_f.DIM);    fillrandom(vres); __TYPE__ *pres = vres.data();

  EvalRed<GpuConv1D_FromHost>(Sum_f,Nx, Ny, pres, pa, px, py, pb, pc, pz);

  // TEST with 102
  int NN = 102;
  auto Sum_f_102 = Sum_Reduction(((((Var(0,1,2) * Sqrt(Sum(Square((Var(1,102,0) - Var(2,102,1)))))) + (IntCst(1) + (Var(3,1,2) * Square(Sqrt(Sum(Square((Var(1,102,0) - Var(2,102,1))))))))) * Exp((Var(4,1,2) * Sqrt(Sum(Square((Var(1,102,0) - Var(2,102,1)))))))) * Var(5,1,1)),0) ;


  std::vector<__TYPE__> vx_102(Nx*NN);     fillrandom(vx_102);   __TYPE__ *px_102 = vx_102.data();
  std::vector<__TYPE__> vy_102(Ny*NN);     fillrandom(vy_102);   __TYPE__ *py_102 = vy_102.data();
  std::vector<__TYPE__> vres_102(Nx*Sum_f_102.DIM);    fillrandom(vres_102); __TYPE__ *pres_102 = vres_102.data();

  EvalRed<GpuConv1D_FromHost>(Sum_f_102,Nx, Ny, pres_102, pa, px_102, py_102, pb, pc, pz);

}



