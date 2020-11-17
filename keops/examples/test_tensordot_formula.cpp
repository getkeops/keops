
// compile with
//		g++ -I.. -std=c++14 -O3 -o build/test_tensordot_formula test_tensordot_formula.cpp

#include <iostream>
#include <algorithm>
#include <keops_includes.h>


using namespace keops;

__TYPE__ floatrand() {
  return ((__TYPE__) std::rand()) / RAND_MAX - .5;    // random value between -.5 and .5
}

template< class V >
void fillrandom(V& v) {
  generate(v.begin(), v.end(), floatrand);    // fills vector with random values
}

// a function to display output of reduction
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

int main() {
  
  // In this part we define the symbolic variables of the function
  auto x = Vi(0, 3 * 12 * 12);   // x is the second variable and represents a 3D vector, "i"-indexed.
  auto y = Vj(1, 12 * 12);   // y is the third variable and represents a 3D vector, "j"-indexed.
  
  
  // symbolic expression of the function ------------------------------------------------------
  
  // here we define f = x : y in usual notations
  auto f = TensorDot(x, y, Ind(3, 12, 12), Ind(12, 12), Ind(1, 2), Ind(0, 1));
  
  // We define the reduction operation on f. Here a sum reduction, performed over the "j" index, and resulting in a "i"-indexed variable
  auto Sum_f = Sum_Reduction(f, 0);  // 0 means output of reduction will be "i"-indexed (0 means"i", 1 means "j")
  
  
  // now we test ------------------------------------------------------------------------------
  
  int Nx = 50, Ny = 10;
  
  // here we define actual data for all variables and feed it it with random values
  std::vector< __TYPE__ > vx(Nx * x.DIM); fillrandom(vx); __TYPE__* px = vx.data();
  std::vector< __TYPE__ > vy(Ny * y.DIM); fillrandom(vy); __TYPE__* py = vy.data();
  
  // also a vector for the output
  std::vector< __TYPE__ > vres(Nx * Sum_f.DIM); fillrandom(vres); __TYPE__* pres = vres.data();
  
  std::cout << "Testing Sum reduction of :" << std::endl;;
  std::cout << PrintFormula(f);
  std::cout << std::endl;
  
  std::cout << std::endl << "Output:" << std::endl;
  EvalRed< CpuConv >(Sum_f, Nx, Ny, pres, px, py);
  DispValues(pres, 5, Sum_f.DIM);
  
}




