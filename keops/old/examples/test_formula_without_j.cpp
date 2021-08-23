// test convolution
// compile with 
//		g++ -I.. -std=c++14 -O3 -o build/test_formula_without_j test_formula_without_j.cpp

// we define an arbitrary function using available blocks,
// then test its convolution on the CPU

// Here we build the function f(x,y,u,v,beta) = <u,v>^2 * exp(-p*|x-y|^2) * beta
// where p is a scalar parameter, x, y, beta are 3D vectors, and u, v are 4D vectors
// and the convolution is res_i = sum_j f(x_i,y_j,u_i,v_j,beta_j)

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
  auto x = Vi(0, 2);   // x is the second variable and represents a 3D vector, "i"-indexed.
  auto y = Vi(1, 2);   // y is the third variable and represents a 3D vector, "j"-indexed.
  
  
  // symbolic expression of the function ------------------------------------------------------
  
  // here we define f = x : y in usual notations
  // auto f = TensorDot(x,y, Ind(3,12,12), Ind(12,12), Ind(1,2), Ind(0,1));
  auto f = Add(x, y);
  
  // We define the reduction operation on f. Here a sum reduction, performed over the "j" index, and resulting in a "i"-indexed variable
  auto Sum_f = Sum_Reduction(f, 0);  // 0 means output of reduction will be "i"-indexed (0 means"i", 1 means "j")
  
  
  // now we test ------------------------------------------------------------------------------
  
  int Nx = 5000;
  
  // here we define actual data for all variables and feed it it with random values
  std::vector< __TYPE__ > vx(Nx * x.DIM); fillrandom(vx); __TYPE__* px = vx.data();
  std::vector< __TYPE__ > vy(Nx * y.DIM); fillrandom(vy); __TYPE__* py = vy.data();
  
  
  // also a vector for the output
  std::vector< __TYPE__ > vres(Nx * Sum_f.DIM); fillrandom(vres); __TYPE__* pres = vres.data();
  
  std::cout << "Testing Sum reduction of :" << std::endl;;
  std::cout << PrintFormula(f);
  std::cout << std::endl;
  
  std::cout << std::endl << "Output:" << std::endl;
  EvalRed< CpuConv >(Sum_f, Nx, 1, pres, px, py);
  DispValues(pres, 5, Sum_f.DIM);
  
}
