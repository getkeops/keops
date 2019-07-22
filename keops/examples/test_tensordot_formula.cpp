// test convolution
// compile with
//		g++ -I.. -std=c++11 -O3 -o build/test_simple test_simple.cpp

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
  auto x = Vi(0,8); 	 // x is the second variable and represents a 3D vector, "i"-indexed.
  auto y = Vj(1,4); 	 // y is the third variable and represents a 3D vector, "j"-indexed.


  // symbolic expression of the function ------------------------------------------------------

  // here we define f = <x,y> in usual notations
  auto f = TensorDot(x,y,2);

  // We define the reduction operation on f. Here a sum reduction, performed over the "j" index, and resulting in a "i"-indexed variable
  auto Sum_f = Sum_Reduction(f,0);  // 0 means output of reduction will be "i"-indexed (0 means"i", 1 means "j")


  // now we test ------------------------------------------------------------------------------

  int Nx=5, Ny=10;

  // here we define actual data for all variables and feed it it with random values
  std::vector<__TYPE__> vx(Nx*x.DIM);    fillrandom(vx); __TYPE__ *px = vx.data();
  std::vector<__TYPE__> vy(Ny*y.DIM);    fillrandom(vy); __TYPE__ *py = vy.data();


  // also a vector for the output
  std::vector<__TYPE__> vres(Nx*Sum_f.DIM);    fillrandom(vres); __TYPE__ *pres = vres.data();

  std::cout << "testing Sum reduction" << std::endl;
  EvalRed<CpuConv>(Sum_f,Nx, Ny, pres, px, py);

  std::cout << "output:" << std::endl;
  DispValues(pres,5,Sum_f.DIM);

}




