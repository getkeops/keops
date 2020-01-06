// test convolution with autodiff
// compile with
//		g++ -I.. -std=c++14 -O3 -o build/test_autodiff test_autodiff.cpp

// we define an arbitrary function using available blocks,
// then test its convolution on the CPU, then get its gradient and test again the convolution

// Here we build the function f(x,y,u,v,beta) = <u,v>^2 * exp(-p*|x-y|^2) * beta
// where p is a scalar parameter, x, y, beta are 3D vectors, and u, v are 4D vectors
// and the convolution is res_i = sum_j f(x_i,y_j,u_i,v_j,beta_j)
// then we define the gradients of this reduction with respect to x and y 
// (i.e. the gradient of x -> sum_j f(x_i,y_j,...) and y -> sum_j f(x_i,y_j,...)), with new input variable eta (3D).

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


int main() {
  
  // symbolic expression of the function ------------------------------------------------------
  
  auto x = Vi(0, 3);   // x is the second variable and represents a 3D vector, "i"-indexed.
  auto y = Vj(1, 3);   // y is the third variable and represents a 3D vector, "j"-indexed.
  
  
  // here we define f = <u,v>^2 * exp(-p*|x-y|^2) * beta in usual notations
  //auto f = Square(u|v) * Exp(-p*SqNorm2(x-y)) * beta;
  auto f = Exp(x + y);
  // We define the reduction operation on f. Here a sum reduction, performed over the "j" index, and resulting in a "i"-indexed variable
  auto Sum_f = Sum_Reduction(f, 0);  // 0 means output of reduction will be "i"-indexed (0 means"i", 1 means "j")
  
  // Now we define gradients of the reduction operation:
  // First we define a new variable to be the input of gradient
  auto eta = Vi(2, 3);
  // now we gradient with respect to x ---------------------------------------------------------------
  auto Grad_x_Sum_f = Grad(Sum_f, x, eta);
  // and gradient with respect to y  --------------------------------------------------------------
  auto Grad_y_Sum_f = Grad(Sum_f, y, eta);
  
  
  // now we test ------------------------------------------------------------------------------
  
  int Nx = 50, Ny = 20;
  
  // here we define actual data for all variables and feed it it with random values
  std::vector< __TYPE__ > vx(Nx * x.DIM); fillrandom(vx); __TYPE__* px = vx.data();
  std::vector< __TYPE__ > vy(Ny * y.DIM); fillrandom(vy); __TYPE__* py = vy.data();
  std::vector< __TYPE__ > ve(Nx * eta.DIM); fillrandom(ve); __TYPE__* pe = ve.data();
  
  // also a vector for the output
  std::vector< __TYPE__ > vres(Nx * Sum_f.DIM); fillrandom(vres); __TYPE__* pres = vres.data();
  
  clock_t begin, end;
  
  std::cout << "testing reduction" << std::endl;
  begin = clock();
  EvalRed< CpuConv >(Sum_f, Nx, Ny, pres, px, py);
  end = clock();
  std::cout << "time for CPU computation : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
  
  vres.resize(Nx * Grad_x_Sum_f.DIM);
  pres = vres.data();
  
  std::cout << "testing gradient wrt x" << std::endl;
  begin = clock();
  EvalRed< CpuConv >(Grad_x_Sum_f, Nx, Ny, pres, px, py, pe);
  end = clock();
  std::cout << "time for CPU computation : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
  
  std::cout << "[" << pres[0] << ", " << pres[1] << ", " << pres[2] << ", " << pres[3] << ", " << pres[4] << ", "
            << pres[5] << "]" << std::endl;
  
  // gradient wrt y, which is a "j" variable.
  
  
  vres.resize(Ny * Grad_y_Sum_f.DIM);
  pres = vres.data();
  
  std::cout << "testing gradient wrt y" << std::endl;
  begin = clock();
  EvalRed< CpuConv >(Grad_y_Sum_f, Nx, Ny, pres, px, py, pe);
  end = clock();
  std::cout << "time for CPU computation : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
  
  
  std::cout << "[" << pres[0] << ", " << pres[1] << ", " << pres[2] << ", " << pres[3] << ", " << pres[4] << ", "
            << pres[5] << "]" << std::endl;
  
}
