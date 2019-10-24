// test convolution with autodiff
// compile with
//		nvcc -I.. -DCUDA_BLOCK_SIZE=192 -DMAXTHREADSPERBLOCK0=1024 -DSHAREDMEMPERBLOCK0=49152 -Wno-deprecated-gpu-targets -std=c++14 -O2 -o build/test_autodiff test_autodiff.cu

// we define an arbitrary function using available blocks,
// then test its convolution on the CPU or GPU, then get its gradients and test again the convolutions

// Here we build the function f(x,y,u,v,beta) = <u,v>^2 * exp(-p*|x-y|^2) * beta
// where p is a scalar parameter, x, y, beta are 3D vectors, and u, v are 4D vectors
// and the convolution is res_i = sum_j f(x_i,y_j,u_i,v_j,beta_j)
// then we define the gradients of this reduction with respect to x and y 
// (i.e. the gradient of x -> sum_j f(x_i,y_j,...) and y -> sum_j f(x_i,y_j,...)), with new input variable eta (3D).

#include <algorithm>

// __TYPE__ is floating point type used in all code; may be float or double
// if not defined here or previously, it will be set by default to float in keops_includes.h below
#ifndef __TYPE__
  #define __TYPE__ double
#endif

#include <keops_includes.h>

using namespace keops;

__TYPE__ floatrand() {
    return ((__TYPE__) std::rand())/RAND_MAX-.5;    // random value between -.5 and .5
}

template < class V > void fillrandom(V& v) {
    generate(v.begin(), v.end(), floatrand);    // fills vector with random values
}



int main() {

    // set Gpu device
    int deviceID = 0;
    
    // In this part we define the symbolic variables of the function
    auto p = Pm(0,1);	 // p is the first variable and is a scalar parameter
    auto x = Vi(1,3); 	 // x is the second variable and represents a 3D vector, "i"-indexed.
    auto y = Vj(2,3); 	 // y is the third variable and represents a 3D vector, "j"-indexed.
    auto u = Vi(3,4); 	 // u is the fourth variable and represents a 4D vector, "i"-indexed.
    auto v = Vj(4,4); 	 // v is the fourth variable and represents a 4D vector, "j"-indexed.
    auto beta = Vj(5,3); // beta is the sixth variable and represents a 3D vector, "j"-indexed.

    // symbolic expression of the function ------------------------------------------------------

    // here we define f = <u,v>^2 * exp(-p*|x-y|^2) * beta in usual notations
    auto f = Square(u|v) * Exp(-p*SqNorm2(x-y)) * beta;
    
    // We define the reduction operation on f. Here a sum reduction, performed over the "j" index, and resulting in a "i"-indexed variable
    auto Sum_f = Sum_Reduction(f,0);  // 0 means output of reduction will be "i"-indexed (0 means"i", 1 means "j")

    // Now we define gradients of the reduction operation:
    // First we define a new variable to be the input of gradient
    auto eta = Vi(6,Sum_f.DIM); 
    // now we gradient with respect to x ---------------------------------------------------------------
    auto Grad_x_Sum_f = Grad(Sum_f,x,eta);
    // and gradient with respect to y  --------------------------------------------------------------
    auto Grad_y_Sum_f = Grad(Sum_f,y,eta);



    // now we test ------------------------------------------------------------------------------

    int Nx=5000, Ny=2000;

    // here we define actual data for all variables and feed it it with random values
    std::vector<__TYPE__> vx(Nx*x.DIM);    fillrandom(vx); __TYPE__ *px = vx.data();
    std::vector<__TYPE__> vy(Ny*y.DIM);    fillrandom(vy); __TYPE__ *py = vy.data();
    std::vector<__TYPE__> vu(Nx*u.DIM);    fillrandom(vu); __TYPE__ *pu = vu.data();
    std::vector<__TYPE__> vv(Ny*v.DIM);    fillrandom(vv); __TYPE__ *pv = vv.data();
    std::vector<__TYPE__> vb(Ny*beta.DIM); fillrandom(vb); __TYPE__ *pb = vb.data();

    // also a vector for the output
    std::vector<__TYPE__> vres(Nx*Sum_f.DIM);    fillrandom(vres); __TYPE__ *pres = vres.data();

    // and three vectors to get copies of the output, for comparing Cpu vs Gpu/1D vs Gpu/2D computations
    std::vector<__TYPE__> resgpu2D(Nx*Sum_f.DIM), resgpu1D(Nx*Sum_f.DIM), rescpu(Nx*Sum_f.DIM);

    // parameter variable
    __TYPE__ params[1];
    __TYPE__ Sigma = 4.0;
    params[0] = 1.0/(Sigma*Sigma);

    clock_t begin, end;

    begin = clock();
    cudaSetDevice(deviceID);        
    end = clock();
    std::cout << "time for GPU initialization : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    std::cout << "blank run" << std::endl;
    begin = clock();
    EvalRed<GpuConv2D_FromHost>(Sum_f,Nx, Ny, pres, params, px, py, pu, pv, pb);
    end = clock();
    std::cout << "time for blank run : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    std::cout << "testing reduction" << std::endl;
    begin = clock();
    EvalRed<GpuConv2D_FromHost>(Sum_f,Nx, Ny, pres, params, px, py, pu, pv, pb);
    end = clock();
    std::cout << "time for GPU computation (2D scheme) : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    resgpu2D = vres;

    begin = clock();
    EvalRed<GpuConv1D_FromHost>(Sum_f,Nx, Ny, pres, params, px, py, pu, pv, pb);
    end = clock();
    std::cout << "time for GPU computation (1D scheme) : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    resgpu1D = vres;

    begin = clock();
    EvalRed<CpuConv>(Sum_f,Nx, Ny, pres, params, px, py, pu, pv, pb);
    end = clock();
    std::cout << "time for CPU computation : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    rescpu = vres;

    // display mean of errors
    __TYPE__ s = 0;
    for(int i=0; i<Nx*Sum_f.DIM; i++)
        s += std::abs(resgpu2D[i]-rescpu[i]);
    std::cout << "mean abs error 2D =" << s/Nx << std::endl;

    s = 0;
    for(int i=0; i<Nx*Sum_f.DIM; i++)
        s += std::abs(resgpu1D[i]-rescpu[i]);
    std::cout << "mean abs error 1D =" << s/Nx << std::endl;



    rescpu.resize(Nx*Grad_x_Sum_f.DIM);
    resgpu2D.resize(Nx*Grad_x_Sum_f.DIM);
    resgpu1D.resize(Nx*Grad_x_Sum_f.DIM);
    vres.resize(Nx*Grad_x_Sum_f.DIM);
    pres = vres.data();

    std::vector<__TYPE__> ve(Nx*eta.DIM); fillrandom(ve); __TYPE__ *pe = ve.data();

    std::cout << "testing gradient wrt x" << std::endl;
    begin = clock();
    EvalRed<GpuConv2D_FromHost>(Grad_x_Sum_f,Nx, Ny,pres, params, px, py, pu, pv, pb, pe);
    end = clock();
    std::cout << "time for GPU computation (2D scheme) : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    resgpu2D = vres;

    begin = clock();
    EvalRed<GpuConv1D_FromHost>(Grad_x_Sum_f,Nx, Ny, pres, params, px, py, pu, pv, pb, pe);
    end = clock();
    std::cout << "time for GPU computation (1D scheme) : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    resgpu1D = vres;

    begin = clock();
    EvalRed<CpuConv>(Grad_x_Sum_f,Nx, Ny, pres, params, px, py, pu, pv, pb, pe);
    end = clock();
    std::cout << "time for CPU computation : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    rescpu = vres;

    // display mean of errors
    s = 0;
    for(int i=0; i<Nx*Grad_x_Sum_f.DIM; i++)
        s += std::abs(resgpu2D[i]-rescpu[i]);
    std::cout << "mean abs error 2D =" << s/Nx << std::endl;

    s = 0;
    for(int i=0; i<Nx*Grad_x_Sum_f.DIM; i++)
        s += std::abs(resgpu1D[i]-rescpu[i]);
    std::cout << "mean abs error 1D =" << s/Nx << std::endl;



    // gradient wrt y, which is a "j" variable.

    rescpu.resize(Ny*Grad_y_Sum_f.DIM);
    resgpu2D.resize(Ny*Grad_y_Sum_f.DIM);
    resgpu1D.resize(Ny*Grad_y_Sum_f.DIM);
    vres.resize(Ny*Grad_y_Sum_f.DIM);
    pres = vres.data();

    std::cout << "testing gradient wrt y" << std::endl;
    begin = clock();
    EvalRed<GpuConv2D_FromHost>(Grad_y_Sum_f,Ny, Nx, pres, params, px, py, pu, pv, pb, pe);
    end = clock();
    std::cout << "time for GPU computation (2D scheme) : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    resgpu2D = vres;

    begin = clock();
    EvalRed<GpuConv1D_FromHost>(Grad_y_Sum_f,Ny, Nx, pres, params, px, py, pu, pv, pb, pe);
    end = clock();
    std::cout << "time for GPU computation (1D scheme) : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    resgpu1D = vres;

    begin = clock();
    EvalRed<CpuConv>(Grad_y_Sum_f,Ny, Nx, pres, params, px, py, pu, pv, pb, pe);
    end = clock();
    std::cout << "time for CPU computation : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    rescpu = vres;

    // display mean of errors
    s = 0;
    for(int i=0; i<Ny*Grad_y_Sum_f.DIM; i++)
        s += std::abs(resgpu2D[i]-rescpu[i]);
    std::cout << "mean abs error 2D=" << s/Ny << std::endl;

    s = 0;
    for(int i=0; i<Ny*Grad_y_Sum_f.DIM; i++)
        s += std::abs(resgpu1D[i]-rescpu[i]);
    std::cout << "mean abs error 1D=" << s/Ny << std::endl;



}



