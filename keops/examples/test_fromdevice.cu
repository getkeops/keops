// test convolution with autodiff
// compile with
//		nvcc -I.. -Wno-deprecated-gpu-targets -std=c++14 -O2 -o build/test_fromdevice test_fromdevice.cu

// testing "from device" convolution, i.e. convolution which is performed on the device
// directly from device data

#include <algorithm>
#include <thrust/device_vector.h>
#include <keops_includes.h>

#define DIMPOINT 3
#define DIMVECT 2

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

int main() {

    int deviceID = 0;
    cudaSetDevice(deviceID);

    // symbolic expression of the function : a gaussian kernel
    auto c = Pm(0,1);
    auto x = Vi(1,DIMPOINT);
    auto y = Vj(2,DIMPOINT);
    auto beta = Vj(3,DIMVECT);
    
    auto f = Exp(-c*SqNorm2(x-y)) * beta; 

    std::cout << std::endl << "Function f : " << std::endl;
    std::cout << PrintFormula(f);
    std::cout << std::endl << std::endl;

    auto Sum_f = Sum_Reduction(f,0);

    // now we test ------------------------------------------------------------------------------

    int Nx=4000, Ny=60000;

    std::vector<__TYPE__> vx(Nx*x.DIM);    fillrandom(vx); __TYPE__ *px = vx.data();
    thrust::device_vector<__TYPE__> vx_d(vx);
    __TYPE__ *x_d = thrust::raw_pointer_cast(vx_d.data());

    std::vector<__TYPE__> vy(Ny*DIMPOINT);    fillrandom(vy); __TYPE__ *py = vy.data();
    thrust::device_vector<__TYPE__> vy_d(vy);
    __TYPE__ *y_d = thrust::raw_pointer_cast(vy_d.data());
   
    std::vector<__TYPE__> vb(Ny*DIMVECT);     fillrandom(vb); __TYPE__ *pb = vb.data();
    thrust::device_vector<__TYPE__> vb_d(vb);
    __TYPE__ *b_d = thrust::raw_pointer_cast(vb_d.data());
   
    thrust::device_vector<__TYPE__> vres_d(Nx*Sum_f.DIM);
    __TYPE__ *res_d = thrust::raw_pointer_cast(vres_d.data());
    
    __TYPE__ param = 0.5;
    thrust::device_vector<__TYPE__> vparam_d(c.DIM);
    vparam_d[0] = param;
    __TYPE__ *param_d = thrust::raw_pointer_cast(vparam_d.data());
    

    clock_t begin, end;

    std::cout << "blank run 1" << std::endl;
    begin = clock();
    EvalRed<GpuConv2D_FromDevice>(Sum_f,Nx, Ny, res_d, param_d, x_d, y_d, b_d);
    end = clock();
    std::cout << "time for blank run 1 : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    std::cout << "blank run 2" << std::endl;
    begin = clock();
    EvalRed<GpuConv2D_FromDevice>(Sum_f,Nx, Ny, res_d, param_d, x_d, y_d, b_d);
    end = clock();
    std::cout << "time for blank run 2 : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;






    std::cout << "testing From_Device mode" << std::endl;
    begin = clock();
    for(int i=0; i<100; i++)
        EvalRed<GpuConv2D_FromDevice>(Sum_f,Nx, Ny, res_d, param_d, x_d, y_d, b_d);
    end = clock();
    std::cout << "time for 100 GPU computations (2D scheme) : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    std::vector<__TYPE__> resgpu2D(Nx*Sum_f.DIM);
    cudaMemcpy(resgpu2D.data(), res_d, Nx*Sum_f.DIM*sizeof(__TYPE__), cudaMemcpyDeviceToHost);

    // display output
    std::cout << std::endl << "resgpu2D_device =";
    DispValues(resgpu2D.data(),5,Sum_f.DIM);

    begin = clock();
    for(int i=0; i<100; i++)
        EvalRed<GpuConv1D_FromDevice>(Sum_f,Nx, Ny, res_d, param_d, x_d, y_d, b_d);
    end = clock();
    std::cout << "time for 100 GPU computations (1D scheme) : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    std::vector<__TYPE__> resgpu1D(Nx*Sum_f.DIM);
    cudaMemcpy(resgpu1D.data(), res_d, Nx*Sum_f.DIM*sizeof(__TYPE__), cudaMemcpyDeviceToHost);

    // display output
    std::cout << std::endl << "resgpu1D_device =";
    DispValues(resgpu1D.data(),5,Sum_f.DIM);

    // display mean of errors
    __TYPE__ s = 0;
    for(int i=0; i<Nx*Sum_f.DIM; i++)
        s += std::abs(resgpu1D[i]-resgpu2D[i]);
    std::cout << "mean abs error 1D/2D = " << s/Nx << std::endl;






    std::cout << "testing From_Host mode" << std::endl;

    std::vector<__TYPE__> resgpu2D_host(Nx*Sum_f.DIM);
    begin = clock();
    for(int i=0; i<100; i++)
        EvalRed<GpuConv2D_FromHost>(Sum_f, Nx, Ny, resgpu2D_host.data(), &param, px, py, pb);
    end = clock();
    std::cout << "time for 100 GPU computations (2D scheme) : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    // display output
    std::cout << std::endl << "resgpu2D_host =";
    DispValues(resgpu2D_host.data(),5,Sum_f.DIM);

    std::vector<__TYPE__> resgpu1D_host(Nx*Sum_f.DIM);
    begin = clock();
    for(int i=0; i<100; i++)
        EvalRed<GpuConv1D_FromHost>(Sum_f, Nx, Ny, resgpu1D_host.data(), &param, px, py, pb);
    end = clock();
    std::cout << "time for 100 GPU computations (1D scheme) : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    // display output
    std::cout << std::endl << "resgpu1D_host =";
    DispValues(resgpu1D_host.data(),5,Sum_f.DIM);

    // display mean of errors
    s = 0;
    for(int i=0; i<Nx*Sum_f.DIM; i++)
        s += std::abs(resgpu1D_host[i]-resgpu2D_host[i]);
    std::cout << "mean abs error 1D/2D = " << s/Nx << std::endl;

    // display mean of errors
    s = 0;
    for(int i=0; i<Nx*Sum_f.DIM; i++)
        s += std::abs(resgpu1D_host[i]-resgpu1D[i]);
    std::cout << "mean abs error host/device = " << s/Nx << std::endl;



}



