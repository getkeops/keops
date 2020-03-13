#include "Halide.h"
#include <stdio.h>
#include "clock.h"

using namespace Halide;

int main(int argc, char **argv) {

    double t1, t2;

    int npoints;
    sscanf(argv[1], "%d", &npoints);

    const int dim = 3;

    ImageParam X(type_of<float>(), 2);
    ImageParam Y(type_of<float>(), 2);
    ImageParam B(type_of<float>(), 1);

    Var x("x");
    Func gauss_conv("gauss_conv");

    RDom y(0, npoints);
    RVar yi;

    gauss_conv(x) = 0.0f;
    Expr tmp0 = X(0,x) - Y(0,y);
    Expr tmp1 = X(1,x) - Y(1,y);
    Expr tmp2 = X(2,x) - Y(2,y);
    gauss_conv(x) += exp(-(tmp0*tmp0+tmp1*tmp1+tmp2*tmp2)) * B(y);

    Func out;
    out(x) = gauss_conv(x);

    Var block, thread;
    out.gpu_tile(x, block, thread, 192);

    Target target = get_host_target();
    target.set_feature(Target::OpenCL);

    out.compile_jit(target);

    Buffer<float> mat_X(dim, npoints);
    Buffer<float> mat_Y(dim, npoints);
    Buffer<float> mat_B(npoints);
    Buffer<float> output(npoints);

    // init randomly
    for (int i = 0; i < npoints; i++) 
    {
        for (int k = 0; k < dim; k++) 
        {
            mat_X(k, i) = ((float)(rand()%1000))/1000 - .5;
            mat_Y(k, i) = ((float)(rand()%1000))/1000 - .5;
        }
        mat_B(i) = ((float)(rand()%1000))/1000 - .5;
    }

    X.set(mat_X);
    Y.set(mat_Y);
    B.set(mat_B);

    out.realize(output);

    // check results
    Buffer<float> output_ref(npoints);
    Buffer<float> output_halide(npoints);

    int Ntest;
    if (npoints<200000)
        Ntest = 100;
    else
        Ntest = 10;

    out.realize(output_halide); //dummy call to initialize GPU

    t1 = current_time();
    for (int n=0; n<Ntest; n++)
        out.realize(output_halide);
    output_halide.copy_to_host();
    t2 = current_time();
    printf("3D Gaussian convolution using Halide\n");
    printf("Number of points : %d\n", npoints);
    printf("Average timing over %d runs : %1.4f ms\n", Ntest, (t2-t1)/Ntest);

    return 0;
}



