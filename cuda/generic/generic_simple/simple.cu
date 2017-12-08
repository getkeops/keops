
// This program shows how to write custom convolution operations that will be performed on the GPU

// nvcc -std=c++11 -Xcompiler -fPIC -shared -o simple.so simple.cu


#include "GpuConv2D.cu"


// Example 1 : Convolution with Gauss kernel in 3D
// gamma_i = sum_j exp(-|x_i-y_j|^2/Sigma^2) beta_j
// where x_i, y_j, beta_j, gamma_j are 3D

struct GaussEval {
    typedef pack<3,3> DIMSX;	// dimensions of "i" variables gamma_i, x_i
    typedef pack<3,3> DIMSY; 	// dimensions of "j" variables y_j, beta_j

    __host__ __device__ __forceinline__ void operator()(float ooSigma2, float* gammai, float* xi, float* yj, float* betaj) {
        float r2 = 0.0f;
        float temp;
        for(int k=0; k<3; k++) {
            temp =  yj[k]-xi[k];
            r2 += temp*temp;
        }
        float s = exp(-r2*ooSigma2);
        for(int k=0; k<3; k++)
            gammai[k] += s * betaj[k];
    }
};

extern "C" int GaussConv(float ooSigma2, float* x, float* y, float* beta, float* gamma, int nx, int ny) {
    return GpuConv2D(GaussEval(), ooSigma2, nx, ny, gamma, x, y, beta);
}


// Example 2 : product of a Gauss kernel and a squared scalar product
// gamma_i = sum_j exp(-|x_i-y_j|^2/Sigma^2)<a_i,b_j>^2
// with x_i, y_j 3D and a_i,b_j 2D, and output gamma_i is 1D

struct GaussSqScalEval {
    typedef pack<1,3,2> DIMSX; 	// dimensions of "i" variables gamma_i, x_i, a_i
    typedef pack<3,2> DIMSY; 	// dimensions of "j" variables y_j, b_j

    __host__ __device__ __forceinline__ void operator()(float ooSigma2, float* gammai, float* xi, float* ai, float* yj, float* bj) {
        float r2 = 0.0f;
        float temp;
        for(int k=0; k<3; k++) {
            temp =  yj[k]-xi[k];
            r2 += temp*temp;
        }
        float s = exp(-r2*ooSigma2);
        temp = 0;
        for(int k=0; k<2; k++)
            temp += ai[k]*bj[k];
        gammai[0] += s * temp * temp;
    }
};

extern "C" int GaussSqScalConv(float ooSigma2, float* x, float* y, float* a, float* b, float* gamma, int nx, int ny) {
    return GpuConv2D(GaussSqScalEval(), ooSigma2, nx, ny, gamma, x, a, y, b);
}


// Example 3 : Convolution with Gauss kernel again, but we put the functor inside a class ; here for example
// it allows to compute 1/sigma^2 from sigma offline

class GaussKer {
    float Sigma, ooSigma2;

  public :

    GaussKer(float sigma) {
        Sigma = sigma;
        ooSigma2 = 1/(sigma*sigma);
    }

    struct sEval { // static wrapper
        typedef pack<3,3> DIMSX;	// dimensions of "i" variables gamma_i, x_i
        typedef pack<3,3> DIMSY; 	// dimensions of "j" variables y_j, beta_j

        __host__ __device__ __forceinline__ void operator()(GaussKer ker, float* gammai, float* xi, float* yj, float* betaj) {
            ker.Eval(gammai,xi,yj,betaj);
        }
    };

    __host__ __device__ __forceinline__ void Eval(float* gammai, float* xi, float* yj, float* betaj) {
        float r2 = 0.0f;
        float temp;
        for(int k=0; k<3; k++) {
            temp =  yj[k]-xi[k];
            r2 += temp*temp;
        }
        float s = exp(-r2*ooSigma2);
        for(int k=0; k<3; k++)
            gammai[k] += s * betaj[k];
    }
};

extern "C" int GaussConv_alt(float Sigma, float* x, float* y, float* beta, float* gamma, int nx, int ny) {
    return GpuConv2D(GaussKer::sEval(), GaussKer(Sigma), nx, ny, gamma, x, y, beta);
}

