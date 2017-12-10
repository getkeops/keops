#ifndef CUDAFUNCTIONS
#define CUDAFUNCTIONS

// CONVENTION WARNINGS:
// - for efficiency, these routines take as input the SQUARED norm
// - they are not "normalized", so beware and look at the formula that is actually implemented
// - for efficiency, the "gaussian" formula is exp(-x2 / s^2), and not exp(-x^2 / 2s^2)

// Defines the elementary Real-valued functions, which take as input a quadratic norm
// and output a kernel value (gaussian...) and its derivatives.

// An abstract radial function f : r^2 -> f(r^2) should define :
template < typename TYPE >
class RadialFunction {

  public:

    virtual __host__ __device__ __forceinline__ TYPE Eval(TYPE u) = 0;  // f  (r2)
    virtual __host__ __device__ __forceinline__ TYPE Diff(TYPE u) = 0;  // f' (r2)
    virtual __host__ __device__ __forceinline__ TYPE Diff2(TYPE u) = 0; // f''(r2)
    virtual __host__ __device__ __forceinline__ void DiffDiff2(TYPE u, TYPE* d1, TYPE* d2) {
        *d1 = Diff(u);
        *d2 = Diff2(u);
    }
};

// Cauchy_s(r2) = 1 / (1 + r2/(s^2))
// It is a scalable version of the Cauchy kernel x -> 1 / (1+x^2)
template < typename TYPE >
class CauchyFunction : public RadialFunction<TYPE> {
    TYPE ooSigma2, ooSigma4; // We precompute the inverse of sigma, for efficiency

  public:

    CauchyFunction() {       // Default constructor, sigma = 1
        ooSigma2 = 1.0;
        ooSigma4 = 1.0;
    }
    CauchyFunction(TYPE sigma) {  // Generic constructor
        ooSigma2 = 1.0/(sigma*sigma);
        ooSigma4 = ooSigma2*ooSigma2;
    }

    __host__ __device__ __forceinline__ TYPE Eval(TYPE r2) {
        return 1.0/(1.0+r2*ooSigma2);
    }
    __host__ __device__ __forceinline__ TYPE Diff(TYPE r2) {
        TYPE u = 1.0+r2*ooSigma2;
        return (- ooSigma2 / (u*u));
    }
    __host__ __device__ __forceinline__ TYPE Diff2(TYPE r2) {
        TYPE u = 1.0+r2*ooSigma2;
        return 2.0*ooSigma4 / (u*u*u);
    }
    __host__ __device__ __forceinline__ void DiffDiff2(TYPE r2, TYPE* d1, TYPE* d2) {
        TYPE u = 1.0/(1.0+r2*ooSigma2);
        *d1 = - ooSigma2 * u * u;
        *d2 = - 2.0 * ooSigma4 * *d1 * u;
    }
};

// Gauss_s(r2) = exp( - r2 / s^2)
// It is a scalable version of the Gaussian kernel x -> exp( - x^2 ) 
template < typename TYPE >
class GaussFunction : public RadialFunction<TYPE> {

    TYPE ooSigma2, ooSigma4; // We precompute the inverse of sigma, for efficiency

  public:

    GaussFunction() {        // Default constructor, sigma = 1
        ooSigma2 = 1.0;
        ooSigma4 = 1.0;
    }
    GaussFunction(TYPE sigma) {  // Generic constructor
        ooSigma2 = 1.0/(sigma*sigma);
        ooSigma4 = ooSigma2*ooSigma2;
    }

    __host__ __device__ __forceinline__ TYPE Eval(TYPE r2) {
        return exp(-r2*ooSigma2);
    }
    __host__ __device__ __forceinline__ TYPE Diff(TYPE r2) {
        return (-ooSigma2*exp(-r2*ooSigma2));
    }
    __host__ __device__ __forceinline__ TYPE Diff2(TYPE r2) {
        return ooSigma4*exp(-r2*ooSigma2);
    }
    __host__ __device__ __forceinline__ void DiffDiff2(TYPE r2, TYPE* d1, TYPE* d2) {
        *d1 = - ooSigma2*exp(-r2*ooSigma2);
        *d2 = - ooSigma2* *d1;
    }
};

// Laplace_s(r2) = exp( - sqrt( s^2 + r2 ) )
// It is a scalable version of the exponential-like kernel x -> exp( - sqrt(1+x^2) ) 
template < typename TYPE >
class LaplaceFunction : public RadialFunction<TYPE> {

    TYPE ooSigma2, ooSigma4;

  public:

    LaplaceFunction() {
        ooSigma2 = 1.0;
        ooSigma4 = 1.0;
    }
    LaplaceFunction(TYPE sigma) {
        ooSigma2 = 1.0/(sigma*sigma);
        ooSigma4 = ooSigma2*ooSigma2;
    }

    __host__ __device__ __forceinline__ TYPE Eval(TYPE r2) {
        return exp(- sqrt( 1.0/ooSigma2 + r2));
    }
    __host__ __device__ __forceinline__ TYPE Diff(TYPE r2) {
        TYPE s = sqrt( 1.0/ooSigma2 + r2);
        return - exp(- s ) / (2.0 * s );
    }
    __host__ __device__ __forceinline__ TYPE Diff2(TYPE r2) {
        TYPE s = sqrt( 1.0/ooSigma2 + r2);
        return .25 * (1.0/(s*s*s) + 1.0/s) * exp(- s);
    }
    __host__ __device__ __forceinline__ void DiffDiff2(TYPE r2, TYPE* d1, TYPE* d2) {
        TYPE s = sqrt( 1.0/ooSigma2 + r2);
        *d1 = - exp(- s ) / (2.0 * s );
        *d2 = .25 * (1.0/(s*s*s) + 1.0/s) * exp(- s);
    }
};


// Energy_s(r2) = 1 / (s^2 + r2)^(1/4)
// It is a scalable and smooth version of the heavy tail kernel x -> 1 / sqrt( ||x|| )
template < typename TYPE >
class EnergyFunction : public RadialFunction<TYPE> {

    TYPE ooSigma2, ooSigma4;

  public:

    EnergyFunction() {
        ooSigma2 = 1.0;
        ooSigma4 = 1.0;
    }
    EnergyFunction(TYPE sigma) {
        ooSigma2 = 1.0/(sigma*sigma);
        ooSigma4 = ooSigma2*ooSigma2;
    }

    __host__ __device__ __forceinline__ TYPE Eval(TYPE r2) {
        return 1.0 / pow( 1.0/ooSigma2 + r2, .25);
    }
    __host__ __device__ __forceinline__ TYPE Diff(TYPE r2) {
        return -.25 / pow( 1.0/ooSigma2 + r2, 1.25);
    }
    __host__ __device__ __forceinline__ TYPE Diff2(TYPE r2) {
        return .3125 / pow( 1.0/ooSigma2 + r2, 2.25);
    }
    __host__ __device__ __forceinline__ void DiffDiff2(TYPE r2, TYPE* d1, TYPE* d2) {
        *d1 = -.25 / pow( 1.0/ooSigma2 + r2, 1.25);
        *d2 = .3125 / pow( 1.0/ooSigma2 + r2, 2.25);
    }
};


// Gauss4_s(r2) = Gauss_s(r2) + Gauss_(s/2)(r2) + Gauss_(s/4)(r2) + Gauss_(s/8)(r2)
// It can be used as a Gaussian approximation to heavy-tail kernels,
// and is most relevant in a "full image" setting where the separability of the Gaussian
// convolution operator comes handy.
// It is provided here for the sake of completeness, and for historical reasons.
template < typename TYPE >
class Sum4GaussFunction : public RadialFunction<TYPE> {
    TYPE ooSigma2, ooSigma4;

  public:

    Sum4GaussFunction() {
        ooSigma2 = 1.0;
        ooSigma4 = 1.0;
    }
    Sum4GaussFunction(TYPE sigma) {
        ooSigma2 = 1.0/(sigma*sigma);
        ooSigma4 = ooSigma2*ooSigma2;
    }

    __host__ __device__ __forceinline__ TYPE Eval(TYPE r2) {
        return exp(-r2*ooSigma2) + exp(-4.0*r2*ooSigma2) + exp(-16.0*r2*ooSigma2) + exp(-64.0*r2*ooSigma2);
    }
    __host__ __device__ __forceinline__ TYPE Diff(TYPE r2) {
        return - ooSigma2*(exp(-r2*ooSigma2) - 4.0*exp(-4.0*r2*ooSigma2) - 16.0*exp(-16.0*r2*ooSigma2) - 64.0*exp(-64.0*r2*ooSigma2));
    }
    __host__ __device__ __forceinline__ TYPE Diff2(TYPE r2) {
        return ooSigma4*(exp(-r2*ooSigma2) + 16.0*exp(-4.0*r2*ooSigma2) + 256.0*exp(-16.0*r2*ooSigma2) + 4096*exp(-64.0*r2*ooSigma2));
    }
    __host__ __device__ __forceinline__ void DiffDiff2(TYPE r2, TYPE* d1, TYPE* d2) {
        *d1 = 0;
        *d2 = 0;
        TYPE u, oos2;
        oos2 = 1;
        for(int k=0; k<4; k++) {
            u = oos2 * exp(-r2*oos2*ooSigma2);
            *d1 -= u;
            *d2 += oos2 * u;
            oos2 *= 4;
        }
        *d1 *= ooSigma2;
        *d2 *= ooSigma4;
    }
};



// Cauchy4_s(r2) = Cauchy_s(r2) + Cauchy_(s/2)(r2) + Cauchy_(s/4)(r2) + Cauchy_(s/8)(r2)
// It can be used as a Cauchy approximation to heavy-tail kernels.
template < typename TYPE >
class Sum4CauchyFunction : public RadialFunction<TYPE> {
    TYPE ooSigma2, ooSigma4;

  public:

    Sum4CauchyFunction() {
        ooSigma2 = 1.0;
        ooSigma4 = 1.0;
    }
    Sum4CauchyFunction(TYPE sigma) {
        ooSigma2 = 1.0/(sigma*sigma);
        ooSigma4 = ooSigma2*ooSigma2;
    }

    __host__ __device__ __forceinline__ TYPE Eval(TYPE r2) {
        return 1.0/(1.0+r2*ooSigma2) + 1.0/(1.0+r2*4.0*ooSigma2) + 1.0/(1.0+r2*16.0*ooSigma2) + 1.0/(1.0+r2*64.0*ooSigma2);
    }
    __host__ __device__ __forceinline__ TYPE Diff(TYPE r2) {
        TYPE u, v = 0, oos2 = 1;
        for(int k=0; k<4; k++) {
            u = 1.0+r2*oos2*ooSigma2;
            v += - oos2 / (u*u);
            oos2 *= 4;
        }
        return v*ooSigma2;
    }
    __host__ __device__ __forceinline__ TYPE Diff2(TYPE r2) {
        TYPE u, v = 0, oos2=1;
        for(int k=0; k<4; k++) {
            u = 1.0+r2*oos2*ooSigma2;
            v += 2.0 * oos2 * oos2 / (u*u*u);
            oos2 *= 4;
        }
        return u*ooSigma4;
    }
    __host__ __device__ __forceinline__ void DiffDiff2(TYPE r2, TYPE* d1, TYPE* d2) {
        *d1 = 0;
        *d2 = 0;
        TYPE u, v, oos2=1;
        for(int k=0; k<4; k++) {
            u = 1.0/(1.0+r2*oos2*ooSigma2);
            v = - oos2 * u * u;
            *d1 += v;
            *d2 -= 2.0 * oos2 * v * u;
            oos2 *= 4;
        }
        *d1 *= ooSigma2;
        *d2 *= ooSigma4;
    }
};



// SumGauss_{w1,s1, w2,s2, ...}(r2) = \sum_i wi * Gauss_{si}(r2)
// It can be a convenient way to generate tunable parametric kernel functions.
// It can be used to generate Gaussian approximations of heavy-tail kernels,
// and is most relevant in a "full image" setting where the separability of the Gaussian
// convolution operator comes handy.
// It is provided here for the sake of completeness, and for historical reasons.
template < typename TYPE >
class SumGaussFunction : public RadialFunction<TYPE> {
    int Nfuns;
    TYPE *Sigmas, *ooSigma2s, *ooSigma4s;
    TYPE *Weights;

  public:

    SumGaussFunction() { }

    __device__ __forceinline__ SumGaussFunction(int nfuns, TYPE* weights, TYPE* sigmas) {
        Nfuns = nfuns;
        cudaMalloc((void**)&Weights, sizeof(TYPE)*Nfuns);
        cudaMalloc((void**)&Sigmas, sizeof(TYPE)*Nfuns);
        cudaMalloc((void**)&ooSigma2s, sizeof(TYPE)*Nfuns);
        cudaMalloc((void**)&ooSigma4s, sizeof(TYPE)*Nfuns);
        cudaMemcpy(Weights, weights, sizeof(TYPE)*Nfuns, cudaMemcpyHostToDevice);
        cudaMemcpy(Sigmas, sigmas, sizeof(TYPE)*Nfuns, cudaMemcpyHostToDevice);
        TYPE *oosigma2s = (TYPE*)malloc(sizeof(TYPE)*Nfuns);
        TYPE *oosigma4s = (TYPE*)malloc(sizeof(TYPE)*Nfuns);
        for(int i=0; i<Nfuns; i++) {
            oosigma2s[i] = 1.0/pow(sigmas[i],2);
            oosigma4s[i] = pow(oosigma2s[i],2);
        }
        cudaMemcpy(ooSigma2s, oosigma2s, sizeof(TYPE)*Nfuns, cudaMemcpyHostToDevice);
        cudaMemcpy(ooSigma4s, oosigma4s, sizeof(TYPE)*Nfuns, cudaMemcpyHostToDevice);
        free(oosigma2s);
        free(oosigma4s);
    }

    __device__ __forceinline__ ~SumGaussFunction() {
        cudaFree(Weights);
        cudaFree(Sigmas);
        cudaFree(ooSigma2s);
        cudaFree(ooSigma4s);
    }
    __device__ __forceinline__ TYPE Eval(TYPE r2) {
        TYPE res = 0.0;
        for(int i=0; i<Nfuns; i++)
            res += Weights[i] * exp(-r2*ooSigma2s[i]);
        return res;
    }
    __device__ __forceinline__ TYPE Diff(TYPE r2) {
        TYPE res = 0.0;
        for(int i=0; i<Nfuns; i++)
            res += Weights[i] * (- ooSigma2s[i] * exp(-r2*ooSigma2s[i]));
        return res;
    }
    __device__ __forceinline__ TYPE Diff2(TYPE r2) {
        TYPE res = 0.0;
        for(int i=0; i<Nfuns; i++)
            res += Weights[i] * (ooSigma4s[i] * exp(-r2*ooSigma2s[i]));
        return res;
    }
    __device__ __forceinline__ void DiffDiff2(TYPE r2, TYPE* d1, TYPE* d2) {
        TYPE tmp;
        *d1 = 0.0;
        *d2 = 0.0;
        for(int i=0; i<Nfuns; i++) {
            tmp = - ooSigma2s[i] * exp(-r2*ooSigma2s[i]);
            *d1 += Weights[i] * tmp;
            *d2 += Weights[i] * (- ooSigma2s[i] * tmp);
        }
    }

};


// SumCauchy_{w1,s1, w2,s2, ...}(r2) = \sum_i wi * Cauchy_{si}(r2)
// It can be a convenient way to generate tunable parametric kernel functions.
template < typename TYPE >
class SumCauchyFunction : public RadialFunction<TYPE> {
    int Nfuns;
    TYPE *Sigmas, *ooSigma2s, *ooSigma4s;
    TYPE *Weights;

  public:

    SumCauchyFunction() { }

    __device__ __forceinline__ SumCauchyFunction(int nfuns, TYPE* weights, TYPE* sigmas) {
        Nfuns = nfuns;
        cudaMalloc((void**)&Weights, sizeof(TYPE)*Nfuns);
        cudaMalloc((void**)&Sigmas, sizeof(TYPE)*Nfuns);
        cudaMalloc((void**)&ooSigma2s, sizeof(TYPE)*Nfuns);
        cudaMalloc((void**)&ooSigma4s, sizeof(TYPE)*Nfuns);
        cudaMemcpy(Weights, weights, sizeof(TYPE)*Nfuns, cudaMemcpyHostToDevice);
        cudaMemcpy(Sigmas, sigmas, sizeof(TYPE)*Nfuns, cudaMemcpyHostToDevice);
        TYPE *oosigma2s = (TYPE*)malloc(sizeof(TYPE)*Nfuns);
        TYPE *oosigma4s = (TYPE*)malloc(sizeof(TYPE)*Nfuns);
        for(int i=0; i<Nfuns; i++) {
            oosigma2s[i] = 1.0/pow(sigmas[i],2);
            oosigma4s[i] = pow(oosigma2s[i],2);
        }
        cudaMemcpy(ooSigma2s, oosigma2s, sizeof(TYPE)*Nfuns, cudaMemcpyHostToDevice);
        cudaMemcpy(ooSigma4s, oosigma4s, sizeof(TYPE)*Nfuns, cudaMemcpyHostToDevice);
        free(oosigma2s);
        free(oosigma4s);
    }
    __device__ __forceinline__ ~SumCauchyFunction() {
        cudaFree(Weights);
        cudaFree(Sigmas);
        cudaFree(ooSigma2s);
        cudaFree(ooSigma4s);
    }
    
    __device__ __forceinline__ TYPE Eval(TYPE r2) {
        TYPE res = 0.0;
        for(int i=0; i<Nfuns; i++)
            res += Weights[i] / (1.0+r2*ooSigma2s[i]);
        return res;
    }
    __device__ __forceinline__ TYPE Diff(TYPE r2) {
        TYPE res = 0.0, u;
        for(int i=0; i<Nfuns; i++) {
            u = 1.0+r2*ooSigma2s[i];
            res += Weights[i] * (- ooSigma2s[i] / (u*u));
        }
        return res;
    }
    __device__ __forceinline__ TYPE Diff2(TYPE r2) {
        TYPE res = 0.0, u;
        for(int i=0; i<Nfuns; i++) {
            u = 1.0+r2*ooSigma2s[i];
            res += (Weights[i] * 2.0 * ooSigma4s[i]) / (u*u*u);
        }
        return res;
    }
    __device__ __forceinline__ void DiffDiff2(TYPE r2, TYPE* d1, TYPE* d2) {
        TYPE u, tmp;
        *d1 = 0.0;
        *d2 = 0.0;
        for(int i=0; i<Nfuns; i++) {
            u = 1.0/(1.0+r2*ooSigma2s[i]);
            tmp = - ooSigma2s[i] * u * u;
            *d1 += Weights[i] * tmp;
            *d2 += Weights[i] * (-2.0 * ooSigma2s[i] * tmp * u);
        }
    }

};

#endif // CUDAFUNCTIONS
