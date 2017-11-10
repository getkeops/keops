#ifndef CUDAFUNCTIONS
#define CUDAFUNCTIONS

template < typename TYPE >
class RadialFunction
{
    
public:
    
    virtual __device__ __forceinline__ TYPE Eval(TYPE u) = 0;
    virtual __device__ __forceinline__ TYPE Diff(TYPE u) = 0;
    virtual __device__ __forceinline__ TYPE Diff2(TYPE u) = 0;
    virtual __device__ __forceinline__ void DiffDiff2(TYPE u, TYPE* d1, TYPE* d2)
    {
        *d1 = Diff(u);
        *d2 = Diff2(u);
    }
};




template < typename TYPE >
class CauchyFunction : public RadialFunction<TYPE>
{
    
public:
    
    CauchyFunction() { }
        
    __device__ __forceinline__ TYPE Eval(TYPE r2)
    {
        return 1.0/(1.0+r2);
    }
    
    __device__ __forceinline__ TYPE Diff(TYPE r2)
    {
        TYPE u = 1.0+r2;
        return (- 1.0 / (u*u));
    }
    
    __device__ __forceinline__ TYPE Diff2(TYPE r2)
    {
        TYPE u = 1.0+r2;
        return 2.0 / (u*u*u);
    }
    
    __device__ __forceinline__ void DiffDiff2(TYPE r2, TYPE* d1, TYPE* d2)
    {
        TYPE u = 1.0/(1.0+r2);
        *d1 = - u * u;
        *d2 = - 2.0 * *d1 * u;
    }
};

template < typename TYPE >
class GaussFunction : public RadialFunction<TYPE>
{

public:
    
    GaussFunction() { }
    
    __device__ __forceinline__ TYPE Eval(TYPE r2)
    {
        return exp(-r2);
    }
    
    __device__ __forceinline__ TYPE Diff(TYPE r2)
    {
        return (-exp(-r2));
    }
    
    __device__ __forceinline__ TYPE Diff2(TYPE r2)
    {
        return exp(-r2);
    }
    
    __device__ __forceinline__ void DiffDiff2(TYPE r2, TYPE* d1, TYPE* d2)
    {
        *d1 = - exp(-r2);
        *d2 = - *d1;
    }
};

template < typename TYPE >
class Sum4GaussFunction : public RadialFunction<TYPE>
{

public:
    
    Sum4GaussFunction() { }
        
    __device__ __forceinline__ TYPE Eval(TYPE r2)
    {
        return exp(-r2) + exp(-4.0*r2) + exp(-16.0*r2) + exp(-64.0*r2);
    }
    
    __device__ __forceinline__ TYPE Diff(TYPE r2)
    {
        return - exp(-r2) - 4.0*exp(-4.0*r2) - 16.0*exp(-16.0*r2) - 64.0*exp(-64.0*r2);
    }
    
    __device__ __forceinline__ TYPE Diff2(TYPE r2)
    {
	return exp(-r2) + 16.0*exp(-4.0*r2) + 256.0*exp(-16.0*r2) + 4096*exp(-64.0*r2);
    }
    
    __device__ __forceinline__ void DiffDiff2(TYPE r2, TYPE* d1, TYPE* d2)
    {
        *d1 = 0;
        *d2 = 0;
        TYPE u, oosigma2;
	oosigma2 = 1;
        for(int k=0; k<4; k++)
        {
            u = oosigma2 * exp(-r2*oosigma2);
            *d1 -= u;
            *d2 += oosigma2 * u;
	    oosigma2 *= 4;
        }
    }
};

template < typename TYPE >
class Sum4CauchyFunction : public RadialFunction<TYPE>
{
 
public:
    
    Sum4CauchyFunction() { }
    
    __device__ __forceinline__ TYPE Eval(TYPE r2)
    {
        return 1.0/(1.0+r2) + 1.0/(1.0+r2*4.0) + 1.0/(1.0+r2*16.0) + 1.0/(1.0+r2*64.0);
    }
  
    __device__ __forceinline__ TYPE Diff(TYPE r2)
    {
        TYPE u, v = 0, oosigma2 = 1;
        for(int k=0; k<4; k++)
        {
            u = 1.0+r2*oosigma2;
            v += - oosigma2 / (u*u);
	    oosigma2 *= 4;
        }
        return v;
    }
   
    __device__ __forceinline__ TYPE Diff2(TYPE r2)
    {
        TYPE u, v = 0, oosigma2=1;
        for(int k=0; k<4; k++)
        {
            u = 1.0+r2*oosigma2;
            v += 2.0 * oosigma2 * oosigma2 / (u*u*u);
	    oosigma2 *= 4;
        }
        return u;
    }

    __device__ __forceinline__ void DiffDiff2(TYPE r2, TYPE* d1, TYPE* d2)
    {
        *d1 = 0;
        *d2 = 0;
        TYPE u, v, oosigma2=1;
        for(int k=0; k<4; k++)
        {
            u = 1.0/(1.0+r2*oosigma2);
            v = - oosigma2 * u * u;
            *d1 += v;
            *d2 -= 2.0 * oosigma2 * v * u;
	    oosigma2 *= 4;
        }
    }
};



template < typename TYPE >
class SumGaussFunction : public RadialFunction<TYPE>
{
	int Nfuns;
        TYPE *Sigmas, *ooSigma2s, *ooSigma4s;
        TYPE *Weights;
    
    public:
    
    SumGaussFunction() { }

    __device__ __forceinline__ SumGaussFunction(int nfuns, TYPE* weights, TYPE* sigmas)
    {
        Nfuns = nfuns;
        cudaMalloc((void**)&Weights, sizeof(TYPE)*Nfuns);
        cudaMalloc((void**)&Sigmas, sizeof(TYPE)*Nfuns);
        cudaMalloc((void**)&ooSigma2s, sizeof(TYPE)*Nfuns);
        cudaMalloc((void**)&ooSigma4s, sizeof(TYPE)*Nfuns);
    	cudaMemcpy(Weights, weights, sizeof(TYPE)*Nfuns, cudaMemcpyHostToDevice);
    	cudaMemcpy(Sigmas, sigmas, sizeof(TYPE)*Nfuns, cudaMemcpyHostToDevice);
	TYPE *oosigma2s = (TYPE*)malloc(sizeof(TYPE)*Nfuns);
	TYPE *oosigma4s = (TYPE*)malloc(sizeof(TYPE)*Nfuns);
        for(int i=0; i<Nfuns; i++)
        {
            oosigma2s[i] = 1.0/pow(sigmas[i],2);
	    oosigma4s[i] = pow(oosigma2s[i],2);
        }
    	cudaMemcpy(ooSigma2s, oosigma2s, sizeof(TYPE)*Nfuns, cudaMemcpyHostToDevice);
    	cudaMemcpy(ooSigma4s, oosigma4s, sizeof(TYPE)*Nfuns, cudaMemcpyHostToDevice);
	free(oosigma2s);
	free(oosigma4s);
    }
      
    __device__ __forceinline__ ~SumGaussFunction()
    {
    	cudaFree(Weights);
    	cudaFree(Sigmas);
    	cudaFree(ooSigma2s);
    	cudaFree(ooSigma4s);
    }
    
    __device__ __forceinline__ TYPE Eval(TYPE r2)
    {
        TYPE res = 0.0;
        for(int i=0; i<Nfuns; i++)
            res += Weights[i] * exp(-r2*ooSigma2s[i]);
        return res;
    }
    
    __device__ __forceinline__ TYPE Diff(TYPE r2)
    {
        TYPE res = 0.0;
        for(int i=0; i<Nfuns; i++)
            res += Weights[i] * (- ooSigma2s[i] * exp(-r2*ooSigma2s[i]));
        return res;
    }

    __device__ __forceinline__ TYPE Diff2(TYPE r2)
    {
        TYPE res = 0.0;
        for(int i=0; i<Nfuns; i++)
            res += Weights[i] * (ooSigma4s[i] * exp(-r2*ooSigma2s[i]));
        return res;
    }

    __device__ __forceinline__ void DiffDiff2(TYPE r2, TYPE* d1, TYPE* d2)
    {
        TYPE tmp;
        *d1 = 0.0;
        *d2 = 0.0;
        for(int i=0; i<Nfuns; i++)
        {
            tmp = - ooSigma2s[i] * exp(-r2*ooSigma2s[i]);
	    *d1 += Weights[i] * tmp;
            *d2 += Weights[i] * (- ooSigma2s[i] * tmp);
        }
    }

};

template < typename TYPE >
class SumCauchyFunction : public RadialFunction<TYPE>
{
	int Nfuns;
        TYPE *Sigmas, *ooSigma2s, *ooSigma4s;
        TYPE *Weights;
    
    public:
    
    SumCauchyFunction() { }

    __device__ __forceinline__ SumCauchyFunction(int nfuns, TYPE* weights, TYPE* sigmas)
    {
        Nfuns = nfuns;
        cudaMalloc((void**)&Weights, sizeof(TYPE)*Nfuns);
        cudaMalloc((void**)&Sigmas, sizeof(TYPE)*Nfuns);
        cudaMalloc((void**)&ooSigma2s, sizeof(TYPE)*Nfuns);
        cudaMalloc((void**)&ooSigma4s, sizeof(TYPE)*Nfuns);
    	cudaMemcpy(Weights, weights, sizeof(TYPE)*Nfuns, cudaMemcpyHostToDevice);
    	cudaMemcpy(Sigmas, sigmas, sizeof(TYPE)*Nfuns, cudaMemcpyHostToDevice);
	TYPE *oosigma2s = (TYPE*)malloc(sizeof(TYPE)*Nfuns);
	TYPE *oosigma4s = (TYPE*)malloc(sizeof(TYPE)*Nfuns);
        for(int i=0; i<Nfuns; i++)
        {
            oosigma2s[i] = 1.0/pow(sigmas[i],2);
	    oosigma4s[i] = pow(oosigma2s[i],2);
        }
    	cudaMemcpy(ooSigma2s, oosigma2s, sizeof(TYPE)*Nfuns, cudaMemcpyHostToDevice);
    	cudaMemcpy(ooSigma4s, oosigma4s, sizeof(TYPE)*Nfuns, cudaMemcpyHostToDevice);
	free(oosigma2s);
	free(oosigma4s);
    }
      
    __device__ __forceinline__ ~SumCauchyFunction()
    {
    	cudaFree(Weights);
    	cudaFree(Sigmas);
    	cudaFree(ooSigma2s);
    	cudaFree(ooSigma4s);
    }
    
    __device__ __forceinline__ TYPE Eval(TYPE r2)
    {
        TYPE res = 0.0;
        for(int i=0; i<Nfuns; i++)
            res += Weights[i] / (1.0+r2*ooSigma2s[i]);
        return res;
    }
           
    __device__ __forceinline__ TYPE Diff(TYPE r2)
    {
        TYPE res = 0.0, u;
        for(int i=0; i<Nfuns; i++)
	{
		u = 1.0+r2*ooSigma2s[i];
	        res += Weights[i] * (- ooSigma2s[i] / (u*u));
	}
        return res;
    }

    __device__ __forceinline__ TYPE Diff2(TYPE r2)
    {
        TYPE res = 0.0, u;
        for(int i=0; i<Nfuns; i++)
	{
		u = 1.0+r2*ooSigma2s[i];
        	res += (Weights[i] * 2.0 * ooSigma4s[i]) / (u*u*u);
	}
        return res;
    }

    __device__ __forceinline__ void DiffDiff2(TYPE r2, TYPE* d1, TYPE* d2)
    {
        TYPE u, tmp;
        *d1 = 0.0;
        *d2 = 0.0;
        for(int i=0; i<Nfuns; i++)
        { 
		u = 1.0/(1.0+r2*ooSigma2s[i]);
        	tmp = - ooSigma2s[i] * u * u;
		*d1 += Weights[i] * tmp;
        	*d2 += Weights[i] * (-2.0 * ooSigma2s[i] * tmp * u);
        }
    }

 

};

#endif // CUDAFUNCTIONS
