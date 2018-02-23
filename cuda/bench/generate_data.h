#include <vector>
#include <algorithm>

// Some convenient functions
__TYPE__ generate_rand() {
    return ((__TYPE__)rand())/RAND_MAX-.5;    // random value between -.5 and .5
}

template < class V > void fillrandom(V& v) {
    generate(v.begin(), v.end(), generate_rand);    // fills vector with random values
}

template <typename T> 
class data{
    private:

    inline  static T generate_rand() {
        return ((T)rand())/RAND_MAX-.5;    // random value between -.5 and .5
    }

    inline void fillrandom2(std::vector<T>& v) {
        generate(v.begin(), v.end(), generate_rand);    // fills vector with random values
    }

    public:

    int Nx, Ny, dimPoint, dimVect;

    std::vector<T> vf, vx, vy, vu, vv;
    T *f, *x, *y, *u, *v;
    data (int);
    //// wrap variables
    std::vector<T*> vargs;

    T **args;

    T params[1];
}; 

template <typename T>
data<T>::data(int a){ 
    Nx=a;
    Ny= Nx *2 ;
    dimPoint = 3;
    dimVect = 3;

    vf.resize(Nx*dimPoint); fillrandom2(vf); f = vf.data(); 
    vx.resize(Nx*dimPoint); fillrandom2(vx); x = vx.data(); 
    vy.resize(Ny*dimPoint); fillrandom2(vy); y = vy.data(); 
    vu.resize(Nx*dimVect);  fillrandom2(vu); u = vu.data(); 
    vv.resize(Ny*dimVect);  fillrandom2(vv); v = vv.data(); 

    std::vector<T*> vargs(4);

    vargs[0]=x; vargs[1]=y; vargs[2]=v; vargs[3]=u;

    args = vargs.data();

    T Sigma = 1;

    params[0] = 1.0/(Sigma*Sigma);
}

