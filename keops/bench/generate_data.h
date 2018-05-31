#include <vector>
#include <algorithm>

// Some convenient functions
__TYPE__ generate_rand() {
    return ((__TYPE__) std::rand())/RAND_MAX-.5;    // random value between -.5 and .5
}

template < class V > void fillrandom(V& v) {
    generate(v.begin(), v.end(), generate_rand);    // fills vector with random values
}

template <typename T>
class data {
  private:

    inline  static T generate_rand() {
        return ((T) std::rand())/RAND_MAX-.5;    // random value between -.5 and .5
    }

    inline void fillrandom(std::vector<T>& v) {
        generate(v.begin(), v.end(), generate_rand);    // fills vector with random values
    }

  public:

    data (int);
    //data() = default;
    
    // dimensions
    int Nx, Ny, dimPoint, dimVect;
    
    // data
    std::vector<T> vf, vx, vy, vu, vv;
    T *f, *x, *y, *u, *v;
    
    // wrap variables
    std::vector<T*> vargs;
    T **args;
    T params[1];
};

template <typename T>
data<T>::data(int a) {
    Nx=a;
    Ny= Nx *2 ;
    dimPoint = 3;
    dimVect = 3;

    vf.resize(Nx*dimPoint); fillrandom(vf); f = vf.data();
    vx.resize(Nx*dimPoint); fillrandom(vx); x = vx.data();
    vy.resize(Ny*dimPoint); fillrandom(vy); y = vy.data();
    vu.resize(Nx*dimVect); fillrandom(vu); u = vu.data();
    vv.resize(Ny*dimVect); fillrandom(vv); v = vv.data();

    T Sigma = 1;
    params[0] = 1.0/(Sigma*Sigma);

    vargs.resize(5);
    vargs[0]=params; vargs[1]=x; vargs[2]=y; vargs[3]=v; vargs[4]=u;
    args = vargs.data();

}

