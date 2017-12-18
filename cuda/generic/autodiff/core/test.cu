
#define __TYPE__ float

#include <iostream>
#include <sstream>

#include "autodiff.h"

using namespace std;

int main() {
    using F = GaussKernel_<2,3>;
    cout << "F=" << endl;
    F::PrintId();
    cout << endl << endl;

    using U = univpack<Var<3,2,1>,F>;
    cout << "U=" << endl;
    U::PrintId();
    cout << endl << endl;

    using AF = F::AllTypes;
    cout << "All types in F:" << endl;
    AF::PrintId();
    cout << endl;

    return 0;
}