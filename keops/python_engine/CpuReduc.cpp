#include <cmath>

#ifdef USE_OPENMP
#include <omp.h>
#endif

extern "C" int Eval(int nx, int ny, {TYPE}* out {args}) {{
    {TYPE}* args[{nargs}];
    {loadargs}
    {definep}
    {loadp}
    #pragma omp parallel for
    for (int i = 0; i < nx; i++) {{
        {TYPE} fout[{DIMFOUT}];
        {definex}
        {definey}
        {TYPEACC} acc[{DIMRED}];
        {loadx}
        {InitializeReduction}
        for (int j = 0; j < ny; j++) {{
            {loady}
            {call}
            {ReducePairShort}
        }}
        {FinalizeOutput}
    }}
    return 0;
}}

