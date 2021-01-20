#include <cmath>

#ifdef USE_OPENMP
#include <omp.h>
#endif

extern "C" int CpuReduc(int nx, int ny, {TYPE}* out, {TYPE}** args) {{
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

