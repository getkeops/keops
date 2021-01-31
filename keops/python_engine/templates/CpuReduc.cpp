#include <cmath>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#define DIRECT_SUM 0
#define BLOCK_SUM 1
#define KAHAN_SCHEME 2

extern "C" int Eval(int nx, int ny, {dtype}* out, {signature_list(args)}) {{
    #pragma omp parallel for
    for (int i = 0; i < nx; i++) {{
        {fout.declare()}
        {acc.declare()}
#if SUM_SCHEME == BLOCK_SUM
        // additional tmp vector to store intermediate results from each block
        {tmp.declare()}
#elif SUM_SCHEME == KAHAN_SCHEME
        // additional tmp vector to accumulate errors
        {tmp_kahan.declare()}
#endif
        {red_formula.InitializeReduction(acc)}
#if SUM_SCHEME == BLOCK_SUM
        {red_formula.InitializeReduction(tmp)}
#elif SUM_SCHEME == KAHAN_SCHEME
        {tmp_kahan.assign(c_zero_float)}
#endif
        for (int j = 0; j < ny; j++) {{
            {red_formula.formula(fout,table)}
#if SUM_SCHEME == BLOCK_SUM
            {red_formula.ReducePairShort(tmp, fout, j)}
            if ((j+1)%200) {{
                {red_formula.ReducePair(acc, tmp)}
                {red_formula.InitializeReduction(tmp)}
            }}
#elif SUM_SCHEME == KAHAN_SCHEME
            {red_formula.KahanScheme(acc, fout, tmp_kahan)}
#else
            {red_formula.ReducePairShort(acc, fout, j)}
#endif
        }}
#if SUM_SCHEME == BLOCK_SUM
        {red_formula.ReducePair(acc, tmp)}
#endif
        {red_formula.FinalizeOutput(acc, outi, i)}
    }}
    return 0;
}}

