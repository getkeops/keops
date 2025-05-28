from keopscore.formulas.reductions import *
from keopscore.formulas.GetReduction import GetReduction
from keopscore.utils.code_gen_utils import Var_loader, new_c_varname, pointer, c_include


class MapReduce:
    """
    base class for map-reduce schemes
    """

    def __init__(
        self,
        red_formula_string,
        aliases,
        nargs,
        dtype,
        dtypeacc,
        sum_scheme_string,
        tagHostDevice,
        tagCpuGpu,
        tag1D2D,
        use_half,
        use_fast_math,
        device_id,
    ):
        self.red_formula_string = red_formula_string
        self.aliases = aliases

        self.red_formula = GetReduction(red_formula_string, aliases=aliases)

        self.dtype = dtype
        self.dtypeacc = dtypeacc
        self.nargs = nargs
        self.sum_scheme_string = sum_scheme_string
        self.tagHostDevice, self.tagCpuGpu, self.tag1D2D = (
            tagHostDevice,
            tagCpuGpu,
            tag1D2D,
        )
        self.use_half = use_half
        self.use_fast_math = use_fast_math
        self.device_id = device_id
        self.varloader = Var_loader(self.red_formula)

    def get_code(self):
        self.headers = "#define C_CONTIGUOUS 1\n" + """
        #define HD  __device__

// HD versions for float
HD inline float compat_exp(float x) { return expf(x); }
HD inline float compat_floor(float x) { return floorf(x); }
HD inline float compat_pow(float base, float exp) { return powf(base, exp); }
HD inline float compat_fabs(float x) { return fabsf(x); }
HD inline bool compat_isinf(float x) { return isinf(x); }
HD inline bool compat_isnan(float x) { return isnan(x); }

// HD versions for double
HD inline double compat_exp(double x) { return exp(x); }
HD inline double compat_floor(double x) { return floor(x); }
HD inline double compat_pow(double base, double exp) { return pow(base, exp); }
HD inline double compat_fabs(double x) { return fabs(x); }
HD inline bool compat_isinf(double x) { return isinf(x); }
HD inline bool compat_isnan(double x) { return isnan(x); }


#define MY_EUL 0.57721566490153286060
#define MY_BIG 1.44115188075855872E+17
#define MY_MACHEP 1.11022302462515654042E-16
#define MY_MAXLOG 7.08396418532264106224E2

HD static const int nA = 13;

HD static const double MY_A0[] = {
    1.00000000000000000
};

HD static const double MY_A1[] = {
    1.00000000000000000
};

HD static const double MY_A2[] = {
    -2.00000000000000000,
    1.00000000000000000
};

HD static const double MY_A3[] = {
    6.00000000000000000,
    -8.00000000000000000,
    1.00000000000000000
};

HD static const double MY_A4[] = {
    -24.0000000000000000,
    58.0000000000000000,
    -22.0000000000000000,
    1.00000000000000000
};

HD static const double MY_A5[] = {
    120.000000000000000,
    -444.000000000000000,
    328.000000000000000,
    -52.0000000000000000,
    1.00000000000000000
};

HD static const double MY_A6[] = {
    -720.000000000000000,
    3708.00000000000000,
    -4400.00000000000000,
    1452.00000000000000,
    -114.000000000000000,
    1.00000000000000000
};

HD static const double MY_A7[] = {
    5040.00000000000000,
    -33984.0000000000000,
    58140.0000000000000,
    -32120.0000000000000,
    5610.00000000000000,
    -240.000000000000000,
    1.00000000000000000
};

HD static const double MY_A8[] = {
    -40320.0000000000000,
    341136.000000000000,
    -785304.000000000000,
    644020.000000000000,
    -195800.000000000000,
    19950.0000000000000,
    -494.000000000000000,
    1.00000000000000000
};

HD static const double MY_A9[] = {
    362880.000000000000,
    -3733920.00000000000,
    11026296.0000000000,
    -12440064.0000000000,
    5765500.00000000000,
    -1062500.00000000000,
    67260.0000000000000,
    -1004.00000000000000,
    1.00000000000000000
};

HD  const double MY_A10[] = {
    -3628800.00000000000,
    44339040.0000000000,
    -162186912.000000000,
    238904904.000000000,
    -155357384.000000000,
    44765000.0000000000,
    -5326160.00000000000,
    218848.000000000000,
    -2026.00000000000000,
    1.00000000000000000
};

HD static const double MY_A11[] = {
    39916800.0000000000,
    -568356480.000000000,
    2507481216.00000000,
    -4642163952.00000000,
    4002695088.00000000,
    -1648384304.00000000,
    314369720.000000000,
    -25243904.0000000000,
    695038.000000000000,
    -4072.00000000000000,
    1.00000000000000000
};

HD static const double MY_A12[] = {
    -479001600.000000000,
    7827719040.00000000,
    -40788301824.0000000,
    92199790224.0000000,
    -101180433024.000000,
    56041398784.0000000,
    -15548960784.0000000,
    2051482776.00000000,
    -114876376.000000000,
    2170626.00000000000,
    -8166.00000000000000,
    1.00000000000000000
};

HD static const double *MY_A[] = {
    MY_A0, MY_A1, MY_A2,
    MY_A3, MY_A4, MY_A5,
    MY_A6, MY_A7, MY_A8,
    MY_A9, MY_A10, MY_A11,
    MY_A12
};

HD static const int Adegs[] = {
    0, 0, 1,
    2, 3, 4,
    5, 6, 7,
    8, 9, 10,
    11
};

HD double my_gamma(double in0) {

    double out0;

    if (compat_isinf(in0) && in0 < 0) {
        out0 = -1.0 / 0.0;
    } else if (in0 < 0. && in0 == compat_floor(in0)) {
        out0 = 1.0 / 0.0;
    } else {
        out0 = tgamma(in0);
    }
//std::cout << "eeeeeee  "<< out0 << std::endl;
    return out0;
}




HD double my_polevl(double x, const double coef[], int N) {
    double ans;
    const double *p;

    p = coef;
    ans = *p++;

    // y = a * x_i + b
    for (int i = 0; i < N; i++) {
        ans = ans * x + *p++;
    }

    return ans;
}



HD double expn_large_n(int n, double x) {

    int k;
    double p = n;
    double lambda = x / p;
    double multiplier = 1 / p / (lambda + 1) / (lambda + 1);
    double fac = 1;
    double res = 1;
    double expfac, term;

    expfac = compat_exp(-lambda * p) / (lambda + 1) / p;

    if (expfac == 0) {
        return 0;
    }

    /* Do the k = 1 term outside the loop since A[1] = 1 */
    fac *= multiplier;
    res += fac;

    for (k = 2; k < nA; k++) {
        fac *= multiplier;
        term = fac * my_polevl(lambda, MY_A[k], Adegs[k]);
        res += term;
        if (compat_fabs(term) < MY_MACHEP * compat_fabs(res)) {
            break;
        }
    }

    return expfac * res;
}


HD double expn(int n, double x) {

    double ans, r, t, yk, xk;
    double pk, pkm1, pkm2, qk, qkm1, qkm2;
    double psi, z;
    int i, k;
    double big = MY_BIG;

    if (compat_isnan(x)) {
        return nan(""); // std::nan(x);
    } else if (n < 0 || x < 0) {
        return nan(""); // std::nan(x);
    }

    if (x > MY_MAXLOG) {
        return 0.0;
    }

    if (x == 0.0) {
        if (n < 2) {
            return 1.0 / 0.0;
        } else {
            return (1.0 / (n - 1.0));
        }
    }

    if (n == 0) {
        return (exp(-x) / x);
    }

    /* Asymptotic expansion for large n */
    if (n > 50) {
        ans = expn_large_n(n, x);
        return (ans);
    }

    /* Continued fraction */
    if (x > 1.0) {
        k = 1;
        pkm2 = 1.0;
        qkm2 = x;
        pkm1 = 1.0;
        qkm1 = x + n;
        ans = pkm1 / qkm1;

        do {
            k += 1;
            if (k & 1) {
                yk = 1.0;
                xk = n + (k - 1) / 2;
            } else {
                yk = x;
                xk = k / 2;
            }
            pk = pkm1 * yk + pkm2 * xk;
                qk = qkm1 * yk + qkm2 * xk;
                if (qk != 0) {
                    r = pk / qk;
                    t = fabs((ans - r) / r);
                    ans = r;
                } else {
                    t = 1.0;
                }
                pkm2 = pkm1;
                pkm1 = pk;
                qkm2 = qkm1;
                qkm1 = qk;
                if (fabs(pk) > big) {
                    pkm2 /= big;
                    pkm1 /= big;
                    qkm2 /= big;
                    qkm1 /= big;
                }
        } while (t > MY_MACHEP);

        ans *= exp(-x);
    }

    /* Power series expansion */
    psi = -MY_EUL - log(x);
    for (i = 1; i < n; i++) {
        psi = psi + 1.0 / i;
    }

    z = -x;
    xk = 0.0;
    yk = 1.0;
    pk = 1.0 - n;
    if (n == 1) {
        ans = 0.0;
    } else {
        ans = 1.0 / pk;
    }
    do {
        xk += 1.0;
            yk *= z / xk;
            pk += 1.0;
            if (pk != 0.0) {
                ans += yk / pk;
            }
            if (ans != 0.0)
                t = compat_fabs(yk / ans);
            else
                t = 1.0;
        //std::cout << ans << std::endl;
    } while (t > MY_MACHEP);

    k = xk;
    t = n;
    r = n - 1;
    //std::cout << "d   "<< my_gamma(t) << std::endl;
    ans = (compat_pow(z, r) * psi / my_gamma(t)) - ans;
    //std::cout << "dddd  "<< ans << std::endl;
    return (ans);

}
"""

        if self.use_half == 1:
            self.headers += "#define USE_HALF 1\n"
            self.headers += c_include("cuda_fp16.h")
        else:
            self.headers += "#define USE_HALF 0\n"

        red_formula = self.red_formula
        formula = red_formula.formula
        dtype = self.dtype
        dtypeacc = self.dtypeacc
        nargs = self.nargs
        self.sum_scheme = eval(self.sum_scheme_string)(red_formula, dtype)

        self.i = i = c_variable("signed long int", "i")
        self.j = j = c_variable("signed long int", "j")

        nx = c_variable("signed long int", "nx")
        ny = c_variable("signed long int", "ny")

        self.xi = c_array(dtype, self.varloader.dimx, "xi")
        self.param_loc = c_array(dtype, self.varloader.dimp, "param_loc")

        argname = new_c_varname("arg")
        self.arg = c_variable(pointer(pointer(dtype)), argname)
        self.args = [self.arg[k] for k in range(nargs)]

        self.acc = c_array(dtypeacc, red_formula.dimred, "acc")
        self.acctmp = c_array(dtypeacc, red_formula.dimred, "acctmp")
        self.fout = c_array(dtype, formula.dim, "fout")
        self.outi = c_array(dtype, red_formula.dim, f"(out + i * {red_formula.dim})")
