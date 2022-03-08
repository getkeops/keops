from keopscore.formulas.Operation import Operation
from keopscore.utils.code_gen_utils import use_pragma_unroll

####################################
######  Tensor Dot Product     #####
####################################


def prod(x):
    # product of all elements in list of integers
    res = 1
    for item in x:
        res *= item
    return res


def select(x, ind):
    # indexing of list via list of integers
    return [x[i] for i in ind]


def delete(x, ind):
    # delete items given by indices in list
    n = len(x)
    indkeep = list(set(range(n)) - set(ind))
    indkeep.sort()
    return select(x, indkeep)


def cumprod_array(x):
    # special cumulative product
    if len(x) == 0:
        return x
    else:

        def cumprod(x):
            res = x.copy()
            for i in range(1, len(x)):
                res[i] *= res[i - 1]
            return res

        return cumprod(x[1:][::-1])[::-1] + [1]


def permutation(perm, arr):
    if perm is None:
        return arr
    else:
        tmp = sorted(range(len(perm)), key=perm.__getitem__)
        return select(arr, tmp)


class TensorDot(Operation):
    string_id = "TensorDot"

    def __init__(self, fa, fb, dimsfa, dimsfb, contfa, contfb, permute=None):

        dimsfa = list(dimsfa)
        dimsfb = list(dimsfb)
        contfa = list(contfa)
        contfb = list(contfb)

        assert select(dimsfb, contfb) == select(dimsfa, contfa)

        assert fa.dim == prod(dimsfa)
        assert fb.dim == prod(dimsfb)

        super().__init__(fa, fb)

        self.dimfa = dimsfa
        self.dimfb = dimsfb
        self.contdims = select(dimsfa, contfa)

        self.indices_keepdim_a = delete(list(range(len(dimsfa))), contfa)
        self.keepdims_a = delete(dimsfa, contfa)
        self.contdims_a = select(dimsfa, contfa)
        self.list_strides_dimsfa = cumprod_array(dimsfa)

        self.indices_keepdim_b = delete(list(range(len(dimsfb))), contfb)
        self.keepdims_b = delete(dimsfb, contfb)
        self.contdims_b = select(dimsfb, contfb)
        self.list_strides_dimsfb = cumprod_array(dimsfb)

        self.keepdims = self.keepdims_a + self.keepdims_b
        self.list_strides_keepdim = cumprod_array(permutation(permute, self.keepdims))

        self.dim = fa.dim * fb.dim
        self.dim = int(self.dim / prod(self.contdims) ** 2) if len(contfa) else self.dim

        if permute is None:
            permute = list(range(len(self.keepdims)))
        else:
            assert permutation(permute, permute) == list(range(len(self.keepdims)))

        self.permute = permute

        # loop
        self.loopdim = self.keepdims + self.contdims_a
        self.dimloop = prod(self.loopdim)
        self.number_of_dimloop = len(dimsfa) + len(dimsfb) - len(contfa)

        self.ala = list(range(len(self.keepdims_a))) + list(
            range(len(self.keepdims), self.number_of_dimloop)
        )

        self.ali = self.indices_keepdim_a + contfa
        self.list_indices_a_intot = permutation(self.ali, self.ala)

        self.bla = list(range(len(self.keepdims_a), len(self.keepdims))) + list(
            range(len(self.keepdims), self.number_of_dimloop)
        )

        self.bli = self.indices_keepdim_b + contfb
        self.list_indices_b_intot = permutation(self.bli, self.bla)

        # Gradient
        self.dimfa_grad = permutation(permute, self.keepdims)

        self.list_indices_keepdim_a_inout = list(range(0, len(self.keepdims_a)))
        self.reordered_contfa = permutation(contfb, contfa)
        self.reordered_keepdim_a = permutation(
            select(permute, self.list_indices_keepdim_a_inout), self.indices_keepdim_a
        )
        self.moveaxis_a = self.reordered_keepdim_a + self.reordered_contfa

        self.list_indices_keepdim_b_inout = list(
            range(len(self.keepdims_a), len(self.keepdims))
        )
        self.reordered_contfb = permutation(contfa, contfb)
        self.reordered_keepdim_b = permutation(
            select(permute, self.list_indices_keepdim_b_inout), self.indices_keepdim_b
        )
        self.moveaxis_b = self.reordered_keepdim_b + self.reordered_contfb

        self.contfa_grad = select(permute, self.list_indices_keepdim_b_inout)
        self.contfb_grad = select(permute, self.list_indices_keepdim_a_inout)

    def Op(self, out, table, arg0, arg1):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out

        str_code = ""

        for i in range(len(self.loopdim)):
            str_code += (
                f"for(int TD_var_{chr(70 + i)}=0; TD_var_{chr(70 + i)}<{self.loopdim[i]}; ++TD_var_{chr(70 + i)})"
                + "{\n"
                + i * "    "
            )

        list_indices_keepdim = permutation(self.permute, range(len(self.keepdims)))
        str_out_indices = ""
        for i, v in enumerate(list_indices_keepdim):
            str_out_indices += (
                f"TD_var_{chr(70 + v)} * {self.list_strides_keepdim[i]} + "
            )

        str_a_indices = ""
        for i, v in enumerate(self.list_indices_a_intot):
            str_a_indices += f"TD_var_{chr(70 + v)} * {self.list_strides_dimsfa[i]} + "

        str_b_indices = ""
        for i, v in enumerate(self.list_indices_b_intot):
            str_b_indices += f"TD_var_{chr(70 + v)} * {self.list_strides_dimsfb[i]} + "

        str_code += (
            len(self.loopdim) * "    "
            + f"{out.id}[{str_out_indices[:-2]}] += {arg0.id}[{str_a_indices[:-2]}] * {arg1.id}[{str_b_indices[:-2]}];\n"
        )

        str_code += len(self.loopdim) * "}\n"

        return f"""
                    #if C_CONTIGUOUS     // row major
                        {use_pragma_unroll()}
                        for (int i = 0; i < {out.dim}; i++)
                            {out.id}[i] = ({out.dtype})(0.0f);
                        
                        {use_pragma_unroll()}                       
                        {str_code}
                    #else               // column major
                        
                    #endif
                """

    def DiffT(self, v, gradin):
        f = self.children[0]
        g = self.children[1]
        return f.DiffT(
            v,
            TensorDot(
                gradin,
                g,
                self.dimfa_grad,
                self.dimfb,
                self.contfa_grad,
                self.indices_keepdim_b,
                self.moveaxis_a,
            ),
        ) + g.DiffT(
            v,
            TensorDot(
                gradin,
                f,
                self.dimfa_grad,
                self.dimfa,
                self.contfb_grad,
                self.indices_keepdim_a,
                self.moveaxis_b,
            ),
        )
