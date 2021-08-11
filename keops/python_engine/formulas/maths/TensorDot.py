import numpy as np

from keops.python_engine.formulas.Operation import Operation
from keops.python_engine.utils.code_gen_utils import use_pragma_unroll

####################################
######  Tensor Dot Product     #####
####################################


class TensorDot(Operation):
    string_id = "TensorDot"

    def __init__(self, fa, fb, dimsfa, dimsfb, contfa, contfb, permute=None):
        # print(dimsfa, dimsfb, contfa, contfb, permute)

        assert (dimsfb[contfb] == dimsfa[contfa]).all()
        assert (fa.dim == dimsfa.prod())
        assert (fb.dim == dimsfb.prod())

        super().__init__(fa, fb)

        self.dimfa = dimsfa
        self.dimfb = dimsfb
        self.contdims = dimsfa[contfa]

        self.indices_keepdim_a = np.delete(np.arange(len(dimsfa)), contfa)
        self.keepdims_a = np.delete(dimsfa, contfa)
        self.contdims_a = dimsfa[contfa]
        self.list_strides_dimsfa = self.cumprod_array(dimsfa)

        self.indices_keepdim_b = np.delete(np.arange(len(dimsfb)), contfb)
        self.keepdims_b = np.delete(dimsfb, contfb)
        self.contdims_b = dimsfb[contfb]
        self.list_strides_dimsfb = self.cumprod_array(dimsfb)

        self.keepdims = np.concatenate((self.keepdims_a, self.keepdims_b))
        self.list_strides_keepdim = self.cumprod_array(
            self.permutation(permute, self.keepdims)
        )

        self.dim = fa.dim * fb.dim
        self.dim = (
            int(self.dim / self.contdims.prod() ** 2) if len(contfa) else self.dim
        )

        if permute is None:
            permute = np.arange(len(self.keepdims))
        else:
            assert (
                    self.permutation(permute, permute) == np.arange(len(self.keepdims))
            ).all()

        self.permute = permute

        # loop
        self.loopdim = np.concatenate((self.keepdims, self.contdims_a))
        self.dimloop = self.loopdim.prod()
        self.number_of_dimloop = len(dimsfa) + len(dimsfb) - len(contfa)

        self.ala = np.concatenate(
            (
                np.arange(0, len(self.keepdims_a)),
                np.arange(len(self.keepdims), self.number_of_dimloop),
            ),
            axis=None,
        ).copy()
        self.ali = np.concatenate((self.indices_keepdim_a, contfa), axis=None)
        self.list_indices_a_intot = self.permutation(self.ali, self.ala)

        self.bla = np.concatenate((np.arange(len(self.keepdims_a), len(self.keepdims)),
                                   np.arange(len(self.keepdims), self.number_of_dimloop),),
                                  axis=None,
                                  ).copy()
        self.bli = np.concatenate((self.indices_keepdim_b, contfb), axis=None)
        self.list_indices_b_intot = self.permutation(self.bli, self.bla)

        # Gradient
        self.dimfa_grad = self.permutation(permute, self.keepdims)

        self.list_indices_keepdim_a_inout = np.arange(0, len(self.keepdims_a))
        self.reordered_contfa = self.permutation(contfb, contfa)
        self.reordered_keepdim_a = self.permutation(permute[self.list_indices_keepdim_a_inout], self.indices_keepdim_a)
        self.moveaxis_a = np.concatenate((self.reordered_keepdim_a, self.reordered_contfa), axis=None)

        self.list_indices_keepdim_b_inout = np.arange(len(self.keepdims_a), len(self.keepdims))
        self.reordered_contfb = self.permutation(contfa, contfb)
        self.reordered_keepdim_b = self.permutation(permute[self.list_indices_keepdim_b_inout], self.indices_keepdim_b)
        self.moveaxis_b = np.concatenate((self.reordered_keepdim_b, self.reordered_contfb), axis=None)

        self.contfa_grad = permute[self.list_indices_keepdim_b_inout]
        self.contfb_grad = permute[self.list_indices_keepdim_a_inout]

    def looper(self, loopdim):
        """Evil looping function!"""

        inds = self.cartesian_product(*(np.arange(i) for i in loopdim))

        list_indices_a = inds[:, self.list_indices_a_intot]
        a_indices = (list_indices_a * self.list_strides_dimsfa).sum(axis=1)

        list_indices_b = inds[:, self.list_indices_b_intot]
        b_indices = (list_indices_b * self.list_strides_dimsfb).sum(axis=1)

        list_indices_keepdim = self.permutation(
            self.permute, inds[:, : len(self.keepdims)]
        )
        out_indices = (list_indices_keepdim * self.list_strides_keepdim).sum(axis=1)

        return out_indices, a_indices, b_indices

    @staticmethod
    def cartesian_product(*arrays):
        """From https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points"""
        broadcastable = np.ix_(*arrays)
        broadcasted = np.broadcast_arrays(*broadcastable)
        rows, cols = np.prod(broadcasted[0].shape), len(broadcasted)
        dtype = np.result_type(*arrays)

        out = np.empty(rows * cols, dtype=dtype)
        start, end = 0, rows
        for a in broadcasted:
            out[start:end] = a.reshape(-1)
            start, end = end, end + rows
        return out.reshape(cols, rows).T

    @staticmethod
    def cumprod_array(x):
        if len(x) == 0:
            return x
        else:
            # return np.concatenate((np.cumprod(x[:-1][::1])[::-1], [1]))
            return np.concatenate((np.cumprod(x[1:][::-1])[::-1], [1]))

    @staticmethod
    def permutation(perm, arr):
        """Permute column of an array"""

        if perm is None:
            return arr

        _perm = perm.astype(int).flatten().copy()
        _arr = arr.copy()

        rs = False
        if len(_arr.shape) == 1:
            _arr = _arr.reshape(1, -1)
            rs = True
        elif len(_arr.squeeze().shape) >= 3:
            raise RuntimeError()

        res = _arr[:, np.argsort(_perm)]

        if rs:
            return res.flatten()
        else:
            return res

    def Op(self, out, table, arg0, arg1):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out

        str_code = ""

        for i in range(len(self.loopdim)):
            str_code += f"for(int TD_var_{chr(70 + i)}=0; TD_var_{chr(70 + i)}<{self.loopdim[i]}; ++TD_var_{chr(70 + i)})" + "{\n" + i * "    "

        list_indices_keepdim = self.permutation(self.permute, np.arange(len(self.keepdims)))
        str_out_indices = ""
        for i, v in enumerate(list_indices_keepdim):
            str_out_indices += f"TD_var_{chr(70 + v)} * {self.list_strides_keepdim[i]} + "

        str_a_indices = ""
        for i, v in enumerate(self.list_indices_a_intot):
            str_a_indices += f"TD_var_{chr(70 + v)} * {self.list_strides_dimsfa[i]} + "

        str_b_indices = ""
        for i, v in enumerate(self.list_indices_b_intot):
            str_b_indices += f"TD_var_{chr(70 + v)} * {self.list_strides_dimsfb[i]} + "

        str_code += len(
            self.loopdim) * "    " + f"{out.id}[{str_out_indices[:-2]}] += {arg0.id}[{str_a_indices[:-2]}] * {arg1.id}[{str_b_indices[:-2]}];\n"

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
        from keops.python_engine.formulas import Ind

        f = self.children[0]
        g = self.children[1]
        return f.DiffT(
            v,
            TensorDot(
                gradin,
                g,
                Ind(self.dimfa_grad),
                Ind(self.dimfb),
                Ind(self.contfa_grad),
                Ind(self.indices_keepdim_b),
                Ind(self.moveaxis_a),
            ),
        ) + g.DiffT(
            v,
            TensorDot(
                gradin,
                f,
                Ind(self.dimfa_grad),
                Ind(self.dimfa),
                Ind(self.contfb_grad),
                Ind(self.indices_keepdim_a),
                Ind(self.moveaxis_b),
            ),
        )
