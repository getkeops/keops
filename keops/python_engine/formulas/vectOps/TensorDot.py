from keops.python_engine.formulas.Operation import Operation
import numpy as np


####################################
######  Tensor Dot Product     #####
####################################
from keops.python_engine.utils.code_gen_utils import c_variable


class TensorDot(Operation):
    string_id = "TensorDot"

    def __init__(self, fa, fb, dimsfa, dimsfb, contfa, contfb, permute=None):

        assert (dimsfb[contfb] == dimsfa[contfa])

        assert (fa.dim == dimsfa.prod())
        assert (fb.dim == dimsfb.prod())

        super().__init__(fa, fb)

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
        self.list_strides_keepdim = self.cumprod_array(self.permutation(permute, self.keepdims))

        self.dim = fa.dim * fb.dim
        self.dim = int(self.dim / self.contdims.prod() ** 2) if len(contfa)else 1

        # loop
        self.loopdim = np.concatenate((self.keepdims, self.contdims_a))
        self.dimloop = self.loopdim.prod()
        self.number_of_dimloop = len(dimsfa) - len(contfa) + len(dimsfb);

        ala = np.concatenate( (np.arange(0, len(self.keepdims_a)), np.arange(len(self.keepdims), self.number_of_dimloop)), axis=None);
        ali = np.concatenate((self.indices_keepdim_a, contfa), axis=None);
        self.list_indices_a_intot = self.permutation(ali, ala);

        bla = np.concatenate((np.arange(len(self.keepdims_a), len(self.keepdims)), np.arange(len(self.keepdims), self.number_of_dimloop)), axis=None);
        bli = np.concatenate((self.indices_keepdim_b, contfb), axis=None);
        self.list_indices_b_intot = self.permutation(bli, bla);

        if permute is None:
            permute = np.arange(self.dim)

        self.permute = permute

    def looper(self, loopdim):
        """Evil looping function!"""

        inds = self.cartesian_product(*(np.arange(i) for i in loopdim))

        list_indices_a = inds[:, self.list_indices_a_intot]
        a_indices = (list_indices_a * self.list_strides_dimsfa).sum(axis=1)

        list_indices_b = inds[:, self.list_indices_b_intot]
        b_indices = (list_indices_b * self.list_strides_dimsfb).sum(axis=1)

        list_indices_keepdim = self.permutation(self.permute, inds[:,:len(self.keepdims)])
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
            return np.concatenate((np.cumprod(x[1:][::-1])[::-1], [1]))

    @staticmethod
    def permutation(perm, arr):
        """Permute column of an array"""
        if perm is None:
            return arr
        perm = perm.astype(int)

        rs = False
        if len(arr.shape) == 1:
            arr = arr.reshape(1, -1)
            rs = True
        elif len(arr.shape) > 2:
            raise RuntimeError

        def swap(_arr, _i, _j):
            tmp = _arr[:, _i]
            _arr[:, _i] = _arr[:, _j]
            _arr[:, _j] = tmp
            return _arr

        n = arr.shape[1]

        for i in range(n):
            j = perm[i]
            while j < i:
                j = perm[j]
            arr = swap(arr, i, j)

        if rs:
            return arr.reshape(-1)
        else:
            return arr

    def Op(self, out, table, arg0, arg1):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out

        out_indices, a_indices, b_indices = self.looper(self.loopdim)
        str_code = ""
        for i in range(len(out_indices)):
            str_code += f"                            " + \
                   f"{out.id}[{out_indices[i]}] += {arg0.id}[{a_indices[i]}] * {arg1.id}[{b_indices[i]}];\n"

        return f"""
                    #if C_CONTIGUOUS     // row major
                                    
                        for (int i = 0; i < {out.dim}; i++)
                            {out.id}[i] = ({out.dtype})(0.0f);
                    
                        {str_code}
                    #else               // column major
                        
                    #endif
                """

    def DiffT(self, v, gradin):
        from keops.python_engine.formulas import MatVecMult, VecMatMult
        f = self.children[0]
        g = self.children[1]
        return f.Grad(v, MatVecMult(gradin, g)) + g.Grad(v, VecMatMult(f, gradin))
