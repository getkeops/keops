from keopscore.formulas.Operation import Operation

# /////////////////////////////////////////////////////////////////////////
# ////     Kronecker product                                          ////
# /////////////////////////////////////////////////////////////////////////


class Kron(Operation):
    """
    Kronecker product of two Tensor (with same number od dims). It is design to work as np.kron. Some
    implementation details:
    if A is of size [d0, d1, d2] and B is of size [D0, D1, D2], the result is of size [d0*D0, d1*D1, D2*D2],
        Kron(A,B) == einsum('ijk, lmn -> iljmkn', A, B)
                  == tensordot(A, B,  [d0, d1, d2], [D0, D1, D2, [], [], [0, 2, 1, 3])
    """

    def __new__(cls, A, B, dimsfa, dimsfb):
        from keopscore.formulas import TensorDot

        assert len(dimsfa) == len(dimsfb)
        return TensorDot(
            A,
            B,
            dimsfa,
            dimsfb,
            [],
            [],
            permute=list(range(0, len(dimsfa) * 2, 2))
            + list(range(1, len(dimsfa) * 2, 2)),
        )
