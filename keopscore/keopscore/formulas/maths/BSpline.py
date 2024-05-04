from keopscore.formulas.Operation import Operation
from keopscore.formulas.maths.Extract import Extract
from keopscore.utils.meta_toolbox import c_variable, c_for_loop, c_zero_float
from keopscore.utils.meta_toolbox import c_fixed_size_array, c_comment
from keopscore.utils.misc_utils import KeOps_Error

# //////////////////////////////////////////////////////////////
# ////       BSPLINE VECTOR : BSPLINE<knots,x,order>        ////
# //////////////////////////////////////////////////////////////

# N.B.: BSpline_Impl creates a vector of size "n_knots - 1"
# since we need this amount of space as an intermediate buffer for the computation
# of the final "n_knots - order - 1" coefficients.
# In the final BSpline operation, we simply use "extract" to discard
# the "order" irrelevant coefficients.


class BSpline_Impl(Operation):

    # parameters for testing the operation (optional)
    enable_test = True  # enable testing for this operation
    disable_testgrad = True  # disable testing of the gradient
    nargs = 2  # number of arguments
    test_argdims = [5, 1]  # dimensions of arguments for testing
    test_params = [3]  # values of parameters for testing
    torch_op = None  # equivalent PyTorch operation


class BSpline_Impl_Factory:

    def __init__(self, order):

        class Class(BSpline_Impl):
            string_id = "BSpline_Impl"

            def __init__(self, knots, x):

                if order < 0:
                    KeOps_Error(
                        f"The order of a BSpline must be >= 0, but received {order}."
                    )

                if x.dim != 1:
                    KeOps_Error(
                        f"BSplines must be sampled at scalar values, "
                        f"but received a vector of size {x.dim}."
                    )

                if knots.dim < order + 2:
                    KeOps_Error(
                        f"BSplines of order {order} require at least {order+2} knots, "
                        f"but only received a vector of size {knots.dim}."
                    )

                super().__init__(knots, x)
                self.order = order
                self.dim = knots.dim - 1

            def Op(self, out, table, knots, inX):
                # returns the atomic piece of c++ code to evaluate the function on arg and return
                # the result in out
                x = c_variable("float")
                ratio_1 = c_variable("float")
                ratio_2 = c_variable("float")
                init_loop, j = c_for_loop(0, self.dim, 1, pragma_unroll=True)
                outer_loop, k = c_for_loop(1, self.order + 1, 1, pragma_unroll=True)
                inner_loop, i = c_for_loop(
                    0, c_variable("int", self.dim) - k, 1, pragma_unroll=True
                )

                inner_loop_code = c_comment(
                    'Compute the second ratio "omega_i+1,k" = (x - t_i+1) / (t_i+k+1 - t_i+1):'
                )
                +ratio_2.assign(
                    (knots[i + 1] < knots[i + 1 + k]).ternary(
                        (x - knots[i + 1]) / (knots[i + 1 + k] - knots[i + 1]),
                        c_zero_float,
                    )
                )
                +c_comment("In place computation of B_i,k+1(x) as")
                +c_comment(
                    "omega_i,k(x) * B_i,k(x) + (1 - omega_i+1,k(x)) * B_i+1,k(x)"
                )
                +out[i].assign(
                    ratio_1 * out[i]
                    + (c_variable("float", "1.0f") - ratio_2) * out[i + 1]
                )
                +c_comment("Update the ratios as i -> i+1:")
                +ratio_1.assign(ratio_2)

                code = c_comment("The sampling 'time':")
                +x.declare_assign(inX[0])
                +c_comment(
                    "Order 1 Spline: one-hot encoding of whether t[j] <= x < t[j+1]"
                )
                +init_loop(out[j].assign((knots[j] <= x).logical_and(x < knots[j + 1])))

                +c_comment("Recursive De Boor's algorithm:")
                +ratio_1.declare()
                +ratio_2.declare()
                +outer_loop(
                    c_comment(
                        f'Compute the first ratio "omega_i,k" = (x - t_i) / (t_i+k - t_i) for i=0:'
                    )
                    + ratio_1.assign(
                        (knots[0] < knots[k]).ternary(
                            (x - knots[0]) / (knots[k] - knots[0]),
                            c_zero_float,
                        )
                    )
                    + c_comment("Loop over out[0:len(out)-k]")
                    + inner_loop(inner_loop_code)
                )
                return code

            def GradFun(self, v, gradin):
                raise NotImplementedError(
                    "KeOps BSplines are not yet differentiable."
                    "Please consider disabling autograd for the relevant variables"
                    "with 'x.detach()' "
                    "or submit a feature request at https://github.com/getkeops/keops/issues."
                )

        self.Class = Class

    def __call__(self, knots, x):
        return self.Class(knots, x)


# TODO Jean: The two ways of defining an alias seem equivalent to me...
#            Which one should we use?
if True:

    def BSpline(knots, x, order):
        return Extract(BSpline_Impl_Factory(order)(knots, x), 0, knots.dim - order - 1)

else:

    class BSpline:
        def __new__(cls, knots, x, order):
            return Extract(
                BSpline_Impl_Factory(order)(knots, x), 0, knots.dim - order - 1
            )

        enable_test = False
