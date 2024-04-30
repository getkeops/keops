from keopscore.formulas.variables import Var, Zero
from keopscore.formulas.variables.IntCst import IntCst_Impl
from keopscore.utils.meta_toolbox import (
    new_c_name,
    c_fixed_size_array,
    c_comment,
    c_instruction_from_string,
    c_block,
)
from keopscore.formulas.Operation import Operation
import keopscore
from keopscore.utils.code_gen_utils import GetInds
from keopscore.utils.unique_object import unique_object


class Factorize_Impl(Operation):
    pass


class Factorize_Impl_Factory(metaclass=unique_object):

    def __init__(self, aliasvar):

        class Class(Factorize_Impl):
            string_id = "Factorize"

            def recursive_str(self):
                f, g = self.children
                return f"{f.__repr__()} with {self.aliasvar.__repr__()}={g.__repr__()}"

            def __init__(self, f, g):
                super().__init__(f, g)
                self.dim = f.dim
                self.aliasvar = aliasvar
                if isinstance(f, Factorize_Impl):
                    self.defined_temp_Vars = f.defined_temp_Vars + [aliasvar]
                else:
                    self.defined_temp_Vars = [aliasvar]

            def __call__(self, out, table):
                """returns the C++ code string corresponding to the evaluation of the formula
                - out is a c_variable in which the result of the evaluation is stored
                - table is the list of c_variables corresponding to actual local variables
                required for evaluation : each Var(ind,*,*) corresponds to table[ind]"""
                from keopscore.formulas.variables.Var import Var

                res = c_comment(f"Starting code block for {self.__repr__()}.")
                if keopscore.debug_ops:
                    print(f"Building code block for {self.__repr__()}")
                    print("out=", out)
                    print("dim of out : ", out.dim)
                    print("table=", table)
                    for v in table:
                        print(f"dim of {v} : ", v.dim)
                if keopscore.debug_ops_at_exec:
                    res += c_instruction_from_string(
                        f'printf("\\n\\nComputing {self.__repr__()} :\\n");\n'
                    )

                f, g = self.children
                aliasvar = self.aliasvar

                # Evaluation of g
                # We first create a new c_array to store the result of the child operation.
                # This c_array must have a unique name in the code, to avoid conflicts
                # when we will recursively evaluate nested operations.
                template_string_id = "out_" + g.string_id.lower()
                outg_name = new_c_name(template_string_id)
                outg = c_fixed_size_array(out.dtype, g.dim, outg_name)
                # Now we append into string the C++ code to declare the array
                res += outg.declare()
                # Now we evaluate g and append the result into string
                res += g(outg, table)

                # we put a new entry for the temporary variable in the table.
                table.append(outg)
                # This will fix the index for the temp variable. So we must finally
                # change the index of this temp variable to match its position in table.
                newind = len(table) - 1
                newaliasvar = Var(newind, aliasvar.dim, aliasvar.cat)
                newf = f.replace(aliasvar, newaliasvar)

                # Evaluation of f
                res += newf(out, table)

                if keopscore.debug_ops:
                    print(f"Finished building code block for {self.__repr__()}")

                res += c_comment(f"// Finished code block for {self.__repr__()}.")
                res = c_block(res)
                return res

            def DiffT(self, v, gradin):
                f, g = self.children
                aliasvar = self.aliasvar
                f = f.replace(aliasvar, g)
                return Factorize(f.DiffT(v, gradin), g)

        self.Class = Class

    def __call__(self, f, g):
        return self.Class(f, g)


def Factorize_(formula, g, v):
    if isinstance(formula, Factorize_Impl):
        if set.intersection(set(formula.defined_temp_Vars), set(g.Vars_)):
            f_inner, g_inner = formula.children
            v_inner = formula.aliasvar
            return Factorize_Impl_Factory(v_inner)(Factorize_(f_inner, g, v), g_inner)
    res = Factorize_Impl_Factory(v)(formula, g)
    return res


def Factorize(formula, g):
    if type(g) in (Var, Zero, IntCst_Impl):
        return formula
    # we get a new negative index (negative because it must not refer to an actual input tensor index)
    inds = GetInds(formula.Vars_)
    minind = min(inds) if len(inds) > 0 else 0
    newind = -1 if minind >= 0 else minind - 1
    v = Var(newind, g.dim, 3)
    newformula, cnt = formula.replace_and_count(g, v)
    if cnt > 1:
        return Factorize_(newformula, g, v)
    else:
        return formula


def AutoFactorize(formula):
    def RecSearch(formula, g):
        newformula = Factorize(formula, g)
        if newformula != formula:
            return newformula
        for child in g.children:
            newformula = RecSearch(formula, child)
            if newformula != formula:
                return newformula
        return formula

    newformula = RecSearch(formula, formula)
    if newformula != formula:
        return AutoFactorize(newformula)
    return formula
