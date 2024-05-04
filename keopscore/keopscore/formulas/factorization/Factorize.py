from keopscore.formulas.variables.Var import Var, Var_Impl
from keopscore.formulas.variables.Zero import Zero_Impl
from keopscore.formulas.variables.IntCst import IntCst_Impl
from keopscore.formulas.variables.RatCst import RatCst_Impl
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
                return f"[{f.__repr__()} with {self.aliasvar.__repr__()}={g.__repr__()}]"

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
                if newaliasvar!=aliasvar:
                    newf = f.replace(aliasvar, newaliasvar)
                else:
                    newf = f

                # Evaluation of f
                res += newf(out, table)

                if keopscore.debug_ops:
                    print(f"Finished building code block for {self.__repr__()}")

                res += c_comment(f"// Finished code block for {self.__repr__()}.")
                res = c_block(res)
                return res

            def DiffT_fun_old(self, v, gradin):
                f, g = self.children
                aliasvar = self.aliasvar
                f = f.replace(aliasvar, g)
                return Factorize(f.DiffT(v, gradin), g)
            
            def DiffT_fun(self, v, gradin):
                f = Defactorize(self)
                return DiffT(f,v,gradin).val
                #return f.DiffT(v,gradin)
            
            def DiffT_fun_new(self, v, gradin):
                f, g = self.children
                aliasvar = self.aliasvar
                f = f.replace(aliasvar, g)
                return f.DiffT(v,gradin)

        self.Class = Class

    def __call__(self, f, g):
        return self.Class(f, g)

class DiffT(metaclass=unique_object):
    def __init__(self, f, v, gradin):
        self.val = f.DiffT(v,gradin)


def Factorize_(formula, g, v):
    if isinstance(formula, Factorize_Impl):
        if set.intersection(set(formula.defined_temp_Vars), set(g.Vars_)):
            f_inner, g_inner = formula.children
            v_inner = formula.aliasvar
            return Factorize_Impl_Factory(v_inner)(Factorize_(f_inner, g, v), g_inner)
    res = Factorize_Impl_Factory(v)(formula, g)
    return res


def Factorize(formula, g):
    if (isinstance(g, Var_Impl) and g.ind<0) or isinstance(g, Zero_Impl) or isinstance(g, IntCst_Impl):
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


def Defactorize(f):
    if isinstance(f, Factorize_Impl):
        aliasvar = f.aliasvar
        f,g = f.children
        f = f.replace(aliasvar,g)
        return Defactorize(f)
    newchildren = [Defactorize(child) for child in f.children]
    return type(f)(*newchildren)

def AutoFactorize_new(f):

    def can_factorize(f):
        for cls in [Zero_Impl, IntCst_Impl, RatCst_Impl]:
            if isinstance(f,cls):
                return False
        if isinstance(f, Var_Impl) and f.ind<0:
            return False
        return True
    
    def detect(f, parent, newind):
        if hasattr(f, "visited"):
            f.parents += parent
            if not hasattr(f, "alias") and can_factorize(f):
                f.alias = Var(newind[0], f.dim, 3)
                newind[0] -= 1
        else:
            f.parents = parent
            for child in f.children:
                detect(child, [f], newind)
            f.visited = True  
            f.to_factorize = [] 
    
    def ping_parents(f):
        if hasattr(f, "counter_parents"):
            f.counter_parents += 1
        else:
            f.counter_parents = 1
            for parent in f.parents:
                ping_parents(parent)
    
    def ping_parents_clean(f):
        if hasattr(f, "counter_parents"):
            delattr(f, "counter_parents")
            for parent in f.parents:
                ping_parents_clean(parent)
    
    def get_first_divergent(f):
        g = f
        if f.counter_parents>1:
            return f
        elif f.counter_parents==1:
            for child in f.children:
                if hasattr(child, "counter_parents"):
                    return get_first_divergent(child)
            print("Error....")
        else:
            print("Error....")
    
    def get_common_ancestor(f, root):
        ping_parents(f)
        res = get_first_divergent(root)
        ping_parents_clean(f)
        return res

    def reconstruct(f, root):  
        if hasattr(f,"new"):
            return f.new
        newchildren = [reconstruct(child, root) for child in f.children]
        f.new = type(f)(*newchildren)
        f.to_factorize.reverse()
        for g in f.to_factorize:
            f.new = Factorize_Impl_Factory(g.alias)(f.new,g)
        if hasattr(f, "alias"):
            f.new.alias = f.alias
            common_ancestor = get_common_ancestor(f, root)
            common_ancestor.to_factorize.append(f.new)
            f.new = f.new.alias
        return f.new

    def delattrs(f,*names):
        for name in names:
            if hasattr(f,name):
                delattr(f,name)
                
    def clean(f):
        for child in f.children:
            clean(child)
        delattrs(f,"new",
                    "parents",
                    "new",
                    "to_factorize",
                    "visited",
                    "alias")
        
    # we get a new negative index to start with (negative because it must not refer to an actual input tensor index)
    minind = f.Vars_[0].ind if len(f.Vars_)>0 else 0
    newind = [-1] if minind>=0 else [minind-1]

    f = Defactorize(f)
    detect(f,[],newind)
    g = reconstruct(f,f)
    clean(f)
    clean(g)
    return g

