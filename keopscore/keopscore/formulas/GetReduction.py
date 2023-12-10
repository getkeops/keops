import ast, inspect

import keopscore
import keopscore.formulas
from keopscore.utils.code_gen_utils import get_hash_name
from keopscore.formulas.reductions import *
from keopscore.formulas.maths import *
from keopscore.formulas.complex import *
from keopscore.formulas.variables import *
from keopscore.formulas.autodiff import *
from keopscore.formulas.LinearOperators import *
from keopscore.formulas.factorization import *
from functools import reduce

class GetReduction:
    library = {}

    def __new__(self, red_formula_string, aliases=[]):
        string_id_hash = get_hash_name(
            red_formula_string, aliases, keopscore.auto_factorize
        )
        if string_id_hash in GetReduction.library:
            return GetReduction.library[string_id_hash]
        else:
            self.check_formula(red_formula_string)
            aliases_dict = {}
            for alias in aliases:
                self.check_formula(alias)
                if "=" in alias:
                    varname, var = alias.split("=")
                    aliases_dict[varname] = eval(var)
            reduction = eval(red_formula_string, globals(), aliases_dict)
            if keopscore.auto_factorize:
                formula = reduction.children[0]
                new_formula = AutoFactorize(formula)
                reduction.children[0] = new_formula
            GetReduction.library[string_id_hash] = reduction
            return reduction

    @staticmethod
    def check_formula(string):
        formula_class = dict(inspect.getmembers(keopscore.formulas))
        parsed = ast.parse(string)
        for node in ast.walk(parsed):
            if isinstance(node, ast.Call) and (
                node.func.id not in formula_class.keys()
            ):
                pass
                #print(node.func.id)
                # raise NotImplementedError
