import ast, inspect

import keops.formulas
from keops.utils.code_gen_utils import get_hash_name
from keops.formulas.reductions import *
from keops.formulas.maths import *
from keops.formulas.complex import *
from keops.formulas.variables import *
from keops.formulas.autodiff import *


class GetReduction:
    library = {}

    def __new__(self, red_formula_string, aliases=[]):
        string_id_hash = get_hash_name(red_formula_string, aliases)
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
            GetReduction.library[string_id_hash] = reduction
            return reduction

    @staticmethod
    def check_formula(string):
        formula_class = dict(inspect.getmembers(keops.formulas))
        parsed = ast.parse(string)
        for node in ast.walk(parsed):
            if isinstance(node, ast.Call) and (
                node.func.id not in formula_class.keys()
            ):
                print(node.func.id)
                # raise NotImplementedError
