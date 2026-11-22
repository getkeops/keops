import ast
import inspect
import keyword
import re

import keopscore.config
from keopscore.utils.messages import KeOps_Print
from keopscore.utils.code_gen_utils import get_hash_name
from keopscore.formulas import *


class GetReduction:
    library = {}

    def __new__(self, red_formula_string, aliases=[]):
        string_id_hash = get_hash_name(
            red_formula_string,
            aliases,
            keopscore.config.reduction.get_auto_factorize(),
        )
        if string_id_hash in GetReduction.library:
            return GetReduction.library[string_id_hash]
        else:
            aliases_dict = {}
            keyword_map = {}
            reserved_text = red_formula_string + " " + " ".join(aliases)
            for idx, alias in enumerate(aliases):
                if "=" in alias:
                    varname, var = alias.split("=", 1)
                    varname = varname.strip()
                    var = var.strip()
                    self.check_formula(var)
                    if keyword.iskeyword(varname):
                        safe_name = f"_keops_kw_{idx}_{varname}"
                        while safe_name in reserved_text:
                            safe_name = "_" + safe_name
                        keyword_map[varname] = safe_name
                        varname = safe_name
                    aliases_dict[varname] = eval(var)
                else:
                    self.check_formula(alias)
            eval_formula_string = red_formula_string
            if keyword_map:
                pattern = re.compile(
                    r"\b(?:" + "|".join(map(re.escape, keyword_map)) + r")\b"
                )
                eval_formula_string = pattern.sub(
                    lambda m: keyword_map[m.group(0)], eval_formula_string
                )
            self.check_formula(eval_formula_string)
            reduction = eval(eval_formula_string, globals(), aliases_dict)
            if keopscore.config.reduction.get_auto_factorize():
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
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and (node.func.id not in formula_class.keys())
            ):
                KeOps_Print(node.func.id)
                # raise NotImplementedError
