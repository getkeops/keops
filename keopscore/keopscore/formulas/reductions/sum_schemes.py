from keopscore.utils.meta_toolbox.c_instruction import (
    c_instruction,
    c_empty_instruction,
    c_instruction_from_string,
)
from keopscore.utils.code_gen_utils import (
    c_array,
    c_zero_float,
    c_if,
    c_variable,
)


class Sum_Scheme:
    def __init__(self, red_formula, dtype, dimred=None):
        self.red_formula = red_formula
        if dimred is None:
            self.dimred = red_formula.dimred
        else:
            self.dimred = dimred

    def declare_temporary_accumulator(self):
        return self.tmp_acc.declare()

    def initialize_temporary_accumulator_first_init(self):
        return c_empty_instruction

    def initialize_temporary_accumulator_block_init(self):
        return c_empty_instruction

    def periodic_accumulate_temporary(self, acc, j):
        return c_empty_instruction

    def final_operation(self, acc):
        return c_empty_instruction


class direct_sum(Sum_Scheme):
    def declare_temporary_accumulator(self):
        return c_empty_instruction

    def initialize_temporary_accumulator(self):
        return c_empty_instruction

    def accumulate_result(self, acc, fout, j, hack=False):
        tmp = self.red_formula.ReducePairShort(acc, fout, j)
        return self.red_formula.ReducePairShort(acc, fout, j)


class block_sum(Sum_Scheme):
    def __init__(self, red_formula, dtype, dimred=None):
        super().__init__(red_formula, dtype, dimred)
        self.tmp_acc = c_array(dtype, self.dimred, "tmp")

    def initialize_temporary_accumulator(self):
        return c_instruction_from_string(
            "signed long int period_accumulate = ny<10 ? 100 : sqrt(ny)"
        ) + self.red_formula.InitializeReduction(self.tmp_acc)

    def initialize_temporary_accumulator_block_init(self):
        return self.red_formula.InitializeReduction(self.tmp_acc)

    def accumulate_result(self, acc, fout, j, hack=False):
        tmp_acc = acc if hack else self.tmp_acc
        return self.red_formula.ReducePairShort(tmp_acc, fout, j)

    def periodic_accumulate_temporary(self, acc, j):
        condition = c_variable("bool", f"!(({j.id}+1)%period_accumulate)")
        return c_if(
            condition,
            self.red_formula.ReducePair(acc, self.tmp_acc)
            + self.red_formula.InitializeReduction(self.tmp_acc),
        )

    def final_operation(self, acc):
        return self.red_formula.ReducePair(acc, self.tmp_acc)


class kahan_scheme(Sum_Scheme):
    def __init__(self, red_formula, dtype, dimred=None):
        super().__init__(red_formula, dtype, dimred)
        self.tmp_acc = c_array(dtype, red_formula.dim_kahan, "tmp")

    def initialize_temporary_accumulator(self):
        return self.tmp_acc.assign(c_zero_float)

    def initialize_temporary_accumulator_first_init(self):
        return self.initialize_temporary_accumulator()

    def accumulate_result(self, acc, fout, j, hack=False):
        return self.red_formula.KahanScheme(acc, fout, self.tmp_acc)
