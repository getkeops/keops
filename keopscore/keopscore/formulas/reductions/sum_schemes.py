from keopscore.utils.code_gen_utils import (
    c_array,
    c_zero_float,
    c_if,
    c_variable,
)


class Sum_Scheme:

    def __init__(self, red_formula, dtype, dtypeacc, i, dimred=None):
        self.red_formula = red_formula
        if dimred is None:
            self.dimred = red_formula.dimred
        else:
            self.dimred = dimred

        self.acc = c_array(dtypeacc, red_formula.dimred, "acc")
        self.acctmp = c_array(dtypeacc, red_formula.dimred, "acctmp")
        self.fout = c_array(dtype, red_formula.formula.dim, "fout")
        self.outi = c_array(dtype, red_formula.dim, f"(out + {i} * {red_formula.dim})")

    def declare_formula_out(self):
        return self.fout.declare()

    def declare_accumulator(self):
        return self.acc.declare()

    def declare_temporary_accumulator(self):
        return self.tmp_acc.declare()

    def initialize_temporary_accumulator_first_init(self):
        return ""

    def initialize_temporary_accumulator_block_init(self):
        return ""

    def periodic_accumulate_temporary(self, acc, j):
        return ""

    def final_operation(self, acc):
        return ""

    def FinalizeOutput(self, acc, outi, i):
        return self.red_formula.FinalizeOutput(acc, outi, i)


class direct_sum(Sum_Scheme):
    def declare_temporary_accumulator(self):
        return ""

    def initialize_temporary_accumulator(self):
        return ""

    def accumulate_result(self, acc, fout, j, hack=False):
        return self.red_formula.ReducePairShort(acc, fout, j)


class direct_acc(Sum_Scheme):

    def __init__(self, red_formula, dtype, dtypeacc, i, dimred=None):
        super().__init__(red_formula, dtype, dtypeacc, i, dimred)
        self.fout = self.acc = self.outi
        self.fout.set_assign_mode("add_assign")

    def declare_formula_out(self):
        return ""

    def declare_accumulator(self):
        return ""

    def declare_temporary_accumulator(self):
        return ""

    def initialize_temporary_accumulator(self):
        return ""

    def accumulate_result(self, acc, fout, j, hack=False):
        return ""

    def FinalizeOutput(self, acc, outi, i):
        return ""


class block_sum(Sum_Scheme):
    def __init__(self, red_formula, dtype, dtypeacc, i, dimred=None):
        super().__init__(red_formula, dtype, dtypeacc, i, dimred)
        self.tmp_acc = c_array(dtype, self.dimred, "tmp")

    def initialize_temporary_accumulator(self):
        return (
            "signed long int period_accumulate = ny<10 ? 100 : sqrt(ny);\n"
            + self.red_formula.InitializeReduction(self.tmp_acc)
        )

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
    def __init__(self, red_formula, dtype, dtypeacc, i, dimred=None):
        super().__init__(red_formula, dtype, dtypeacc, i, dimred)
        self.tmp_acc = c_array(dtype, red_formula.dim_kahan, "tmp")

    def initialize_temporary_accumulator(self):
        return self.tmp_acc.assign(c_zero_float)

    def initialize_temporary_accumulator_first_init(self):
        return self.initialize_temporary_accumulator()

    def accumulate_result(self, acc, fout, j, hack=False):
        return self.red_formula.KahanScheme(acc, fout, self.tmp_acc)
