from keopscore.utils.code_gen_utils import (
    c_tensor,
    c_zero_float,
    c_if,
    c_variable,
)


class Sum_Scheme:
    def __init__(self, red_formula, dtype, shapered=None):
        self.red_formula = red_formula
        if shapered is None:
            self.shapered = red_formula.shapered
        else:
            self.shapered = shapered

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


class direct_sum(Sum_Scheme):
    def declare_temporary_accumulator(self):
        return ""

    def initialize_temporary_accumulator(self):
        return ""

    def accumulate_result(self, acc, fout, j, hack=False):
        return self.red_formula.ReducePairShort(acc, fout, j)


class block_sum(Sum_Scheme):
    def __init__(self, red_formula, dtype, dimred=None):
        super().__init__(red_formula, dtype, dimred)
        self.tmp_acc = c_tensor(dtype, self.shapered, "tmp")

    def initialize_temporary_accumulator(self):
        return (
            "int period_accumulate = ny<10 ? 100 : sqrt(ny);\n"
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
        res = self.red_formula.ReducePair(acc, self.tmp_acc)
        print("final_operation")
        print(res)
        input()
        return res


class kahan_scheme(Sum_Scheme):
    def __init__(self, red_formula, dtype, dimred=None):
        super().__init__(red_formula, dtype, dimred)
        self.tmp_acc = c_tensor(dtype, red_formula.shape_kahan, "tmp")

    def initialize_temporary_accumulator(self):
        return self.tmp_acc.assign(c_zero_float)

    def initialize_temporary_accumulator_first_init(self):
        return self.initialize_temporary_accumulator()

    def accumulate_result(self, acc, fout, j, hack=False):
        return self.red_formula.KahanScheme(acc, fout, self.tmp_acc)
