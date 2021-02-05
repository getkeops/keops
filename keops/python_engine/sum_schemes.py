
class sum_scheme:
    def __init__(self, red_formula, dtype):
        self.red_formula = red_formula
    def declare_temporary_accumulator():
        return self.tmp_acc.declare()
    def initialize_temporary_accumulator_first_init(self):
        return ""
    def initialize_temporary_accumulator_block_init(self):
        return ""
    def periodic_accumulate_temporary(self, acc):
        return ""
    def final_operation(self, acc):  
        return ""

class direct_sum(sum_scheme):
    def declare_temporary_accumulator(self):
        return ""
    def initialize_temporary_accumulator(self):
        return ""
    def accumulate_result(self, acc, fout, j):  
        return self.red_formula.ReducePairShort(acc, fout, j)
    
class block_sum(sum_scheme):
    def __init__(self, red_formula, dtype):
        super().__init__(red_formula, dtype)
        self.tmp_acc = c_array("tmp", dtype, red_formula.dimred)
    def initialize_temporary_accumulator(self):
        return self.red_formula.InitializeReduction(self.tmp_acc)
    def initialize_temporary_accumulator_block_init(self):
        return self.initialize_temporary_accumulator()
    def accumulate_result(self, acc, fout, j):    
        return self.red_formula.ReducePairShort(self.tmp_acc, fout, j)
    def periodic_accumulate_temporary(self, acc):
        return c_if( f"({j()}+1)%200",
                        self.red_formula.ReducePair(acc, self.tmp_acc),
                        self.red_formula.InitializeReduction(self.tmp_acc)
                    )
    def final_operation(self, acc):  
        return self.red_formula.ReducePair(acc, self.tmp_acc)

class kahan_scheme(sum_scheme):
    def __init__(self, red_formula, dtype):
        super().__init__(red_formula, dtype)
        self.tmp_acc = c_array("tmp", dtype, red_formula.dim_kahan)
    def initialize_temporary_accumulator(self):    
        return self.tmp_acc.assign(c_zero_float)
    def initialize_temporary_accumulator_first_init(self):
        return self.initialize_temporary_accumulator()
    def accumulate_result(self, acc, fout, j):  
        return self.red_formula.KahanScheme(acc, fout, self.tmp_acc)
        
        
    