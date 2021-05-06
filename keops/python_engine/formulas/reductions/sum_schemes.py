from keops.python_engine.code_gen_utils import c_array, c_zero_float, c_if


class Sum_Scheme:
    def __init__(self, red_formula, dtype):
        self.red_formula = red_formula
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
    def accumulate_result(self, acc, fout, j):  
        return self.red_formula.ReducePairShort(acc, fout, j)
    
class block_sum(Sum_Scheme):
    def __init__(self, red_formula, dtype):
        super().__init__(red_formula, dtype)
        self.tmp_acc = c_array(dtype, red_formula.dimred, "tmp")
    def initialize_temporary_accumulator(self):
        return "int period_accumulate = ny<10 ? 100 : sqrt(ny);\n" + self.red_formula.InitializeReduction(self.tmp_acc)
    def initialize_temporary_accumulator_block_init(self):
        return self.red_formula.InitializeReduction(self.tmp_acc)
    def accumulate_result(self, acc, fout, j):    
        return self.red_formula.ReducePairShort(self.tmp_acc, fout, j)
    def periodic_accumulate_temporary(self, acc, j):
        return c_if( f"!(({j.id}+1)%period_accumulate)",
                        self.red_formula.ReducePair(acc, self.tmp_acc),
                        self.red_formula.InitializeReduction(self.tmp_acc)
                    )
    def final_operation(self, acc):  
        return self.red_formula.ReducePair(acc, self.tmp_acc)

class kahan_scheme(Sum_Scheme):
    def __init__(self, red_formula, dtype):
        super().__init__(red_formula, dtype)
        self.tmp_acc = c_array(dtype, red_formula.dim_kahan, "tmp")
    def initialize_temporary_accumulator(self):    
        return self.tmp_acc.assign(c_zero_float)
    def initialize_temporary_accumulator_first_init(self):
        return self.initialize_temporary_accumulator()
    def accumulate_result(self, acc, fout, j):  
        return self.red_formula.KahanScheme(acc, fout, self.tmp_acc)
        
        
    
