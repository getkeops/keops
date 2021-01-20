from utils import *

class code_writer:
    def __call__(self):
        f = open(self.source_file)
        string = f.read()
        f.close()
        return self.format_code(string)
    def format_code(self, string):
        return string.format(**self.dict_format)    
    
class CpuReduc(code_writer):
    def __init__(self, red_formula, dtype, dtypeacc):
        self.source_file = "link_autodiff.cpp"
        formula = red_formula.formula
        self.dtype = dtype
        self.dtypeacc = dtypeacc
        pdtype = pointer(dtype)
        pdtypeacc = pointer(dtypeacc)
        formula = red_formula.formula
        acc = c_variable("acc",pdtypeacc)
        xi = c_variable("xi",pdtype)
        yj = c_variable("yj",pdtype)
        fout = c_variable("fout",pdtype)
        out = c_variable("out",pdtype)
        args = c_variable("args",pointer(pdtype))
        zero = c_variable("0","int")
        i = c_variable("i","int")
        j = c_variable("j","int")
        pp = c_variable("pp",pdtype)
        outi = c_variable(f"(out + i * {red_formula.dim})", out.dtype) 
        inds = GetInds(formula._Vars)
        table = [None]*(max(inds)+1)
        loadp, table = load_vars(red_formula.dimsp, red_formula.indsp, zero, pp, args, table)
        loadx, table = load_vars(red_formula.dimsx, red_formula.indsi, i, xi, args, table)
        loady, table = load_vars(red_formula.dimsy, red_formula.indsj, j, yj, args, table)
        self.dict_format = { 
            "TYPE" : self.dtype,
            "TYPEACC" : self.dtypeacc,
            "DIMRED" : red_formula.dimred,
            "DIMP" : red_formula.dimp,
            "DIMX" : red_formula.dimx,
            "DIMY" : red_formula.dimy,
            "DIMOUT" : red_formula.dim,
            "DIMFOUT" : formula.dim,
            "InitializeReduction" : red_formula.InitializeReduction(acc),
            "ReducePairShort" : red_formula.ReducePairShort(acc, fout, j),
            "FinalizeOutput" : red_formula.FinalizeOutput(acc, outi, i),
            "definep" : declare_array(pp,red_formula.dimp),
            "definex" : declare_array(xi,red_formula.dimx),
            "definey" : declare_array(yj,red_formula.dimy),
            "loadp" : loadp,
            "loadx" : loadx,
            "loady" : loady,
            "call" : formula(fout,table),
        }