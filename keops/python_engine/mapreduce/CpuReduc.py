from keops.python_engine.mapreduce.MapReduce import MapReduce
from keops.python_engine.mapreduce.CpuAssignZero import CpuAssignZero
from keops.python_engine.utils.code_gen_utils import c_include, signature_list, call_list
from keops.python_engine.compilation import Cpu_link_compile
from keops.python_engine import use_jit, debug_ops


class CpuReduc(MapReduce, Cpu_link_compile):
    # class for generating the final C++ code, Cpu version

    AssignZero = CpuAssignZero

    def __init__(self, *args):
        if use_jit:
            raise ValueError("JIT compiling not yet implemented in Cpu mode")
        MapReduce.__init__(self, *args)
        Cpu_link_compile.__init__(self)

    def get_code(self, for_jit=False):
        
        if for_jit:
            raise ValueError("JIT compiling not yet implemented in Cpu mode")

        super().get_code()

        i = self.i
        j = self.j
        dtype = self.dtype
        red_formula = self.red_formula
        fout = self.fout
        outi = self.outi
        acc = self.acc
        args = self.args
        argshapes = self.argshapes
        table = self.varloader.direct_table(args, i, j)
        sum_scheme = self.sum_scheme

        headers = ["cmath", "omp.h"]
        if debug_ops:
            headers.append("iostream")
        self.headers += c_include(*headers)

        self.code = f"""
                        {self.headers}
                        int CpuConv(int nx, int ny, {dtype}* out, {signature_list(args)}) {{
                            #pragma omp parallel for
                            for (int i = 0; i < nx; i++) {{
                                {fout.declare()}
                                {acc.declare()}
                                {sum_scheme.declare_temporary_accumulator()}
                                {red_formula.InitializeReduction(acc)}
                                {sum_scheme.initialize_temporary_accumulator()}
                                for (int j = 0; j < ny; j++) {{
                                    {red_formula.formula(fout,table)}
                                    {sum_scheme.accumulate_result(acc, fout, j)}
                                    {sum_scheme.periodic_accumulate_temporary(acc, j)}
                                }}
                                {sum_scheme.final_operation(acc)}
                                {red_formula.FinalizeOutput(acc, outi, i)}
                            }}
                            return 0;
                        }}
                        
                        extern "C" int launch_keops(int nx, int ny, int device_id, int **ranges, {dtype}* out, {signature_list(args)}, {signature_list(argshapes)}) {{
                            
                            if ({red_formula.tagJ}==0) {{
                                int tmp = ny;
                                ny = nx;
                                nx = tmp;
                            }}
                            
                            return CpuConv(nx, ny, out, {call_list(args)});
                        }}
                    """
