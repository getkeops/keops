from keopscore.binders.nvrtc.Gpu_link_compile import Gpu_link_compile
from keopscore.formulas.reductions.sum_schemes import block_sum, kahan_scheme
from keopscore.mapreduce.gpu.GpuAssignZero import GpuAssignZero
from keopscore.mapreduce.MapReduce import MapReduce
from keopscore.utils.code_gen_utils import c_variable, c_array, use_pragma_unroll
from keopscore.utils.misc_utils import KeOps_Error


class GpuReduc2D(MapReduce, Gpu_link_compile):
    # class for generating the final C++ code, Gpu version

    AssignZero = GpuAssignZero

    def __init__(self, *args):
        MapReduce.__init__(self, *args)
        Gpu_link_compile.__init__(self)
        self.dimy = self.varloader.dimy

    def get_code(self):

        super().get_code()

        i = self.i
        j = self.j
        red_formula = self.red_formula
        dtype = self.dtype
        dimin = red_formula.dimred
        dimout = red_formula.dim
        varloader = self.varloader
        dtypeacc = self.dtypeacc
        acc2 = c_array(dtypeacc, dimin, "acc2")

        inloc = c_array(dtype, dimin, f"(in + (tid+y*nx)*{dimin})")
        outloc = c_array(dtype, dimout, f"(out+tid*{dimout})")

        dimsx = varloader.dimsx
        dimsy = varloader.dimsy
        dimsp = varloader.dimsp
        indsi = varloader.indsi
        indsj = varloader.indsj
        indsp = varloader.indsp
        dimx = sum(dimsx)
        dimy = sum(dimsy)
        dimp = sum(dimsp)
        dimred = red_formula.dimred
        dimfout = red_formula.formula.dim

        fout = c_array(dtype, dimfout, "fout")
        param_loc = c_array(dtype, dimp, "param_loc")
        xi = c_array(dtype, dimx, "xi")

        sum_scheme = self.sum_scheme

        # N.B. To be consistent with the convention used in GpuConv1D, when SUM_SCHEME == BLOCK_SUM=1 we accumulate results in TYPE
        # instead of __TYPEACC__ in each block, __TYPEACC__ will be used only to sum up results from each block
        if isinstance(sum_scheme, block_sum):
            acc = c_array(dtype, dimred, "acc")
        elif isinstance(sum_scheme, direct_sum):
            acc = c_array(dtypeacc, dimred, "acc")
        else:
            KeOps_Error("incorrect reduction scheme")

        yjloc = c_array(dtype, varloader.dimy, f"(yj + threadIdx.x * {varloader.dimy})")
        arg = self.arg
        args = self.args
        yjrel = c_array(dtype, dimy, "yjrel")
        table = varloader.table(self.xi, yjrel, self.param_loc)

        jrelloc = c_variable("int", "(blockDim.x*blockIdx.y+jrel)")

        tid = c_variable("int", "tid")

        self.code = f"""
                          
                        {self.headers}
                        
                        extern "C" __global__ void reduce2D({dtype} *in, {dtype} *out, int sizeY, int nx) {{
                            /* Function used as a final reduction pass in the 2D scheme,
                             * once the block reductions have been made.
                             * Takes as input:
                             * - in,  a  sizeY * (nx * DIMIN ) array
                             * - out, an          nx * DIMOUT   array
                             *
                             * Computes, in parallel, the "columnwise"-reduction (which correspond to lines of blocks)
                             * of *in and stores the result in out.
                             */
                            int tid = blockIdx.x * blockDim.x + threadIdx.x;

                            /* As shown below, the code that is used to store the block-wise sum
                              "tmp" in parallel is:
                                if(i<nx)
                                    for(int k=0; k<DIMX1; k++)
                                        (*px)[blockIdx.y*DIMX1*nx+i*DIMX1+k] = tmp[k];
                            */

                            /* // This code should be a bit more efficient (more parallel) in the case
                               // of a simple "fully parallel" reduction op such as "sum", "max" or "min"
                            TYPE res = 0;
                            if(tid < nx*DIMVECT) {{
                                for (int i = 0; i < sizeY; i++)
                                    res += in[tid + i*nx*DIMVECT]; // We use "+=" as a reduction op. But it could be anything, really!
                                // res = in[tid+ nx* DIMVECT];
                                out[tid] = res;
                            }}
                            */

                            // However, for now, we use a "vectorized" reduction op.,
                            // which can also handle non-trivial reductions such as "LogSumExp"
                            {acc2.declare()}
                            {red_formula.InitializeReduction(acc2)} // acc = 0
                            if(tid < nx) {{
                                for (int y = 0; y < sizeY; y++) {{
                                    {red_formula.ReducePair(acc2, inloc)} // acc += in[(tid+y*nx) *DIMVECT : +DIMVECT]; 
                                }}
                                {red_formula.FinalizeOutput(acc2, outloc, tid)}
                            }}
                        }}
                        
                        
                        
                        
                        extern "C" __global__ void GpuConv2DOnDevice(int nx, int ny, {dtype} *out, {dtype} **{arg.id}) {{
                            
                            {fout.declare()}
                            
                            // Load the parameter vector in the Thread Memory, for improved efficiency
                            {param_loc.declare()}
                            {varloader.load_vars("p", param_loc, args)} // load parameters variables from global memory to local thread memory
                            
                            // Weird syntax to create a pointer in shared memory.
                            extern __shared__ char yj_char[];
                            {dtype}* const yj = reinterpret_cast<{dtype}*>(yj_char);
                            
                            // Step 1 : Load in Thread Memory the information needed in the current line ---------------------------
                            int i = blockIdx.x * blockDim.x + threadIdx.x;
                            {xi.declare()}
                            
                            {acc.declare()}
                            
                            {sum_scheme.tmp_acc.declare()} //# if isinstance(sum_scheme, kahan_scheme) else ""
                            
                            if(i<nx) {{ // we will compute outi only if i is in the range
                                {red_formula.InitializeReduction(acc)} // acc = 0
                                {sum_scheme.tmp_acc.assign(c_zero_float) if isinstance(sum_scheme, kahan_scheme) else ""}
                                
                                // Load xi from device global memory.
                                // Remember that we use an interleaved memory scheme where
                                // xi = [ x1i, x2i, x3i, ... ].                  
                            	{varloader.load_vars('i', xi, args, row_index=i)} // load xi variables from global memory to local thread memory
                            }}
                            
                            // Step 2 : Load in Shared Memory the information needed in the current block of the product -----------
                            // In the 1D scheme, we use a loop to run through the line.
                            // In the 2D scheme presented here, the computation is done in parallel wrt both lines and columns.
                            // Hence, we use "blockId.y" to get our current column number.
                            int j = blockIdx.y * blockDim.x + threadIdx.x; // Same blockDim in x and y : squared tiles.
                            if(j<ny) {{ // we load yj from device global memory only if j<ny
                                {varloader.load_vars("j", yjloc, args, row_index=j)} // load yj variables from global memory to shared memory
                            }}
                            // More precisely : the j-th line of py is loaded to yj, at a location which depends on the
                            // current threadId.
                            
                            __syncthreads(); // Make sure nobody lags behind
                            
                            // Step 3 : Once the data is loaded, execute fun --------------------------------------------------------
                            // N.B.: There's no explicit summation here. Just calls to fun, which *accumulates* the results
                            //       along the line, but does not *have* to use a "+=" as reduction operator.
                            //       In the future, we could provide other reductions: max, min, ... whatever's needed.
                            if(i<nx) {{ // we compute x1i only if needed
                                {dtype}* yjrel = yj; // Loop on the columns of the current block.
                                for(int jrel = 0; (jrel<blockDim.x) && ((blockDim.x*blockIdx.y+jrel)< ny); jrel++, yjrel+={dimy}) {{
                                    {red_formula.formula(fout,table)} // Call the function, which outputs results in fout
                                    {sum_scheme.accumulate_result(acc, fout, jrelloc, hack=True)}
                                }}
                            }}
                            __syncthreads();

                            // Step 4 : Save the result in global memory -----------------------------------------------------------
                            // The current thread has computed the "linewise-sum" of a small block of the full Kernel Product
                            // matrix, which corresponds to KP[ blockIdx.x * blockDim.x : (blockIdx.x+1) * blockDim.x ,
                            //                                  blockIdx.y * blockDim.x : (blockIdx.y+1) * blockDim.x ]
                            // We accumulate it in the output array out, which has in fact gridSize.y * nx
                            // lines of size DIMRED. The final reduction, which "sums over the block lines",
                            // shall be done in a later step.
                            if(i<nx) {{
                                {use_pragma_unroll()}
                                for(int k=0; k<{dimred}; k++) {{
                                    out[blockIdx.y*{dimred}*nx+i*{dimred}+k] = acc[k];
                                }}
                            }}
                        }}
                    """
