def binders_definitions(dtype, red_formula, varloader, tagHostDevice, tagCpuGpu, tag1D2D):
    indsI_str = str(varloader.indsi)[1:-1]
    indsJ_str = str(varloader.indsj)[1:-1]
    indsP_str = str(varloader.indsp)[1:-1]
    dimsX_str = str(varloader.dimsx)[1:-1]
    dimsY_str = str(varloader.dimsy)[1:-1]
    dimsP_str = str(varloader.dimsp)[1:-1]
    return f"""
                #include <string>
                #include <vector>
                #include <iostream>

                #define __TYPE__ {dtype}
                #define keops_tagIJ {red_formula.tagI}
                #define keops_pos_first_argI {varloader.pos_first_argI}
                #define keops_pos_first_argJ {varloader.pos_first_argJ}
                #define keops_dimout {red_formula.dim}
                #define keops_nminargs {varloader.nminargs}
                #define keops_nvarsI {varloader.nvarsi}
                #define keops_nvarsJ {varloader.nvarsj}
                #define keops_nvarsP {varloader.nvarsp}
                #define tagHostDevice {tagHostDevice}
                #define tagCpuGpu {tagCpuGpu}
                #define tag1D2D {tag1D2D}
                
                int keops_indsI[keops_nvarsI] = {{ {indsI_str} }};
                int keops_indsJ[keops_nvarsJ] = {{ {indsJ_str} }};
                int keops_indsP[keops_nvarsP] = {{ {indsP_str} }};
                int keops_dimsX[keops_nvarsI] = {{ {dimsX_str} }};
                int keops_dimsY[keops_nvarsJ] = {{ {dimsY_str} }};
                int keops_dimsP[keops_nvarsP] = {{ {dimsP_str} }};

                #define Error_msg_no_cuda "[KeOps] This KeOps shared object has been compiled without cuda support: - 1) to perform computations on CPU, simply set tagHostDevice to 0 - 2) to perform computations on GPU, please recompile the formula with a working version of cuda."

                #define index_t int*
                
                int get_ndim(index_t shape) {{
                	return shape[0];
                }}

                int get_size(index_t shape, int pos) {{
                	return shape[pos+1];
                }}
	
                void keops_error(std::string message) {{
                	throw std::runtime_error(message);
                }}

                #if C_CONTIGUOUS
                int get_size_batch(index_t shape, int nbatch, int b) {{
                  return get_size(shape, b);
                }};
                #else
                int get_size_batch(index_t shape, int nbatch, int b) {{
                  return get_size(shape, nbatch - b);
                }};
                #endif
                
                void check_nargs(int nargs, int nminargs) {{
                	if(nargs<nminargs) {{
                		keops_error("[KeOps] : not enough input arguments");
                	}}
                }}

            """