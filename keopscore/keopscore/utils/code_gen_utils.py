import os
from hashlib import sha256

import keopscore
from keopscore.config.config import disable_pragma_unrolls

from keopscore.utils.misc_utils import KeOps_Error, KeOps_Message


def get_hash_name(*args):
    return sha256("".join(list(str(arg) for arg in args)).encode("utf-8")).hexdigest()[
        :10
    ]


from keopscore.utils.meta_toolbox import (
    c_zero_int,
    c_array_from_address,
    c_expression_from_string,
    c_empty_instruction,
    c_variable,
    c_value_dtype,
    c_pointer_dtype,
    c_for,
    c_block,
    c_instruction_from_string,
    disable_pragma_unrolls,
    use_pragma_unroll,
)


#######################################################################
# .  KeOps related helpers
#######################################################################


def GetDims(Vars):
    # returns the list of dim fields (dimensions) of a list of Var instances
    return tuple(v.dim for v in Vars)


def GetInds(Vars):
    # returns the list of ind fields (indices) of a list of Var instances
    return tuple(v.ind for v in Vars)


def GetCats(Vars):
    # returns the list of cat fields (categories) of a list of Var instances
    return tuple(v.cat for v in Vars)


class Var_loader:
    def __init__(self, red_formula, force_all_local):
        formula = red_formula.formula
        tagI, tagJ = red_formula.tagI, red_formula.tagJ

        mymin = lambda x: min(x) if len(x) > 0 else -1

        self.Varsi = formula.Vars(
            cat=tagI
        )  # list all "i"-indexed variables in the formula
        self.nvarsi = len(self.Varsi)  # number of "i"-indexed variables
        self.indsi = GetInds(self.Varsi)  # list indices of "i"-indexed variables
        self.pos_first_argI = mymin(self.indsi)  # first index of "i"-indexed variables
        self.dimsx = GetDims(self.Varsi)  # list dimensions of "i"-indexed variables
        self.dimx = sum(self.dimsx)  # total dimension of "i"-indexed variables

        self.Varsj = formula.Vars(
            cat=tagJ
        )  # list all "j"-indexed variables in the formula
        self.nvarsj = len(self.Varsj)  # number of "j"-indexed variables
        self.indsj = GetInds(self.Varsj)  # list indices of "j"-indexed variables
        self.pos_first_argJ = mymin(self.indsj)  # first index of "j"-indexed variables
        self.dimsy = GetDims(self.Varsj)  # list dimensions of "j"-indexed variables
        self.dimy = sum(self.dimsy)  # total dimension of "j"-indexed variables

        self.Varsp = formula.Vars(cat=2)  # list all parameter variables in the formula
        self.nvarsp = len(self.Varsp)  # number of parameter variables
        self.indsp = GetInds(self.Varsp)  # list indices of parameter variables
        self.pos_first_argP = mymin(self.indsp)  # first index of parameter variables
        self.dimsp = GetDims(self.Varsp)  # list indices of parameter variables
        self.dimp = sum(self.dimsp)  # total dimension of parameter variables

        self.inds = GetInds(formula.Vars_)
        self.nminargs = max(self.inds) + 1 if len(self.inds) > 0 else 0

        self.dims = GetDims(formula.Vars_)
        self.cats = GetCats(formula.Vars_)

        if force_all_local or len(formula.Vars_) == 0:
            self.is_local_var = [True] * self.nminargs
            self.dimx_local, self.dimy_local, self.dimp_local = (
                self.dimx,
                self.dimy,
                self.dimp,
            )
        else:
            self.is_local_var = [False] * self.nminargs
            dims_sorted, inds_sorted, cats_sorted = zip(
                *sorted(zip(self.dims, self.inds, self.cats))
            )
            dimcur = 0
            for k in range(len(dims_sorted)):
                dimcur += dims_sorted[k]
                if dimcur < keopscore.config.config.lim_dim_local_var:
                    self.is_local_var[inds_sorted[k]] = True
                else:
                    break

            cnt = [0, 0, 0]
            for k in range(len(dims_sorted)):
                if self.is_local_var[inds_sorted[k]] == True:
                    if cats_sorted[k] == tagI:
                        cnt[0] += dims_sorted[k]
                    elif cats_sorted[k] == tagJ:
                        cnt[1] += dims_sorted[k]
                    else:
                        cnt[2] += dims_sorted[k]
            self.dimx_local, self.dimy_local, self.dimp_local = cnt

    def table(
        self, xi, yj, pp, args, i, j, offsetsi=None, offsetsj=None, offsetsp=None
    ):
        return table(
            self.nminargs,
            self.dimsx,
            self.dimsy,
            self.dimsp,
            self.indsi,
            self.indsj,
            self.indsp,
            xi,
            yj,
            pp,
            self.is_local_var,
            args,
            i,
            j,
            offsetsi,
            offsetsj,
            offsetsp,
        )

    def direct_table(self, args, i, j):
        return direct_table(
            self.nminargs,
            self.dimsx,
            self.dimsy,
            self.dimsp,
            self.indsi,
            self.indsj,
            self.indsp,
            args,
            i,
            j,
        )

    def load_vars(self, cat, *args, **kwargs):
        if cat == "i":
            dims, inds = self.dimsx, self.indsi
        elif cat == "j":
            dims, inds = self.dimsy, self.indsj
        elif cat == "p":
            dims, inds = self.dimsp, self.indsp
        return load_vars(dims, inds, *args, **kwargs, is_local=self.is_local_var)


def table(
    nminargs,
    dimsx,
    dimsy,
    dimsp,
    indsi,
    indsj,
    indsp,
    xi,
    yj,
    pp,
    is_local,
    args,
    i,
    j,
    offsetsi=None,
    offsetsj=None,
    offsetsp=None,
):
    res = [None] * nminargs
    for dims, inds, loc, row_index, offsets in (
        (dimsx, indsi, xi, i, offsetsi),
        (dimsy, indsj, yj, j, offsetsj),
        (dimsp, indsp, pp, c_zero_int, offsetsp),
    ):
        k = 0
        for u in range(len(dims)):
            if is_local[inds[u]]:
                res[inds[u]] = c_array_from_address(dims[u], loc.c_address + k)
                k += dims[u]
            else:
                row_index_str = (
                    f"({row_index.id}+{offsets.id}[{u}])" if offsets else row_index.id
                )
                arg = args[inds[u]]
                expr_u = c_expression_from_string(
                    f"{arg.id}+{row_index_str}*{dims[u]}", c_pointer_dtype(arg.dtype)
                )
                res[inds[u]] = c_array_from_address(dims[u], expr_u)
    return res


def direct_table(nminargs, dimsx, dimsy, dimsp, indsi, indsj, indsp, args, i, j):
    res = [None] * nminargs
    for dims, inds, row_index in (
        (dimsx, indsi, i),
        (dimsy, indsj, j),
        (dimsp, indsp, c_zero_int),
    ):
        for u in range(len(dims)):
            arg = args[inds[u]]
            res[inds[u]] = c_array_from_address(dims[u], arg + row_index * dims[u])
    return res


def table4(
    nminargs,
    dimsx,
    dimsy,
    dimsp,
    dims_new,
    indsi,
    indsj,
    indsp,
    inds_new,
    xi,
    yj,
    pp,
    arg_new,
):
    res = [None] * nminargs
    for dims, inds, xloc in (
        (dimsx, indsi, xi),
        (dimsy, indsj, yj),
        (dimsp, indsp, pp),
        (dims_new, inds_new, arg_new),
    ):
        k = 0
        for u in range(len(dims)):
            res[inds[u]] = c_array_from_address(dims[u], xloc.c_address + k)
            k += dims[u]

    return res


def load_vars(
    dims,
    inds,
    xloc,
    args,
    row_index=c_zero_int,
    offsets=None,
    indsref=None,
    is_local=None,
):
    # returns a c++ code used to create a local copy of slices of the input tensors, for evaluating a formula
    # - dims is a list of integers giving dimensions of variables
    # - dims is a list of integers giving indices of variables
    # - xloc is a c_array, the local array which will receive the copy
    # - args is a list of c_variable, representing pointers to input tensors
    # - row_index is a c_variable (of dtype="int"), specifying which row of the matrix should be loaded
    # - offsets is an optional c_array (of dtype="int"), specifying variable-dependent offsets (used when broadcasting batch dimensions)
    # - indsref is an optional list of integers, giving index mapping for offsets
    # - is_local is an optional list of booleans, used in case not all variables must be loaded
    #
    # Example: assuming i=c_variable("int", "5"), xloc=c_variable("float", "xi") and px=c_variable("float**", "px"), then
    # if dims = [2,2,3] and inds = [7,9,8], the call to
    #   load_vars (dims, inds, xi, [arg0, arg1,..., arg9], row_index=i )
    # will output the following code:
    #   xi[0] = arg7[5*2+0];
    #   xi[1] = arg7[5*2+1];
    #   xi[2] = arg9[5*2+0];
    #   xi[3] = arg9[5*2+1];
    #   xi[4] = arg8[5*3+0];
    #   xi[5] = arg8[5*3+1];
    #   xi[6] = arg8[5*3+2];
    #
    # Example (with offsets): assuming i=c_variable("signed long int", "5"),
    # xloc=c_variable("float", "xi"), px=c_variable("float**", "px"),
    # and offsets = c_array("signed long int", 3, "offsets"), then
    # if dims = [2,2,3] and inds = [7,9,8], the call to
    #   load_vars (dims, inds, xi, [arg0, arg1,..., arg9], row_index=i, offsets=offsets)
    # will output the following code:
    #   xi[0] = arg7[(5+offsets[0])*2+0];
    #   xi[1] = arg7[(5+offsets[0])*2+1];
    #   xi[2] = arg9[(5+offsets[1])*2+0];
    #   xi[3] = arg9[(5+offsets[1])*2+1];
    #   xi[4] = arg8[(5+offsets[2])*3+0];
    #   xi[5] = arg8[(5+offsets[2])*3+1];
    #   xi[6] = arg8[(5+offsets[2])*3+2];
    #
    # Example (with offsets and indsref): assuming i=c_variable("signed long int", "5"),
    # xloc=c_variable("float", "xi"), px=c_variable("float**", "px"),
    # offsets = c_array("signed long int", 3, "offsets"),
    # if dims = [2,2,3] and inds = [7,9,8],
    # and indsref = [8,1,7,3,9,2], then since 7,8,9 are at positions 2,0,4 in indsref,
    # the call to
    #   load_vars (dims, inds, xi, [arg0, arg1,..., arg9], row_index=i, offsets=offsets, indsref=indsref)
    # will output the following code:
    #   xi[0] = arg7[(5+offsets[2])*2+0];
    #   xi[1] = arg7[(5+offsets[2])*2+1];
    #   xi[2] = arg9[(5+offsets[0])*2+0];
    #   xi[3] = arg9[(5+offsets[0])*2+1];
    #   xi[4] = arg8[(5+offsets[4])*3+0];
    #   xi[5] = arg8[(5+offsets[4])*3+1];
    #   xi[6] = arg8[(5+offsets[4])*3+2];
    if len(dims) > 0:
        a = c_variable("signed long int", "a")
        res = a.declare_assign(0)
        for u in range(len(dims)):
            l = indsref.index(inds[u]) if indsref else u
            row_index_l = row_index + offsets[l] if offsets else row_index
            if is_local[inds[u]]:
                v = c_variable("signed long int", "v")
                res += c_for(
                    v.declare_assign(0),
                    v < dims[u],
                    v.plus_plus,
                    xloc[a].assign(args[inds[u]][row_index_l * dims[u] + v])
                    + a.plus_plus,
                )
        return c_block(res)
    else:
        return c_empty_instruction


def load_vars_chunks(
    inds, dim_chunk, dim_chunk_load, dim_org, xloc, args, k, row_index=c_zero_int
):
    #
    # loads chunks of variables, unrolling dimensions and indices.
    #
    # Example:
    #   load_chunks([7,9,8], 3, 2, 11, xi, px, k, row_index=i)
    # with:
    # xi = c_variable("float", "xi"),
    # px = c_variable("float**", "px")
    # i = c_variable("signed long int","5"),
    # k = c_variable("signed long int","k")
    # means : there are 3 chunks of vectors to load. They are located
    # at positions 7, 9 and 8 in px. Now i=5 and dim_org=11, so we start
    # to load vectors at positions px[7]+5*11, px[9]+5*11, px[8]+5*11.
    # For each, we load the kth chunk, assuming vector is divided
    # into chunks of size 3. And finally, we stop after loading 2 values.
    #
    # So we will execute:
    #   xi[0] = px[7][5*11+k*3];
    #   xi[1] = px[7][5*11+k*3+1];
    #   xi[2] = px[9][5*11+k*3];
    #   xi[3] = px[9][5*11+k*3+1];
    #   xi[4] = px[8][5*11+k*3];
    #   xi[5] = px[8][5*11+k*3+1];
    string = ""
    if len(inds) > 0:
        string += "{"
        string += "signed long int a=0;\n"
        for u in range(len(inds)):
            string += use_pragma_unroll() + "\n"
            string += f"for(signed long int v=0; v<{dim_chunk_load}; v++) {{\n"
            string += f"    {xloc.id}[a] = {args[inds[u]].id}[{row_index.id}*{dim_org}+{k.id}*{dim_chunk}+v];\n"
            string += "     a++;\n"
            string += "}"
        string += "}"
    return c_instruction_from_string(string)


def load_vars_chunks_offsets(
    inds,
    indsref,
    dim_chunk,
    dim_chunk_load,
    dim_org,
    xloc,
    args,
    k,
    offsets,
    row_index=c_zero_int,
):
    # Version with variable-dependent offsets (used when broadcasting batch dimensions)
    # indsref gives mapping for offsets indexing
    # Example:
    #   load_vars_chunks_offsets([2,3,1], [8,9,7,3,1,2], 3, 2, 11, xi, px, k, offsets, row_index=i)
    # Since 2,3,1 are at positions 5,3,4 respectively in the list [8,9,7,3,1,2],
    # will output:
    #   xi[0] = px[2][(5+offsets[5])*11+k*3];
    #   xi[1] = px[2][(5+offsets[5])*11+k*3+1];
    #   xi[2] = px[3][(5+offsets[3])*11+k*3];
    #   xi[3] = px[3][(5+offsets[3])*11+k*3+1];
    #   xi[4] = px[1][(5+offsets[4])*11+k*3];
    #   xi[5] = px[1][(5+offsets[4])*11+k*3+1];
    string = ""
    if len(inds) > 0:
        string = "{"
        string += "signed long int a=0;\n"
        for u in range(len(inds)):
            l = indsref.index(inds[u])
            string += use_pragma_unroll() + "\n"
            string += f"for(signed long int v=0; v<{dim_chunk_load}; v++) {{\n"
            string += f"    {xloc.id}[a] = {args[inds[u]].id}[({row_index.id}+{offsets.id}[{l}])*{dim_org}+{k.id}*{dim_chunk}+v];\n"
            string += "     a++;\n"
            string += "}"
        string += "}"
    return c_instruction_from_string(string)


def varseq_to_array(vars, vars_ptr_name):
    # returns the C++ code corresponding to storing the values of a sequence of variables
    # into an array.

    dtype = vars[0].dtype
    nvars = len(vars)

    # we check that all variables have the same type
    if not all(var.dtype == dtype for var in vars[1:]):
        KeOps_Error("[KeOps] internal error ; incompatible dtypes in varseq_to_array.")
    string = f"""   {dtype} {vars_ptr_name}[{nvars}];
              """
    for i in range(nvars):
        string += f"""  {vars_ptr_name}[{i}] = {vars[i].id};
                   """
    return c_instruction_from_string(string)


def clean_keops(recompile_jit_binary=True, verbose=True):
    import keopscore.config.config
    from keopscore.config.config import get_build_folder

    build_path = get_build_folder()
    use_cuda = keopscore.config.config.use_cuda
    if use_cuda:
        from keopscore.config.config import jit_binary
    else:
        jit_binary = None
    for f in os.scandir(build_path):
        if recompile_jit_binary or f.path != jit_binary:
            os.remove(f.path)
    if verbose:
        KeOps_Message(f"{build_path} has been cleaned.")
    from keopscore.get_keops_dll import get_keops_dll

    get_keops_dll.reset()
    if use_cuda and recompile_jit_binary:
        from keopscore.binders.nvrtc.Gpu_link_compile import Gpu_link_compile

        Gpu_link_compile.compile_jit_compile_dll()
