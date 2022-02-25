# special computation scheme for dim>100

enable_chunk = True


def get_enable_chunk():
    global enable_chunk
    return enable_chunk


def set_enable_chunk(val):
    global enable_chunk
    if val == 1:
        enable_chunk = True
    elif val == 0:
        enable_chunk = False


dimchunk = 64
dim_treshold_chunk = 146
specdims_use_chunk = [99, 100, 102, 120, 133, 138, 139, 140, 141, 142]

# special mode for formula of the type sum_j k(x_i,y_j)*b_j with high dimensional b_j
enable_final_chunk = True


def set_enable_finalchunk(val):
    global enable_final_chunk
    if val == 1:
        enable_final_chunk = True
    elif val == 0:
        enable_final_chunk = False


dimfinalchunk = 64


def get_dimfinalchunk():
    global dimfinalchunk
    return dimfinalchunk


def set_dimfinalchunk(val):
    global dimfinalchunk
    dimfinalchunk = val


def use_final_chunks(red_formula):
    global enable_final_chunk
    global mult_var_highdim
    global dim_treshold_chunk
    return (
        enable_final_chunk and mult_var_highdim and red_formula.dim > dim_treshold_chunk
    )


mult_var_highdim = False


def set_mult_var_highdim(val):
    global mult_var_highdim
    if val == 1:
        mult_var_highdim = True
    elif val == 0:
        mult_var_highdim = False
