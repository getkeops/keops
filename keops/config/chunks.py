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
dim_treshold_chunk = 143
specdim_use_chunk1 = -1  # originally 80 but deactivated for release 1.4.2
specdim_use_chunk2 = 109
specdim_use_chunk3 = 112
specdim_use_chunk4 = 114

# special mode for formula of the type sum_j k(x_i,y_j)*b_j with high dimensional b_j
enable_final_chunk = True


def get_enable_finalchunk():
    global enable_finalchunk
    return enable_finalchunk


def set_enable_finalchunk(val):
    global enable_finalchunk
    if val == 1:
        enable_finalchunk = True
    elif val == 0:
        enable_finalchunk = False


dimfinalchunk = 64


def get_dimfinalchunk():
    global dimfinalchunk
    return dimfinalchunk


def set_dimfinalchunk(val):
    global dimfinalchunk
    dimfinalchunk = val


def use_final_chunks():
    global enable_final_chunk
    global mult_var_highdim
    return (enable_final_chunk and mult_var_highdim)


mult_var_highdim = False


def set_mult_var_highdim(val):
    global mult_var_highdim
    if val == 1:
        mult_var_highdim = True
    elif val == 0:
        mult_var_highdim = False
