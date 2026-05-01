import pykeops.config as pykeopsconfig


def get_gpu_number():
    return pykeopsconfig.cuda.get_n_gpus()
