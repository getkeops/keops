from keopscore.utils.gpu_utils import get_gpu_props


def get_gpu_number():
    return get_gpu_props()[0]
