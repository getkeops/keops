# Import the configuration classes
from .base_config import Config
from .cuda import CUDAConfig
from .openmp import OpenMPConfig
from .Platform import DetectPlatform

# Instantiate the configurations
config = Config()
platform_detector = DetectPlatform()
cuda_config = CUDAConfig()
openmp_config = OpenMPConfig()

__all__ = [
    "config",
    "platform_detector",
    "cuda_config",
    "openmp_config",
    "get_config",
    "get_platform_config",
    "get_cuda_config",
    "get_openmp_config",
]

# Lazy initializers
_instances = {}


def get_instance(key, factory):
    if key not in _instances:
        _instances[key] = factory
    return _instances[key]


def get_config():
    return get_instance("config", config)


def get_cuda_config():
    return get_instance("cuda_config", cuda_config)


def get_openmp_config():
    return get_instance("openmp_config", openmp_config)


def get_platform_config():
    return get_instance("platform_detector", platform_config)
