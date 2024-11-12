# Import the configuration classes
from .config import ConfigNew
from .cuda import CUDAConfig
from .openmp import OpenMPConfig
from .Platform import DetectPlatform

# Instantiate the configurations
config = ConfigNew()
platform_detector = DetectPlatform()
cuda_config = CUDAConfig()
openmp_config = OpenMPConfig()

__all__ = ['config', 'platform_detector', 'cuda_config', 'openmp_config']