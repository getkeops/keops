import warnings

warnings.simplefilter("default")
warnings.warn(
    "[pyKeOps]: the kernel_product syntax is deprecated. Please consider using the LazyTensor helper instead.",
    DeprecationWarning,
)

from .kernels import Kernel, kernel_product, kernel_formulas
from .formula import Formula
