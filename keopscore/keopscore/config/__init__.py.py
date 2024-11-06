import sys
import shutil
from pathlib import Path
import os

# Import the configuration classes
from base_config import ConfigNew
from Platform import DetectPlatform
from cuda import CUDAConfig
from openmp import OpenMPConfig

def main():
    # Define status indicators
    check_mark = '✅'
    cross_mark = '❌'

    # Create instances of the configuration classes
    platform_detector = DetectPlatform()
    cuda_config = CUDAConfig()
    openmp_config = OpenMPConfig()
    config = ConfigNew()  # Base configuration

    # Header
    print("\nKeOps Configuration and System Health Check")
    print("=" * 60)

    # General Information
    print(f"\nGeneral Information")
    print("-" * 60)
    platform_detector.print_os()
    platform_detector.print_python_version()
    platform_detector.print_env_type()

    # Python Executable Path
    python_path = Path(sys.executable)
    python_path_exists = python_path.exists()
    python_status = check_mark if python_path_exists else cross_mark
    print(f"Python Executable: {python_path} {python_status}")

    # Environment Path
    env_path = os.environ.get('PATH', '')
    print(f"System PATH Environment Variable:")
    print(env_path)

    # Compiler Configuration
    print(f"\nCompiler Configuration")
    print("-" * 60)
    compiler_path = shutil.which(config.cxx_compiler) if config.cxx_compiler else None
    compiler_available = compiler_path is not None
    compiler_status = check_mark if compiler_available else cross_mark
    config.print_cxx_compiler()
    print(f"C++ Compiler Path: {compiler_path or 'Not Found'} {compiler_status}")
    if not compiler_available:
        print(f"  {cross_mark} Compiler '{config.cxx_compiler}' not found on the system.")

    # OpenMP Support
    openmp_status = check_mark if openmp_config.get_use_OpenMP() else cross_mark
    print(f"\nOpenMP Support")
    print("-" * 60)
    openmp_config.print_use_OpenMP()
    if openmp_config.get_use_OpenMP():
        openmp_lib_path = openmp_config.openmp_lib_path or 'Not Found'
        print(f"OpenMP Library Path: {openmp_lib_path}")
    else:
        print(f"  {cross_mark} OpenMP support is disabled or not available.")

    # CUDA Support
    cuda_status = check_mark if cuda_config.get_use_cuda() else cross_mark
    print(f"\nCUDA Support")
    print("-" * 60)
    cuda_config.print_use_cuda()
    if cuda_config.get_use_cuda():
        print(f"CUDA Version: {cuda_config.cuda_version}")
        print(f"Number of GPUs: {cuda_config.n_gpus}")
        print(f"GPU Compile Flags: {cuda_config.gpu_compile_flags}")
        # CUDA Include Path
        cuda_include_path = cuda_config.get_cuda_include_path
        cuda_include_status = check_mark if cuda_include_path else cross_mark
        print(f"CUDA Include Path: {cuda_include_path or 'Not Found'} {cuda_include_status}")

        # Attempt to find CUDA compiler
        nvcc_path = shutil.which('nvcc')
        nvcc_status = check_mark if nvcc_path else cross_mark
        print(f"CUDA Compiler (nvcc): {nvcc_path or 'Not Found'} {nvcc_status}")
        if not nvcc_path:
            print(f"  {cross_mark} CUDA compiler 'nvcc' not found in PATH.")
    else:
        # CUDA is disabled; display the CUDA message
        print(f"  {cross_mark} {cuda_config.cuda_message}")

    # Conda or Virtual Environment Paths
    print(f"\nEnvironment Paths")
    print("-" * 60)
    env_type = platform_detector.get_env_type()
    if env_type.startswith("conda"):
        conda_env_path = Path(os.environ.get('CONDA_PREFIX', ''))
        conda_env_status = check_mark if conda_env_path.exists() else cross_mark
        print(f"Conda Environment Path: {conda_env_path} {conda_env_status}")
    elif env_type == "virtualenv":
        venv_path = Path(sys.prefix)
        venv_status = check_mark if venv_path.exists() else cross_mark
        print(f"Virtualenv Path: {venv_path} {venv_status}")
    else:
        print("Not using Conda or Virtualenv.")

    # Paths and Directories
    print(f"\nPaths and Directories")
    print("-" * 60)
    # Check if paths exist
    paths = [
        ('Base Directory Path', Path(config.base_dir_path)),
        ('Template Path', Path(config.template_path)),
        ('Bindings Source Directory', Path(config.bindings_source_dir)),
        ('KeOps Cache Folder', Path(config.keops_cache_folder)),
        ('Default Build Path', Path(config.default_build_path)),
    ]
    for name, path in paths:
        path_exists = path.exists()
        status = check_mark if path_exists else cross_mark
        print(f"{name}: {path} {status}")
        if not path_exists:
            print(f"Path '{path}' does not exist.")

    # JIT Binary
    config.print_jit_binary()
    jit_binary_path = Path(config.jit_binary)
    jit_binary_exists = jit_binary_path.exists()
    jit_binary_status = check_mark if jit_binary_exists else cross_mark
    print(f"JIT Binary Exists: {'Yes' if jit_binary_exists else 'No'} {jit_binary_status}")

    # Environment Variables
    print(f"\nEnvironment Variables")
    print("-" * 60)
    config.print_environment_variables()

    # Conclusion
    print("\nConfiguration Status Summary")
    print("=" * 60)
    # Determine overall status
    issues = []
    if not compiler_available:
        issues.append(f"C++ compiler '{config.cxx_compiler}' not found.{cross_mark}")
    if not openmp_config.get_use_OpenMP():
        issues.append(f"OpenMP support is disabled or not available.{cross_mark}")
    if cuda_config.get_use_cuda():
        nvcc_path = shutil.which('nvcc')
        if not nvcc_path:
            issues.append(f"CUDA compiler 'nvcc' not found.{cross_mark}")
        if not cuda_config.get_cuda_include_path:
            issues.append(f"CUDA include path not found.{cross_mark}")
    if not Path(config.keops_cache_folder).exists():
        issues.append(f"KeOps cache folder '{config.keops_cache_folder}' does not exist.")
    if issues:
        print(f"Some configurations are missing or disabled:{cross_mark}")
        for issue in issues:
            print(f"{issue}")
    else:
        print(f"{check_mark} All configurations are properly set up.")

if __name__ == "__main__":
    main()
