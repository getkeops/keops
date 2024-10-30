"""
Detect the platform and set the correct path for the keops library

Detect if venv is active and set the correct path for the keops library

Detect if conda is active and set the correct path for the keops library
"""

from Config_new import ConfigNew
import shutil

class DetectPlatform(ConfigNew):
    """
    A class to detect the operating system, virtual environment or conda environment,
    and CUDA detection by inheriting from ConfigNew.
    """

    def __init__(self):
        super().__init__()  # Initialize ConfigNew

    def print_all(self):
        print("\nPlatform Detection Summary")
        print("=" * 40)
        self.print_os()
        self.print_env_type()
        self.print_cuda_details()
        print("=" * 40)

    def print_cuda_details(self):
        self.print_use_cuda()
        if self.get_use_cuda():
            print(f"CUDA Version: {self.get_cuda_version()}")
            print(f"Number of GPUs: {self.n_gpus}")
        else:
            print("CUDA is not available.")
        if self._use_cuda:
            print(f"CUDA Version: {self.cuda_version}")
            print(f"Number of GPUs: {self.n_gpus}")
            print(f"GPU Compile Flags: {self.gpu_compile_flags}")
            # CUDA Include Path
            cuda_include_path = self.cuda_include_path
            print(f"CUDA Include Path: {cuda_include_path or 'Not Found'}")

            # Attempt to find CUDA compiler
            nvcc_path = shutil.which('nvcc')
            print(f"CUDA Compiler (nvcc): {nvcc_path or 'Not Found'}")
            if not nvcc_path:
                print(f"CUDA compiler 'nvcc' not found in PATH.")
        else:
            # CUDA is disabled; display the CUDA message
            print(f"{self.cuda_message}")

if __name__ == "__main__":
    # Create an instance of DetectPlatform
    detector = DetectPlatform()
    # Print all detection results
    detector.print_all()
