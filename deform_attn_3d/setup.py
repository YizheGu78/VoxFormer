from setuptools import setup
import os, glob

# RadarOcc compatibility note:
# The official RadarOcc stack uses PyTorch 1.10.1 + CUDA 11.3. This old
# CUDA toolchain cannot compile kernels directly for very new GPUs such as
# RTX 5060 Ti / sm_120. Build the VoxFormer op for sm_86 and keep PTX so that
# a newer NVIDIA driver may try JIT compilation on newer architectures.
# You can override this in the shell before building, e.g.:
#   export TORCH_CUDA_ARCH_LIST="8.6+PTX"
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6+PTX")

import torch
from torch.utils.cpp_extension import CUDAExtension, CppExtension, BuildExtension


def get_extensions():
    extensions = []
    ext_name = 'deform3dattn_custom_cn'

    # prevent ninja from using too many resources
    try:
        import psutil
        num_cpu = len(psutil.Process().cpu_affinity())
        cpu_use = max(4, num_cpu - 1)
    except (ModuleNotFoundError, AttributeError):
        cpu_use = 4
    os.environ.setdefault('MAX_JOBS', str(cpu_use))

    define_macros = []

    if torch.cuda.is_available():
        print(f'Compiling {ext_name} with CUDA')
        print(f'TORCH_CUDA_ARCH_LIST={os.environ.get("TORCH_CUDA_ARCH_LIST")}')
        define_macros += [('WITH_CUDA', None)]
        op_files = glob.glob('./csrc/*.cu')
        extension = CUDAExtension
    else:
        print(f'Compiling {ext_name} without CUDA')
        op_files = glob.glob('./csrc/*.cpp')
        extension = CppExtension

    include_path = os.path.abspath('./csrc')
    ext_ops = extension(
        name=ext_name,
        sources=op_files,
        include_dirs=[include_path],
        define_macros=define_macros)
    extensions.append(ext_ops)
    return extensions


setup(
    name='extension_example',
    ext_modules=get_extensions(),
    cmdclass={'build_ext': BuildExtension},
)
