import os
from pathlib import Path
from datetime import datetime
import subprocess

from setuptools import setup, find_packages

from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    IS_WINDOWS,
)


def append_nvcc_threads(nvcc_extra_args):
    nvcc_threads = os.getenv("NVCC_THREADS") or "32"
    return nvcc_extra_args + ["--threads", nvcc_threads]


# We use a specific version of cutlass for flash attn
#subprocess.run(["git", "submodule", "update", "--init", "csrc/cutlass"])

cc_flag = []
cc_flag.append("-gencode")
cc_flag.append("arch=compute_90a,code=sm_90a")

this_dir = os.path.dirname(os.path.abspath(__file__))

if IS_WINDOWS:
    cxx_args = ["/O2", "/std:c++17", "/DNDEBUG", "/W0"]
else:
    cxx_args = ["-O3", "-std=c++17", "-DNDEBUG", "-Wno-deprecated-declarations"]

class BuildExecutable(BuildExtension):
    def build_extension(self, ext):
        # Modify linker flags to create an executable
        if os.name == 'posix':  # Linux
            ext.extra_link_args = ['-o', 'cutlass_tutorial']
        else:  # Windows
            ext.extra_link_args = ['/OUT:cutlass_tutorial.exe']

        super().build_extension(ext)

ext_modules = []
ext_modules.append(
    CUDAExtension(
        name="cutlass_tutorial",
        sources=[
            "csrc/cutlass_api.cpp",
            "tutorials/00_basic_gemm.cu",
            "tutorials/01_cutlass_utilities.cu",

            "tutorials/cute/wgmma_sm90.cu",
            "tutorials/cute/wgmma_tma_sm90.cu",

            "csrc/flash_api.cpp",
            "tutorials/flash_attn/flash_fwd_hdim128_bf16_sm90.cu",
            "tutorials/flash_attn/flash_prepare_scheduler.cu",
        ],
        extra_compile_args={
            "cxx": cxx_args,
            "nvcc": append_nvcc_threads(
                [
                    "-O3",
                    "-std=c++17",
                    "-DNDEBUG",
                    "-D_USE_MATH_DEFINES",
                    "-Wno-deprecated-declarations",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "--ptxas-options=-v,--register-usage-level=10"
                ]
                + cc_flag
            ),
        },
        include_dirs=[
            Path(this_dir) / "csrc",
            Path(this_dir) / "csrc" / "cutlass" / "include",
            Path(this_dir) / "csrc" / "cutlass" / "tools" / "util" / "include",
            Path(this_dir) / "tutorials",
            Path(this_dir) / "tutorials" / "flash_attn",
        ],
    )
)


try:
    cmd = ['git', 'rev-parse', '--short', 'HEAD']
    rev = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
except Exception as _:
    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d-%H-%M-%S")
    rev = '+' + date_time_str


setup(
    name="cutlass_tutorial",
    version="1.0.0" + rev,
    packages=find_packages(include=['python']),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
