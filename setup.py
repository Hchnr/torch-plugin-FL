import multiprocessing
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from distutils.command.clean import clean

from setuptools import Extension, find_packages, setup


# Env Variables
IS_DARWIN = platform.system() == "Darwin"
IS_WINDOWS = platform.system() == "Windows"

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

# Only run cmake build for actual build commands, not metadata collection
BUILD_COMMANDS = {"build", "build_ext", "install", "develop", "bdist_wheel", "bdist_egg", "editable_wheel"}
RUN_BUILD_DEPS = any(arg in BUILD_COMMANDS for arg in sys.argv)


def make_relative_rpath_args(path):
    if IS_DARWIN:
        return ["-Wl,-rpath,@loader_path/" + path]
    elif IS_WINDOWS:
        return []
    else:
        return ["-Wl,-rpath,$ORIGIN/" + path]


def get_pytorch_dir():
    import torch

    return os.path.dirname(os.path.realpath(torch.__file__))


def build_deps():
    build_dir = os.path.join(BASE_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)

    cmake_args = [
        "-DCMAKE_INSTALL_PREFIX="
        + os.path.realpath(os.path.join(BASE_DIR, "torch_flagos")),
        "-DPYTHON_INCLUDE_DIR=" + sysconfig.get_paths().get("include"),
        "-DPYTORCH_INSTALL_DIR=" + get_pytorch_dir(),
    ]

    # Add CUDA toolkit path if available
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        cmake_args.append(f"-DCMAKE_CUDA_COMPILER={cuda_home}/bin/nvcc")

    subprocess.check_call(
        ["cmake", BASE_DIR] + cmake_args, cwd=build_dir, env=os.environ
    )

    build_args = [
        "--build",
        ".",
        "--target",
        "install",
        "--config",  # For multi-config generators
        "Release",
        "--",
    ]

    if IS_WINDOWS:
        build_args += ["/m:" + str(multiprocessing.cpu_count())]
    else:
        build_args += ["-j", str(multiprocessing.cpu_count())]

    command = ["cmake"] + build_args
    subprocess.check_call(command, cwd=build_dir, env=os.environ)


class BuildClean(clean):
    def run(self):
        for i in ["build", "install", "torch_flagos/lib"]:
            dirs = os.path.join(BASE_DIR, i)
            if os.path.exists(dirs) and os.path.isdir(dirs):
                shutil.rmtree(dirs)

        for dirpath, _, filenames in os.walk(os.path.join(BASE_DIR, "torch_flagos")):
            for filename in filenames:
                if filename.endswith(".so"):
                    os.remove(os.path.join(dirpath, filename))


def main():
    if RUN_BUILD_DEPS:
        build_deps()

    if IS_WINDOWS:
        # /NODEFAULTLIB makes sure we only link to DLL runtime
        # and matches the flags set for protobuf and ONNX
        extra_link_args: list[str] = ["/NODEFAULTLIB:LIBCMT.LIB"] + [
            *make_relative_rpath_args("lib")
        ]
        # /MD links against DLL runtime
        # and matches the flags set for protobuf and ONNX
        # /EHsc is about standard C++ exception handling
        extra_compile_args: list[str] = ["/MD", "/FS", "/EHsc"]
    else:
        extra_link_args = [*make_relative_rpath_args("lib")]
        extra_compile_args = [
            "-Wall",
            "-Wextra",
            "-Wno-strict-overflow",
            "-Wno-unused-parameter",
            "-Wno-missing-field-initializers",
            "-Wno-unknown-pragmas",
            "-fno-strict-aliasing",
        ]

    ext_modules = [
        Extension(
            name="torch_flagos._C",
            sources=["torch_flagos/csrc/stub.c"],
            language="c",
            extra_compile_args=extra_compile_args,
            libraries=["torch_bindings"],
            library_dirs=[os.path.join(BASE_DIR, "torch_flagos/lib")],
            extra_link_args=extra_link_args,
        )
    ]

    package_data = {
        "torch_flagos": [
            "lib/*.so*",
            "lib/*.dylib*",
            "lib/*.dll",
            "lib/*.lib",
        ]
    }

    setup(
        name="torch_flagos",
        version="0.1.0",
        description="FlagGems operators as a custom PyTorch device (flagos)",
        author="FlagGems Team",
        packages=find_packages(),
        package_data=package_data,
        ext_modules=ext_modules,
        cmdclass={
            "clean": BuildClean,  # type: ignore[misc]
        },
        include_package_data=False,
        python_requires=">=3.8",
        install_requires=[
            "torch>=2.0",
        ],
    )


if __name__ == "__main__":
    main()
