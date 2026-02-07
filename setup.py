import os
import sys
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        cfg = 'Debug' if self.debug else 'Release'
        
        # We explicitly tell CMake where pybind11Config.cmake is located
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DCMAKE_BUILD_TYPE={cfg}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-Dpybind11_DIR={pybind11.get_cmake_dir()}'  # <--- NEW: The critical fix
        ]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # Config and Build
        print(f"Building with CMake args: {cmake_args}") # Debug print
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.', '--config', cfg], cwd=self.build_temp)

setup(
    name='sw_cuda_py',
    version='0.1.0',
    author='Federico Ferrari',
    description='CUDA accelerated Smith-Waterman alignment',
    long_description='',
    ext_modules=[CMakeExtension('sw_cuda_py')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    install_requires=['pybind11'],
)