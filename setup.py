from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import os
import os.path

# TODO: un-hardcode this!
cuda_path       = '/usr/local/cuda'
cuda_include    = os.path.join(cuda_path, "include")
cuda_lib64      = os.path.join(cuda_path, "lib64")
cudnn_path      = '/usr/local/cudnn'
cudnn_include   = os.path.join(cudnn_path, "include")
cudnn_lib64     = os.path.join(cudnn_path, "lib64")

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

        The purpose of this class is to postpone importing pybind11
        until it is actually installed, so that the ``get_include()``
        method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

ext_modules = [
        Extension('pycudnn', ['src/PyCuDNN.cpp'],
            include_dirs=[
                './include',
                cuda_include,
                cudnn_include,
                get_pybind_include(),
                get_pybind_include(user=True),
                ],
            libraries = [
                'cudnn'
                ],
            library_dirs=[
                cuda_lib64,
                cudnn_lib64
                ],
            extra_compile_args=[
                '-std=c++11',
                '-std=c++1y'
                ],
            language='c++'
        ),
]

setup(
      name='pycudnn',
      version='0.0.3',
      author='Konstantyn Komarov',
      author_email='komarov.konstant@utexas.edu',
      description='Python wrapper around NVIDIA CuDNN library',
      ext_modules=ext_modules,
      install_requires=['pybind11>=1.7'],
      zip_safe=False,
      )
