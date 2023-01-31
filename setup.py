from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np


compile_args = ['-O3', '-march=native', '-ffast-math', '-fno-associative-math',
                '-std=c++14']
package = Extension('homs_tc.parser', ['homs_tc/parser.pyx'], 
                    language='c++', extra_compile_args=compile_args,
                    extra_link_args=compile_args, include_dirs=[np.get_include()])
setup(ext_modules=cythonize([package]))
