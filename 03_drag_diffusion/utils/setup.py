from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules = cythonize(["ui_utils2.py"]))