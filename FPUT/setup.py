from setuptools import setup
from Cython.Build import cythonize

setup(
    name="Simulation utils",
    ext_modules=cythonize("Sim_util.pyx", annotate=True),
    zip_safe=False,
)
