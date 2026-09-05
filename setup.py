from setuptools import setup, find_packages

setup(
    name="pyfrac",
    version="1.2.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "numba",
        "dill",
        "requests",
    ],
    extras_require={
        "hmat": ["bigwham4py"],
    },
    python_requires=">=3.7",
)
