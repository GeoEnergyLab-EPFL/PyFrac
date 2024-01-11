from setuptools import find_packages, setup

with open("./README.md", "r") as f:
    long_description = f.read()

setup(
    name="gelpyfrac",
    version="1.1.1",
    description="A simulator for the propagation of planar 3D fluid driven fractures",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GeoEnergyLab-EPFL/pyfrac",
    author="GeoEnergyLab-EPFL",
    author_email="GeoEnergyLab-EPFL@github.com",
    license="GPLv3",
    classifiers=[
        "License :: GNU Lesser General Public License v3",
        "Programming Language :: Python :: 3.10",
        "Operating System :: Linux & MacOS",
    ],
    install_requires=[
        "scipy",
        "dill",
        "matplotlib",
        "requests",
        "numba",
        "numpy",
    ],
    # extras_require={
    #     "dev": ["numpy", "dill"],
    # },
    python_requires=">=3.10",
)
