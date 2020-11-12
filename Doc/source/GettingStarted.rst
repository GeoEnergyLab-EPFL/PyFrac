.. PyFrac documentation master file, created by
   sphinx-quickstart on Mon Jun  4 15:58:10 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Getting started
===============

Running PyFrac will require a functioning installation of Python 3, numpy, SciPy and matplotlib. You can download ANACONDA distribution with all these packages from `[here] <https://www.anaconda.com/distribution/>`_. To run PyFrac using Unix shell or windows command prompt, the directory containing the PyFrac source code is required to be added to the ``PYTHONPATH`` environment variable. It can be added with the following command on windows (give the local path of the ``src`` folder in place of path_of_PyFrac_src)::

    set PYTHONPATH=%PYTHONPATH%;path_of_PyFrac_src

and with the following for linux or mac::

    export PYTHONPATH=${PYTHONPATH}:path_of_PyFrac_src

PyFrac uses dill package for saving files on hard disk. You can use pip to install the latest distribution of the package with the following command::

    pip install dill

If you already have python 3 installed through anaconda, update installed packages to the latest version. You can use the following to update all installed packages::

   conda update --all

Transverse Isotropic Kernel
----------------------------
PyFrac uses a routine written in C++ to evaluate elasticity kernel for transversely isotropic materials. This C++ code has to be compiled before fracture simulation can be performed for transverse isotropic materials. Use the following steps to generate the executable:

.. note::

   The setup below is required only if you want to simulate fracture propagation in transversely isotropic materials.

The code uses the Inside Loop (il) library which requires installation of OpenBLAS. See https://github.com/InsideLoop/InsideLoop. We ship the il source codes with this release for simplicity.  Follow the instruction below for your operating system in order to compile the elastic TI code for planar fracture and rectangular mesh.

windows
^^^^^^^
   1. Download and install OpenBLAS. You can also download binary packages available for windows (preferred).
   2. Download and install MSYS2.
   3. Install gcc and cmake for MSYS2 using the following::

         pacman -S base-devel gcc vim cmake
   4. In case you have downloaded binary packages for OpenBLAS, you would have to provide the location of the OpenBLAS libraries. You can do that by providing the location in the CmakeLists file.
   5. Change directory to the TI_Kernel\ folder in PyFrac. Create the executable using cmake by running the following commands one by one::

         cmake .
         make

   6. Add MSYS2 libraries path (typically C:\\msys64\\usr\\bin) to the windows `PATH` environment variable.

Linux
^^^^^
   1. Install OpenBlas and LAPACK with the following commands::

         sudo apt-get install libopenblas-dev
         sudo apt-get install liblapacke-dev

   2. Install Cmake with the following::

         sudo apt-get -y install cmake

   3. Change directory to the TI_Kernel folder in PyFrac. Create the executable using cmake by running the following commands one by one::

         cmake .
         make

Mac
^^^^
   1. Install OpenBlas with the following::

         brew install openblas

   2. Install Cmake with the following::

         brew install cmake

   3. Change directory to the TI_Kernel folder in PyFrac. Create the executable using cmake by running the following commands one by one::

         cmake .
         make

Generating the documentation
============================
You can generate documentation locally using sphinx. First install shpinx using pip::

    pip install sphinx

Then change directory to the Doc folder present in the PyFrac code. Run the make command to build the documentation in html::

    make html

or in pdf as::

    make latexpdf

After the build is complete, you can access the documentation in the build folder. For html, start with the file named index. The pdf file is located in the subflolder latex.

Running an example
==================

Change directory to the folder containing the PyFrac code. An example can be run from the windows command prompt or linux shell by executing the example script e.g.::

    python ./examples/radial_viscosity_explicit.py

There are scripts available for a set of examples in the examples folders provided with the code, including the scripts to reproduce the results presented in the paper published in Computer Physics Communications (see it on arxiv). The corresponding example number from the paper is mentioned in the name of these scripts.

.. note::

   Some of the examples may take up to 3 hours to run (see the file timing.txt in the examples for run time (in secs) on a mid-2018 MacBook Pro). See also the Readme_examples.md in the examples folder for details.
