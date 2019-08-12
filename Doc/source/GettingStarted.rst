.. PyFrac documentation master file, created by
   sphinx-quickstart on Mon Jun  4 15:58:10 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Getting started
===============

Running PyFrac will require a functioning installation of Python 3, numpy, SciPy and matplotlib. You can download ANACONDA distribution with all these packages from `[here] <https://www.anaconda.com/distribution/>`_. To run PyFrac using Unix shell or windows command prompt, the directory containing the PyFrac source code is required to be added to the ``PYTHONPATH`` environment variable. It can be added with the following command on windows (give the local path of the ``src`` folder in place of "path_of_PyFrac_src")::

    set PYTHONPATH=%PYTHONPATH%;"path_of_PyFrac_src"

and with the following for linux::

    export PYTHONPATH=${PYTHONPATH}:"path_of_PyFrac_src"

PyFrac uses dill package for saving files on hard disk. You can use pip to install the latest distribution of the package with the following command::

    pip install dill


Running an example
-------------------

Change directory to the folder containing the PyFrac code. An example can be run from the windows command prompt or linux shell by executing the example script e.g.::

    python ./examples/radial_viscosity_explicit.py

There are scripts available for a diverse set of examples in the examples folders provided with the source code.