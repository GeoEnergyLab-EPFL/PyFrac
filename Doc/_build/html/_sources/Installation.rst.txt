.. PyFrac documentation master file, created by
   sphinx-quickstart on Mon Jun  4 15:58:10 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Installation
============

Running PyFrac will require a functioning installation of Python 3 and SciPy. You can download ANACONDA distribution with python 3 and SciPy from `[here] <https://www.continuum.io/downloads>`_. The directory containing the PyFrac code is required to be added to the ``PYTHONPATH`` environment variable. It can be added with the following command on windows (give the local path in place of "path_of_PyFrac")::

    set PYTHONPATH=%PYTHONPATH%;path_of_PyFrac

and with the following for linux::

    export PYTHONPATH=${PYTHONPATH}:path_of_PyFrac

dill package
------------

PyFrac uses dill package for saving files on hard disk. You can use pip to install the latest distribution of the package with the following command::

    pip install dill


Running an example
-------------------

Change directory to the folder containing the PyFrac code. An example can be run from the windows command prompt or linux shell by executing the example script e.g::

    python ./examples/radial.py
