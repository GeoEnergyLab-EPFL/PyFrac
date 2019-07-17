.. PyFrac documentation master file, created by
   sphinx-quickstart on Mon Jun  4 15:58:10 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyFrac's documentation!
==================================

PyFrac is a 3D hydraulic fracture simulator for fractures propagating in a plane. Currently, the under development code has the following capabilities:

- Isotropic and transversely isotropic elastic infinite medium
- In-homogeneous and anisotropic fracture toughness
- In-homogeneousand leak-off properties
- In-homogeneous in-situ minimum stress
- Buoyant fluid
- Fracture closure and re-opening (multiple injection)
- Time dependent injection history
- Post processing and visualization routines

The code is largely based on Implicit Level Set Algorithm (see details `[here] <https://www.sciencedirect.com/science/article/pii/S0045782508000443>`_). The numerical scheme has the following features:

- Level set description of the fracture front atop a Cartesian mesh (rectangular elements).
- Multiscale resolution via the coupling of the semi-infinite hydraulic fracture tip solution with the finite discretization.
- Boundary element discretization for quasi-static elasticity.
- Finite volume discretization for lubrication flow.
- Fully coupled implicit hydro-mechanical solver.
- Eikonal equation solved via fast-marching method for fracture front evolution.
- Adaptive time-stepping.
- Implicit/explicit fracture front advancing.
- Re-meshing.

.. image:: /images/footprint_width_3.svg
    :align:   center

.. toctree::
   :maxdepth: 2
   :caption: Contents:


   GettingStarted
   RunningASimulation
   Visualization
   Examples


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
