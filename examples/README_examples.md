# PyFrac Examples

PyFrac is a fluid driven planar 3D fractures simulator written in Python and based on the implicit level set algorithm.
We provide a series of script examples to illustrate the capabilities of the code.
Users can then write similar scripts for their needs.

##Version

1.0

Examples provided
===============

Examples described in the Comp. Phys. Comm. (CPC) paper
--------------------------------

- radial_MtoKtilde_CPC_Ex1.py
    Example of a radial HF from the M-vertex (viscosity/storage) early time solution to the late time Ktilde (leak-off / toughness) solution

- height_contained_CPC_Ex2.py
    Example for the propagation of a HF confined between two layers - transition from a radial geometry to a PKN geometry

- dyke_spreading_CPC_Ex3.py
    Example of dyke (buoyant HF) propagation (zero toughness) - the dyke first moves upward and then spread laterally when reaching neutral buoyancy

- fracture_closure_CPC_Ex4.py
    Example of fracture propagation in a layered stress field, and modeling of closure upon the end of injection

Additional examples
---------------------------------

- buoyancy_line_source.py
    Example of buoyant hydraulic fracture driven by a horizontal line source injection

- ellipse_Anisotropic_toughness.py
    Propagation of an elliptical HF in a medium with an anisotropic toughness. See Zia et al., IJF (2018) for details.

- ellipse_TI_elasticity.py
    Propagation of an elliptical HF in a transversely isotropic elastic material - This example requires to have the elastic TI kernel C++ code compiled

- Experiment_Wu_et_al.py
    Modeling of the experiment described in Wu et al. (2008) - Zero toughness HF with non-symmetric stress contrast

- pre_existing_fracture.py
    Injection into a pre-existing fracture -   we see first the re-opening front (radial M-vertex type), then ballooning of the fracture until KI=KIc is reach and subsequent propagation. Because the initial fracture is not radial, we then see the fracture going back to a radial fracture (isotropic material)

- radial_K_symmetric.py
    Propagation of a radial HF in the toughness dominated regime

- radial_viscosity_explicit.py
    Propagation of a radial HF in the viscosity dominated regime - using explicit front advance - see Zia & Lecampion, IJNAMG (2019) for details.

- toughnnes_anisotropy_jump.py
    Propagation of a HF with anisotropic toughness according to the smoothed heaviside law described in Zia et al., IJF (2018)


**Note:**   Some of these examples may take upto 3 hours to run (see the file timing.txt in the examples for run time (in secs) on a mid-2015 MacBook Pro).
