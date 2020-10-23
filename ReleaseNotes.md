# PyFrac Release Notes

Version 1.1 (October 2020)
-----

- Numerics:
    + New front reconstruction scheme such that the reconstructed fracture front is continuous between cells
    + Non-linear solver now using Anderson acceleration 
    + Block Toeplitz elasticity matrix (for isotropic elasticity only) 
    + Improved tip asymptotics (notably for leak-off regimes)
    + Improved closure/contact algorithm
- Mesh: 
    + possibility of having non-centered mesh (w.r. the orginin (0,0))
    + elements addition on the fly (in specified directions)
    + coarsening (reducing the number of cells in the whole domain)
- Visualization:
    + Plots of tip regimes
    + Fluid fluxes & velocities as vectors at cell edges
- Export to JSON capability
- Use of login to log with different verbosity levels
- Suite of automatic benchmark & regression tests    
- Semi-analytical solution for M-pulse radial HF
- Updated documentations 

Version 1.0 (September 2019)
-----
- Initial version including TI elasticity, remeshing 