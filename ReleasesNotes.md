# PyFrac Releases Notes

Version 1.1.1 (November 2020)
-----

- Numerics:
    + New front reconstruction scheme such that the reconstructed fracture front is continuous between cells  
    + Non-linear solver now using Anderson acceleration   
    + Block Toeplitz elasticity matrix (for isotropic elasticity only)   
    + Improved tip asymptotics (notably for leak-off regimes)  
    + Improved closure/contact algorithm  
- Mesh & Re-meshing: 
    + possibility of having non-centered mesh (with respect to the orginin (0,0))   
    + elements addition on the fly (in specified directions)   
    + coarsening (reducing the number of cells in the whole domain)  
- Visualization:
    + Plots of near-tip asymptotic regimes (k-toughness-red, m-viscosity-blue, m-tilde-leak-off-green)  
    + Fluid fluxes & velocities as vectors at cell edges  
- Export to JSON capability  
- Use of the python logging package to log code outputs with different verbosity levels  
- Suite of automatic benchmark & regression tests
- Additional examples     
- Semi-analytical solution for M-pulse radial HF  
- Updated documentation

Version 1.0 (September 2019)
-----
- Initial version including:
    + Transverse Isotropic elasticity
    + Carter's leak-off
    + Newtonian fluid
    + Heterogeneous initial in-situ normal stress
    + Piece-wise heterogeneous fracture toughness and leak-off properties
    + Buoyancy 
- remeshing capabilities 