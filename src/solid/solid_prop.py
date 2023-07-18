# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Tue Dec 27 17:41:56 2016.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# external imports
import numpy as np
import math


class MaterialProperties:
    """
    Class defining the Material properties of the solid.

    Arguments:
        Mesh (CartesianMesh):           -- the CartesianMesh object describing the mesh.
        Eprime (float):                 -- plain strain modulus.
        Toughness (float):              -- Linear-Elastic Plane-Strain Fracture Toughness.
        Carters_coef (float):           -- Carter's leak off coefficient.
        confining_stress (ndarray):     -- in-situ confining stress field normal to fracture surface.
        grain_size (float):             -- the grain size of the rock; used to calculate the relative roughness.
        K1c_func (function):            -- the function giving the toughness on the domain. It takes one argument
                                           (angle) in case of anisotropic toughness and two arguments (x, y) in case
                                           of heterogeneous toughness. The function is also used to get the
                                           toughness if the domain is re-meshed.
        anisotropic_K1c (bool):         -- flag to specify if the fracture toughness is anisotropic.
        confining_stress_func (function):-- the function giving the in-situ stress on the domain. It should takes
                                           two arguments (x, y) to give the stress on these coordinates. It is also
                                           used to get the stress if the domain is re-meshed.
        Carters_coef_func (function):   -- the function giving the in Carter's leak off coefficient on the domain.
                                           It should takes two arguments (x, y) to give the coefficient on these
                                           coordinates. It is also used to get the leak off coefficient if the
                                           domain is re-meshed.
        TI_elasticity(bool):            -- if True, the medium is elastic transverse isotropic.
        Cij(ndarray):                   -- the transverse isotropic stiffness matrix (in the canonical basis); needs to
                                           be provided if TI_elasticity=True.
        free_surf (bool):               -- the free surface flag. True if the effect of free surface is to be taken
                                           into account.
        free_surf_depth (float):        -- the depth of the fracture from the free surface.
        TI_plane_angle (float):         -- the angle of the plane of the fracture with respect to the free surface.
        minimum_width (float):          -- minimum width corresponding to the asperity of the material.
        gravityValue (ndarray):         -- value of the specific gravity vector at every point
        gravityValueFunc (function):    -- the function giving the specific gravity vector on the domain. It should
                                           take two arguments (x, y) to give the coefficient on these coordinates. It is
                                           also used to get the leak off coefficient if the domain is re-meshed.


    Attributes:
        Eprime (float):             -- plain strain modulus.
        PoissonRatio (float):       -- Poisson's Ratio
        K1c (ndarray):              -- Linear-Elastic Plane-Strain Toughness for each cell.
        Kprime (ndarray):           -- 4*(2/pi)**0.5 * K1c.
        Cl (float):                 -- Carter's leak off coefficient.
        Cprime (ndarray):           -- 2 * Carter's leak off coefficient.
        SigmaO (ndarray):           -- in-situ confining stress field normal to fracture surface.
        grainSize (float):          -- the grain size of the rock; used to calculate the relative roughness.
        anisotropic_K1c (bool):     -- if True, the toughness is considered anisotropic.
        time_dep_toughness(ndarray):-- array to set if the toughness is fracture size or velocity dependent
        Kc1 (float):                -- the fracture toughness along the x-axis, in case it is anisotropic.
        TI_elasticity (bool):       -- the flag specifying transverse isotropic elasticity.
        Cij (ndarray):              -- the transverse isotropic stiffness matrix (in the canonical basis).
        freeSurf (bool):            -- if True, the effect of free surface is to be taken into account.
        FreeSurfDepth (float):      -- the depth of the fracture from the free surface.
        TI_PlaneAngle (float):      -- the angle of the plane of the fracture with respect to the free surface.
        K1cFunc (function):         -- the function giving the toughness on the domain. It takes one argument (angle) in
                                        case of anisotropic toughness and two arguments (x, y) in case of heterogeneous
                                        toughness. The function is also used to get the toughness if the domain is
                                        re-meshed.
        SigmaOFunc (function):      -- the function giving the in-situ stress on the domain. It should takes two
                                        arguments(x, y) to give the stress on these coordinates. It is also used to get
                                        the confining stress if the domain is re-meshed.
        ClFunc (function):          -- the function giving the in Carter's leak off coefficient on the domain. It should
                                        takes two arguments (x, y) to give the coefficient on these coordinates. It is
                                        also used to get the leak off coefficient if the domain is re-meshed.

    """

    def __init__(self, Mesh, Eprime, toughness=0., Carters_coef=0., Carters_t0=None, confining_stress=0.,
                 grain_size=0., K1c_func=None, anisotropic_K1c=False, time_dep_toughness=[False],
                 confining_stress_func=None, Carters_coef_func=None, TI_elasticity=False, Cij=None, free_surf=False,
                 free_surf_depth=1.e300, TI_plane_angle=0., minimum_width=1e-6, pore_pressure=-1.e100, density=2700,
                 density_func=None, gravity_value=[9.81, math.pi/2, -math.pi/2], gravity_value_func=None):
        """
        The constructor function
        """

        if isinstance(Eprime, np.ndarray):  # check if float or ndarray
            raise ValueError("Eprime can not be an array as input! - homogeneous medium only ")
        else:
            self.Eprime = Eprime

        if isinstance(toughness, np.ndarray):  # check if float or ndarray
            if toughness.size == Mesh.NumberOfElts:  # check if size equal to the mesh size
                self.K1c = toughness
                self.Kprime = (32 / math.pi) ** 0.5 * toughness
            else:
                # error
                raise ValueError('Error in the size of toughness input!')

        elif toughness is not None:
            self.K1c = toughness * np.ones((Mesh.NumberOfElts,), float)
            self.Kprime = (32 / math.pi) ** 0.5 * toughness * np.ones((Mesh.NumberOfElts,), float)

        if isinstance(Carters_coef, np.ndarray):  # check if float or ndarray
            if Carters_coef.size == Mesh.NumberOfElts:  # check if size equal to the mesh size
                self.Cl = Carters_coef
                self.Cprime = 2. * Carters_coef
            else:
                raise ValueError('Error in the size of Leak-Off coefficient input!')

        else:
            self.Cl = Carters_coef
            self.Cprime = 2. * Carters_coef * np.ones((Mesh.NumberOfElts,), float)

        #  initial time for the Carter law with which initialize the leak-off into the crack
        self.Carters_t0 = Carters_t0

        if isinstance(confining_stress, np.ndarray):  # check if float or ndarray
            if confining_stress.size == Mesh.NumberOfElts:  # check if size equal to the mesh size
                self.SigmaO = confining_stress
            else:
                raise ValueError('Error in the size of Sigma input!')
        else:
            self.SigmaO = confining_stress * np.ones((Mesh.NumberOfElts,), float)

        self.grainSize = grain_size
        self.anisotropic_K1c = anisotropic_K1c
        if anisotropic_K1c:
            try:
                self.Kc1 = K1c_func(0, 0, 0)
            except TypeError:
                raise SystemExit('The given Kprime function is not correct for anisotropic case! It should take three'
                                 ' arguments, i.e. the coordinates x and y and the angle of propagation (0 to 2Pi) and return a toughness value.')
        else:
            self.Kc1 = None

        if time_dep_toughness[0] == True:
            if time_dep_toughness[1] == "size":
                self.Kc1 = None
                self.velocityDependentToughness = [False]
                self.sizeDependentToughness = [True, time_dep_toughness[2], time_dep_toughness[3], time_dep_toughness[4]]
            elif time_dep_toughness[1] == "velocity":
                self.Kc1 = None
                self.velocityDependentToughness = [True, time_dep_toughness[2], time_dep_toughness[3],
                                                   time_dep_toughness[4], anisotropic_K1c[5]]
                self.sizeDependentToughness = [False]
            else:
                raise SystemExit('Type of time dependent toughness not implemented.')
        else:
            self.sizeDependentToughness = [False]
            self.velocityDependentToughness = [False]

        if K1c_func is not None: # and not self.anisotropic_K1c:
            # the function should return toughness by taking x and y coordinates and the local propagation direction given by the angle alpha
            self.inv_with_heter_K1c = True
            self.anisotropic_K1c = True
            try:
                K1c_func(0.,0.,0.)
            except TypeError:
                raise SystemExit('The  given Kprime function is not correct! It should take two arguments, '
                           'i.e. the x and y coordinates of a point and the angle of propagation (0 to 2Pi) and return a toughness value.')
        else:
            self.inv_with_heter_K1c = False

        self.TI_elasticity = TI_elasticity
        self.Cij = Cij
        if TI_elasticity or free_surf:
            if isinstance(Cij, np.ndarray):  # check if float or ndarray
                if Cij.shape == (6, 6):  # check if size is 6 x 6
                    self.Cij = Cij
                else:
                    raise ValueError('Cij matrix is not a 6x6 array!')
            else:
                raise ValueError('Cij matrix is not a numpy array!')

        self.freeSurf = free_surf
        if free_surf:
            if free_surf_depth == 1.e300:
                raise ValueError("Depth from free surface is to be provided.")
            elif Cij is None:
                raise ValueError("The stiffness matrix (in the canonical basis) is to be provided")
        self.FreeSurfDepth = free_surf_depth
        self.TI_PlaneAngle = TI_plane_angle

        if isinstance(density, np.ndarray):  # check if float or ndarray
            if density.size == Mesh.NumberOfElts:  # check if size equal to the mesh size
                self.density = density
            else:
                # error
                raise ValueError('Error in the size of toughness input!')
        else:
            self.density = density * np.ones((Mesh.NumberOfElts,), float)

        #  initiate the gravity vector
        if isinstance(gravity_value, np.ndarray):  # check if float or ndarray
            if gravity_value.size == Mesh.NumberOfElts:  # check if size equal to the mesh size
                self.gravityValue = gravity_value
            else:
                raise ValueError('Error in the size of gravity input!')
        else:
            gravityX = gravity_value[0] * np.sin(gravity_value[1]) * np.cos(gravity_value[2])
            if abs(gravityX) <= 1e-8:
                gravityX = 0.
            gravityY = gravity_value[0] * np.sin(gravity_value[1]) * np.sin(gravity_value[2])
            if abs(gravityY) <= 1e-8:
                gravityY = 0.
            self.gravityValue = gravityX * np.ones((2*Mesh.NumberOfElts,), float)
            self.gravityValue[1:-1:2] = gravityY

        self.K1cFunc = K1c_func
        self.SigmaOFunc = confining_stress_func
        self.ClFunc = Carters_coef_func
        self.DensityFunc = density_func
        self.gravityValueFunc = gravity_value_func

        # overriding with the values evaluated by the given functions
        if (K1c_func is not None) or (confining_stress_func is not None) or (Carters_coef_func is not None) \
                or (density_func is not None) or (gravity_value_func is not None):
            self.remesh(Mesh)

        self.wc = minimum_width
        self.porePressure = pore_pressure

    # ------------------------------------------------------------------------------------------------------------------

    def remesh(self, mesh):
        """
        This function evaluates the toughness, confining stress and leak off coefficient on the given mesh using the
        functions provided in the MaterialProperties object. It should be evaluated each time re-meshing is done.

        Arguments:
            mesh (CartesianMesh):        -- the CartesianMesh object describing the new mesh.

        """

        if self.K1cFunc is not None and not self.anisotropic_K1c:
            self.K1c = np.empty((mesh.NumberOfElts,), dtype=np.float64)
            for i in range(mesh.NumberOfElts):
                self.K1c[i] = self.K1cFunc(mesh.CenterCoor[i, 0], mesh.CenterCoor[i, 1],0.)
            self.Kprime = self.K1c * ((32 / math.pi) ** 0.5)
        elif self.K1cFunc is not None and self.anisotropic_K1c:
            self.K1c = np.empty((mesh.NumberOfElts,), dtype=np.float64)
            for i in range(mesh.NumberOfElts):
                self.K1c[i] = self.K1cFunc(mesh.CenterCoor[i, 0], mesh.CenterCoor[i, 1], np.pi/2)
            self.Kprime = self.K1c * ((32 / math.pi) ** 0.5)
        else:
            self.Kprime = np.full((mesh.NumberOfElts,), self.Kprime[0])

        if self.SigmaOFunc is not None:
            self.SigmaO = np.empty((mesh.NumberOfElts,), dtype=np.float64)
            for i in range(mesh.NumberOfElts):
                self.SigmaO[i] = self.SigmaOFunc(mesh.CenterCoor[i, 0], mesh.CenterCoor[i, 1])
        else:
            self.SigmaO = np.full((mesh.NumberOfElts,), self.SigmaO[0])

        if self.ClFunc is not None:
            self.Cl = np.empty((mesh.NumberOfElts,), dtype=np.float64)
            self.Cprime = np.empty((mesh.NumberOfElts,), dtype=np.float64)
            for i in range(mesh.NumberOfElts):
                self.Cl[i] = self.ClFunc(mesh.CenterCoor[i, 0], mesh.CenterCoor[i, 1])
            self.Cprime = 2 * self.Cl
        else:
            self.Cprime = np.full((mesh.NumberOfElts,), self.Cprime[0])

        if self.DensityFunc is not None:
            self.density = np.empty((mesh.NumberOfElts,), dtype=np.float64)
            for i in range(mesh.NumberOfElts):
                self.density[i] = self.DensityFunc(mesh.CenterCoor[i, 0], mesh.CenterCoor[i, 1])
        else:
            self.density = np.full((mesh.NumberOfElts,), self.density[0])

        if self.gravityValueFunc is not None:
            self.gravityValue = np.empty((mesh.NumberOfElts,), dtype=np.float64)
            for i in range(mesh.NumberOfElts):
                self.gravityValue[2*i:2*i+1] = self.gravityValueFunc(mesh.CenterCoor[i, 0], mesh.CenterCoor[i, 1])
        else:
            gravityValueInt = np.full((mesh.NumberOfElts,), self.gravityValue[0])
            gravityValueInt[1:-1:2] = self.gravityValue[1]
            self.gravityValue = gravityValueInt

    # ------------------------------------------------------------------------------------------------------------------

    def Kprime_func(self, x, y, alpha):
        return self.K1cFunc(x, y, alpha) * (32. / np.pi) ** 0.5
