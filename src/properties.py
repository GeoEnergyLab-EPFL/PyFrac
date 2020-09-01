# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on 03.04.17.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2019.
All rights reserved. See the LICENSE.TXT file for more details.
"""

import math
import numpy as np
import time
import datetime
from matplotlib.colors import to_rgb
from labels import var_labels, supported_variables, units, err_msg_variable, \
                    unit_conversion, Fig_labels, unidimensional_variables


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


    Attributes:
        Eprime (float):             -- plain strain modulus.
        K1c (ndarray):              -- Linear-Elastic Plane-Strain Toughness for each cell.
        Kprime (ndarray):           -- 4*(2/pi)**0.5 * K1c.
        Cl (float):                 -- Carter's leak off coefficient.
        Cprime (ndarray):           -- 2 * Carter's leak off coefficient.
        SigmaO (ndarray):           -- in-situ confining stress field normal to fracture surface.
        grainSize (float):          -- the grain size of the rock; used to calculate the relative roughness.
        anisotropic_K1c (bool):     -- if True, the toughness is considered anisotropic.
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

    def __init__(self, Mesh, Eprime, toughness=0., Carters_coef=0., confining_stress=0., grain_size=0., K1c_func=None,
                 anisotropic_K1c=False, confining_stress_func = None, Carters_coef_func = None, TI_elasticity=False,
                 Cij = None, free_surf=False, free_surf_depth=1.e300, TI_plane_angle=0., minimum_width=1e-6,
                 pore_pressure=-1.e100):
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
                self.Kc1 = K1c_func(0)
            except TypeError:
                raise SystemExit('The given Kprime function is not correct for anisotropic case! It should take one'
                                 ' argument, i.e. the angle and return a toughness value.')
        else:
            self.Kc1 = None

        if K1c_func is not None and not self.anisotropic_K1c:
            # the function should return toughness by taking x and y coordinates
            try:
                K1c_func(0.,0.)
            except TypeError:
                raise SystemExit('The  given Kprime function is not correct! It should take two arguments, '
                           'i.e. the x and y coordinates of a point and return the toughness at this point.')

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


        self.K1cFunc = K1c_func
        self.SigmaOFunc = confining_stress_func
        self.ClFunc = Carters_coef_func

        # overriding with the values evaluated by the given functions
        if (K1c_func is not None) or (confining_stress_func is not None) or (Carters_coef_func is not None):
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
                self.K1c[i] = self.K1cFunc(mesh.CenterCoor[i, 0], mesh.CenterCoor[i, 1])
            self.Kprime = self.K1c * ((32 / math.pi) ** 0.5)
        elif self.K1cFunc is not None and self.anisotropic_K1c:
            self.K1c = np.empty((mesh.NumberOfElts,), dtype=np.float64)
            for i in range(mesh.NumberOfElts):
                self.K1c[i] = self.K1cFunc(np.pi/2)
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

#-----------------------------------------------------------------------------------------------------------------------


class FluidProperties:
    """
    Class defining the fluid properties.

    Arguments:
        viscosity (ndarray):     -- viscosity of the fluid .
        density (float):         -- density of the fluid.
        rheology (string):       -- string specifying rheology of the fluid. Possible options:

                                     - "Newtonian"
                                     - "non-Newtonian"
        turbulence (bool):       -- turbulence flag. If true, turbulence will be taken into account
        compressibility (float): -- the compressibility of the fluid.

    Attributes:
        viscosity (ndarray):     -- Viscosity of the fluid (note its different from local viscosity, see
                                    fracture class for local viscosity).
        muPrime (float):         -- 12 * viscosity (parallel plates viscosity factor).
        rheology (string):       -- string specifying rheology of the fluid. Possible options:
            
                                     - "Newtonian"
                                     - "Herschel-Bulkley" or "HBF"
                                     - "power-law" or "PLF"
        density (float):         -- density of the fluid.
        turbulence (bool):       -- turbulence flag. If true, turbulence will be taken into account.
        compressibility (float): -- the compressibility of the fluid.
        n (float):               -- flow index of the Herschel-Bulkey fluid.
        k (float):               -- consistency index of the Herschel-Bulkey fluid.
        T0 (float):              -- yield stress of the Herschel-Bulkey fluid.
        Mprime                   -- 2**(n + 1) * (2 * n + 1)**n / n**n  * k

    """

    def __init__(self, viscosity=None, density=1000., rheology="Newtonian", turbulence=False, compressibility=0,
                 n=None, k=None, T0=None):
        """
        Constructor function.

        """
        if viscosity is None:
            # uniform viscosity
            self.viscosity = None
            self.muPrime = None
        elif isinstance(viscosity, np.ndarray):  # check if float or ndarray
            raise ValueError('Viscosity of the fluid can not be an array!. Note that'
                             ' its different from local viscosity')
        else:
            self.viscosity = viscosity
            self.muPrime = 12. * self.viscosity  # the geometric viscosity in the parallel plate solution

        rheologyOptions = ["Newtonian", "Herschel-Bulkley", "HBF", "power-law", "PLF"]
        if rheology in rheologyOptions:  # check if rheology match to any rheology option
            self.rheology = rheology
            if rheology in ["Herschel-Bulkley", "HBF"]:
                if n is None or k is None or T0 is None:
                    raise ValueError("n (flow index), k(consistency index) and T0 (yield stress) are required for a \
                                     Herscel-Bulkley type fluid!")
                self.n = n
                self.k = k
                self.T0 = T0
                self.Mprime = 2**(n + 1) * (2 * n + 1)**n / n**n  * k
                self.var1 = self.Mprime ** (-1 / n)
                self.var2 = 1/n - 1.
                self.var3 = 2. + 1/n
                self.var4 = 1. + 1/n
                self.var5 = n / (n + 1.)   
            elif rheology in ["power-law", "PLF"]:
                if n is None or k is None:
                    raise ValueError("n (flow index) and k(consistency index) are required for a power-law type fluid!")
                self.n = n
                self.k = k
                self.Mprime = 2**(n + 1) * (2 * n + 1)**n / n**n  * k
        else:# error
            raise ValueError('Invalid input for fluid rheology. Possible options: ' + repr(rheologyOptions))

        self.density = density

        if isinstance(turbulence, bool):
            self.turbulence = turbulence
        else:
            # error
            raise ValueError('Invalid turbulence flag. Can be either True or False')

        self.compressibility = compressibility

# ----------------------------------------------------------------------------------------------------------------------


class InjectionProperties:
    """
    Class defining the injection parameters.

    Arguments:
        rate (ndarray):               -- array specifying the time series (row 0) and the corresponding injection
                                         rates (row 1). The times are instant where the injection rate changes.

                                         Attention:
                                            The first time should be zero. The corresponding injection rate would
                                            be taken for initialization of the fracture with an analytical solution,
                                            if required.
        mesh (CartesianMesh):         -- the CartesianMesh object defining mesh.
        source_coordinates (ndarray): -- list or ndarray with a length of 2, specifying the x and y coordinates
                                         of the injection point. Not used if source_loc_func is provided (See below).
        source_loc_func (function):   -- the source location function is used to get the elements in which the fluid is
                                         injected. It should take the x and y coordinates and return True or False
                                         depending upon if the source is present on these coordinates. This function is
                                         evaluated at each of the cell centre coordinates to determine if the cell is
                                         a source element. It should have to arguments (x, y) and return True or False.
                                         It is also called upon re-meshing to get the source elements on the coarse
                                         mesh.
       sink_loc_func (function):      -- the sink location function is used to get the elements where there is a fixed rate
                                         sink. It should take the x and y coordinates and return True or False
                                         depending upon if the sink is present on these coordinates. This function is
                                         evaluated at each of the cell centre coordinates to determine if the cell is
                                         a sink element. It should have to arguments (x, y) and return True or False.
                                         It is also called upon re-meshing to get the source elements on the coarse
                                         mesh.
                                             
      sink_vel_func (function):       -- this function gives the sink velocity at the given (x, y) point.
                                   
    Attributes:
        injectionRate (ndarray):      -- array specifying the time series (row 0) and the corresponding injection
                                         rates (row 1). The time series provide the time when the injection rate
                                         changes.
        sourceCoordinates (ndarray):  -- array with a single row and two columns specifying the x and y coordinate
                                         of the injection point coordinates. If there are more than one source elements,
                                         the average is taken to get an estimate injection cell at the center.
        sourceElem (ndarray):         -- the element(s) where the fluid is injected in the cartesian mesh.
        sourceLocFunc (function):     -- the source location function is used to get the elements in which the fluid is
                                         injected. It should take the x and y coordinates and return True or False
                                         depending upon if the source is present on these coordinates. This function is
                                         evaluated at each of the cell centre coordinates to determine if the cell is
                                         a source element. It should have to arguments (x, y) and return True or False.
                                         It is also called upon re-meshing to get the source elements on the coarse
                                         mesh.
        sinkLocFunc (function):      --  see description of arguments.
        sink_vel_func (function):    --  see description of arguments.
                                         
    """

    def __init__(self, rate, mesh, source_coordinates=None, source_loc_func=None, sink_loc_func=None, sink_vel_func=None):
        """
        The constructor of the InjectionProperties class.
        """

        if isinstance(rate, np.ndarray):
            if rate.shape[0] != 2:
                raise ValueError('Invalid injection rate. The list should have 2 rows (to specify time and'
                                 ' corresponding injection rate) for each entry')
            elif rate[0, 0] != 0:
                raise ValueError("The injection rate should start from zero second i.e. rate[0, 0] should"
                                 " be zero.")
            else:
                self.injectionRate = rate
        else:
            self.injectionRate = np.asarray([[0], [rate]])

        if source_loc_func is None:
            if source_coordinates is not None:
                if len(source_coordinates) == 2:
                    print("Setting the source coordinates to the closest cell center...")
                    self.sourceCoordinates = source_coordinates
                else:
                    # error
                    raise ValueError('Invalid source coordinates. Correct format: a list or numpy array with a single'
                                     ' row and two columns to \n specify x and y coordinate of the source e.g.'
                                     ' np.array([x_coordinate, y_coordinate])')

            else:
                self.sourceCoordinates = [0., 0.]

            self.sourceElem = mesh.locate_element(self.sourceCoordinates[0], self.sourceCoordinates[1])
            if np.isnan(self.sourceElem).any():
                raise ValueError("The given source location is out of the mesh!")
            self.sourceCoordinates = mesh.CenterCoor[self.sourceElem]
            print("Injection point: " + '(x, y) = (' + repr(mesh.CenterCoor[self.sourceElem, 0][0]) +
                                        ',' + repr(mesh.CenterCoor[self.sourceElem, 1][0]) + ')')
            self.sourceLocFunc = None
        else:
            self.sourceLocFunc = source_loc_func
            self.sourceElem = []
            for i in range(mesh.NumberOfElts):
                if self.sourceLocFunc(mesh.CenterCoor[i, 0], mesh.CenterCoor[i, 1]):
                 self.sourceElem.append(i)

        if len(self.sourceElem) == 0:
            raise ValueError("No source element found!")
        self.sourceCoordinates = [np.mean(mesh.CenterCoor[self.sourceElem, 0]),
                                  np.mean(mesh.CenterCoor[self.sourceElem, 1])]
        
        self.sinkLocFunc = sink_loc_func
        self.sinkVelFunc = sink_vel_func
        if sink_loc_func is not None:
            if sink_vel_func is None:
                raise ValueError("Sink velocity function is required for sink elements!")
            
            self.sinkElem = []
            for i in range(mesh.NumberOfElts):
                if self.sinkLocFunc(mesh.CenterCoor[i, 0], mesh.CenterCoor[i, 1]):
                 self.sinkElem.append(i)
            
            self.sinkVel = np.empty(len(self.sinkElem))
            for i in range(len(self.sinkElem)):
                self.sinkVel[i] = sink_vel_func(mesh.CenterCoor[self.sinkElem[i], 0],
                                                mesh.CenterCoor[self.sinkElem[i], 1])
                

    #-------------------------------------------------------------------------------------------------------------------

    def get_injection_rate(self, tm, frac):
        """ This function gives the current injection rate at all of the cells in the domain.

        Arguments:
            tm (float):             -- the time at which the injection rate is required.
            frac (CartesianMesh):   -- the Fracture object containing the mesh and the current fracture elements.

        returns:
            Qin (ndarray):          -- an numpy array of the size of the mesh with injection rates in each of the cell

        """

        Qin = np.zeros(frac.mesh.NumberOfElts, float)
        indxCurTime = max(np.where(tm >= self.injectionRate[0, :])[0])
        currentRate = self.injectionRate[1, indxCurTime]  # current injection rate
        currentSource = np.intersect1d(self.sourceElem, frac.EltCrack)
        Qin[currentSource] = currentRate / len(currentSource)

        return Qin

    #-------------------------------------------------------------------------------------------------------------------


    def remesh(self, new_mesh, old_mesh):
        """ This function is called every time the domian is remeshed.

        Arguments:
            new_mesh (CartesianMesh):   -- the CartesianMesh object describing the new coarse mesh.
            old_mesh (CartesianMesh):   -- the CartesianMesh object describing the old mesh.
        """

        # update source elements according to the new mesh.
        if self.sourceLocFunc is None:
            actv_cells = np.asarray([1])
            for i in self.sourceElem:
                actv_cells = np.append(actv_cells, new_mesh.locate_element(old_mesh.CenterCoor[i, 0],
                                                                           old_mesh.CenterCoor[i, 1]))

            self.sourceElem = list(actv_cells[1::])
        else:
            self.sourceElem = []
            for i in range(new_mesh.NumberOfElts):
                if self.sourceLocFunc(new_mesh.CenterCoor[i, 0], new_mesh.CenterCoor[i, 1]):
                 self.sourceElem.append(i)

        
        if self.sinkLocFunc is not None:
            
            self.sinkElem = []
            for i in range(new_mesh.NumberOfElts):
                if self.sinkLocFunc(new_mesh.CenterCoor[i, 0], new_mesh.CenterCoor[i, 1]):
                 self.sinkElem.append(i)
            
            self.sinkVel = np.empty(len(self.sinkElem))
            for i in range(len(self.sinkElem)):
                    self.sinkVel[i] = self.sinkVelFunc(new_mesh.CenterCoor[self.sinkElem[i], 0],
                                                       new_mesh.CenterCoor[self.sinkElem[i], 1])


# ----------------------------------------------------------------------------------------------------------------------


class LoadingProperties:
    """
        Class defining the mechanical loading properties

        Attributes:
            EltLoaded (ndarray):  -- array of elements that are loaded.
            displ_rate (float):   -- the rate at which the elements in the EltLoaded list are displaced due to the
                                     applied mechanical loading
    """

    def __init__(self,  displ_rate=0., loaded_elts=None):
        """
        The constructor of the LoadingProperties class.

        Arguments:
            displ_rate (float):     -- the rate at which the elements in the EltLoaded list are displaced due to the
                                       applied mechanical loading.
            loaded_elts (ndarray):  -- array of elements that are loaded.
        """

        self.displRate = displ_rate

        if isinstance(loaded_elts, np.ndarray):
            self.EltLoaded = loaded_elts
        else:
            raise ValueError("The loaded elements should be given in the form an ndarray of integers.")

# ----------------------------------------------------------------------------------------------------------------------

class SimulationProperties:
    """
    Class defining the simulation properties.

    Arguments:
        address (str)                -- the folder where the simulation parameters file is located. The file must be
                                        named 'simul_param'. For the description of the arguments and there default
                                        values, see the py:module::default_SimParam .
    Attributes:
        tolFractFront (float):       -- tolerance for the fracture front loop.
        toleranceEHL (float):        -- tolerance for the Elastohydrodynamic solver.
        toleranceVStagnant (float):  -- tolerance on the velocity to decide if a cell is stagnant.
        toleranceProjection (float): -- tolerance for projection iteration for anisotropic case
        maxFrontItrs (int):          -- maximum iterations to for the fracture front loop.
        maxSolverItrs (int):         -- maximum iterations for the EHL iterative solver (Picard-Newton hybrid) in this
                                        case.
        maxProjItrs (int):           -- maximum iterations for the loop to find projection on the front from ribbon.
        tmStpPrefactor (float):      -- factor for time-step adaptivity.
        maxTimeSteps (integer):      -- maximum number of time steps.
        finalTime (float):           -- time where the simulation ends.
        timeStepLimit (float):       -- limit above which time step will not exceed.
        fixedTmStp (ndarray):        -- a float or an array giving the fixed time step. The array should have two rows,
                                        with the first row giving the time at which the time step would change and the
                                        second row giving the corresponding time step. If None is given as time step,
                                        appropriate time step would be calculated.
        maxReattempts (int):         -- maximum number of reattempts in case of failure of a time step. A smaller
                                        time step will be attempted the given number of times.
        reAttemptFactor (float):     -- the factor by which time step is reduced on reattempts.
        plotFigure (boolean):        -- flag specifying to plot fracture trace after the given time period.
        saveToDisk (boolean):        -- flag specifying to save fracture to dist after the given time period.
        plotAnalytical (boolean):    -- if true, analytical solution will also be plotted along with the computed
                                       solution.
        analyticalSol (String):      -- the analytical solution of the radial fracture to be plotted on the fracture. \
                                        Possible options:

                                            - K  (toughness dominated regime, without leak off)
                                            - Kt (toughness dominated regime , with leak off)
                                            - M  (viscosity dominated regime, without leak off)
                                            - Mt (viscosity dominated regime , with leak off)
                                            - E  (elliptical, toughness dominated without leak off)
        bckColor (String):           -- the string specifying the parameter according to which the background of the\
                                        domain is color coded. Possible options:

                                            - sigma0 or confining stress
                                            - K1c or fracture toughness
                                            - Cl or leak-off coefficient
        plotTimePeriod (float):      -- the time period after which the figures are plotted during simulation.
        blockFigure (bool):          -- if True, the plotted figure(s) will be blocked after every time they are
                                        plotted. The simulation will advance when any key will be pressed from keyboard.
        plotTSJump (int):            -- the number of time steps after which the variables given in plotVar attribute
                                        are plotted. E.g. a value of 4 will result in plotting every four time steps.
        plotVar (list):              -- a list of variable(s) to be plotted during simulation. The time / time steps
                                        after which the output is done can be controlled with a number of parameters (
                                        see above).
        saveTimePeriod (float):      -- the time period after which the results are saved to disk during simulation.
        saveTSJump (int):            -- the number of time steps after which the results are saved to disk, e.g. a value
                                        of 4 will result in plotting every four time steps.
        elastohydrSolver (string):   -- the type of solver to solve the elasto-hydrodynamic system. At the moment, two
                                        main solvers can be specified.

                                            - 'implicit_Picard'
                                            - 'implicit_Anderson'
                                            - 'RKL2'
        substitutePressure(bool):    -- a flag specifying the solver to be used. If True, the pressure will be
                                        substituted in the channel elements (see Zia and Lecampion, 2019).
        solveDeltaP (bool):          -- a flag specifying the solver to be used. If True, the change in pressure,
                                        instead of pressure will be solved in the tip cells and the cells where the
                                        width constraint is active (see Zia and Lecampion, 2019).
        solveStagnantTip (bool):     -- if True, the stagnant tip cells will also be solved for width. This may result
                                        in more stable pressure as the elasticity equation will also be solved in those
                                        cells.
        solveTipCorrRib (bool):      -- if True, the tip cells corresponding to the closed ribbon cells will also be
                                        considered as closed and the width will be imposed on them.
        solveSparse (bool):          -- if True, the fluid conductivity matrix will be made with sparse matrix.
        saveRegime (boolean):        -- if True, the regime of the propagation as observed in the ribbon cell (see Zia
                                        and Lecampion 2018, IJF) will be saved.
        verbosity (int):             -- the level of details about the ongoing simulation to be plotted (currently
                                        two levels 1 and 2 are supported).
        enableRemeshing (bool):      -- if True, the computational domain will be compressed by the factor given by
                                        by the variable remeshFactor after the fracture front reaches the end of the
                                        domain.
        remeshFactor (float):        -- the factor by which the domain is compressed on re-meshing.

        meshExtension (bool array):  -- an array of booleans defining if the mesh should be extended in the given
                                        direction or if it should get compressed. The distribution is bottom, top,
                                        left, right
        meshExtensionFactor (float): -- factor by which the current mesh is extended in the extension direction

        frontAdvancing (string):     -- The type of front advancing to be done. Possible options are:

                                            - 'explicit'
                                            - 'predictor-corrector'
                                            - 'implicit'
        gravity (bool):              -- if True, the effect of gravity will be taken into account.
        collectPerfData (bool):      -- if True, the performance data will be collected in the form of a tree.
        paramFromTip (bool):         -- if True, the space dependant parameters such as toughness and leak-off
                                        coefficients will be taken from the tip by projections instead of taking them
                                        from the ribbon cell center. The numerical scheme as a result will become
                                        unstable due to the complexities in finding the projection
        saveReynNumb (boolean):      -- if True, the Reynold's number at each edge of the cells inside the fracture
                                        will be saved.
        saveFluidFlux (boolean):     -- if True, the fluid flux at each edge of the cells inside the fracture
                                        will be saved.
        saveFluidVel (boolean):      -- if True, the fluid velocity at each edge of the cells inside the fracture
                                        will be saved.
        saveEffVisc (boolean)L       -- if True, the Newtonian equivalent viscosity of the non-Newtonian fluid will
                                        be saved.
        TI_KernelExecPath (string):  -- the folder containing the executable to calculate transverse isotropic
                                       kernel or kernel with free surface.
        explicitProjection (bool):   -- if True, direction from last time step will be used to evaluate TI parameters.
        symmetric (bool):            -- if True, the four quadrant of the domain will be considered symmetric and only
                                        one will be solved for. The rest will be replaced by its reflection along the x
                                        and y axes.

                                        Attention:
                                            The symmetric fracture is only implemented in the toughness dominated case.\
                                            Use full domain if viscous fluid is injected.
        enableGPU                    -- if True, the dense matrix vector product for the RKL scheme would be done using
                                        GPU. If False, multithreaded dot product implemented in the explicit_RKL module
                                        will be used to do it.
        nThreads                     -- The number of threads to be used for the dense matrix dot product in the RKL
                                        scheme. By default set to 4.
        projMethod (string):         -- the method by which the angle prescribed by the projections onto the front
                                        are evaluated. Possible options are:

                                            - 'ILSA_orig' (the method described in the original ILSA scheme).
                                            - 'LS_grad' (using gradient of the level set).
        height (float):             -- this parameters is only used in the case of height contained hydraulic fracture
                                       plots, e.g. to plot analytical solutions of PKN and KGD fractures.
        aspectRatio (float):        -- this parameters is only used in the case of elliptical hydraulic fracture
                                       plots, e.g. to plot analytical solutions of anisotropic toughness or TI
                                       elasticity.
        Attention:
            These attributes below are private:

        __outputAddress (string):     -- disk address of the files to be saved. If not given, a new\
                                                  ./Data/"tim stamp" folder will be automatically created.
        __solTimeSeries (ndarray):   -- time series where the solution is required. The time stepping would \
                                                   be adjusted to get solution exactly at the given times.
        __dryCrack_mechLoading(bool):-- if True, the mechanical loading solver will be used.
        __viscousInjection (bool):   -- if True, the the solver will also take the fluid viscosity into \
                                                  account.
        __volumeControl (bool):      -- if True, the the volume control solver will be used.
        __simName (string):          -- the name of the simulation.
        __timeStamp (string):        -- the time at which the simulation properties was created.
        __tipAsymptote (string):     -- the tip asymptote to be used. Can be modified with the provided function.
            
    """

    def __init__(self, address=None):
        """
        The constructor of the SimulationParameters class. See documentation of the class.

        """

        import sys
        if "win32" in sys.platform or "win64" in sys.platform:
            slash = "\\"
        else:
            slash = "/"

        if address is None:
            import default_parameters as simul_param
        else:
            import sys
            sys.path.append(address)
            import simul_param
            sys.path.remove(address)


        # tolerances
        self.tolFractFront = simul_param.toleranceFractureFront
        self.toleranceEHL = simul_param.toleranceEHL
        self.toleranceProjection = simul_param.tol_projection
        self.toleranceVStagnant = simul_param.toleranceVStagnant

        # max iterations
        self.maxFrontItrs = simul_param.max_front_itrs
        self.maxSolverItrs = simul_param.max_solver_itrs
        self.maxProjItrs = simul_param.max_proj_Itrs

        # time and time stepping
        self.maxTimeSteps = simul_param.maximum_steps
        self.tmStpPrefactor = simul_param.tmStp_prefactor
        self.finalTime = simul_param.final_time
        self.set_solTimeSeries(simul_param.req_sol_at)
        self.timeStepLimit = simul_param.timeStep_limit
        self.fixedTmStp = simul_param.fixed_time_step
        if isinstance(self.fixedTmStp, np.ndarray):
            if self.fixedTmStp.shape[0] != 2:
                raise ValueError('Invalid fixed time step. The list should have 2 rows (to specify time and '
                                 'corresponding time step) for each entry')

        # time step re-attempt
        self.maxReattempts = simul_param.max_reattemps
        self.reAttemptFactor = simul_param.reattempt_factor

        # output parameters
        self.plotFigure = simul_param.plot_figure
        self.plotAnalytical = simul_param.plot_analytical
        self.analyticalSol = simul_param.analytical_sol
        self.set_simulation_name(simul_param.sim_name)
        self.set_outputFolder(simul_param.output_folder)
        self.saveToDisk = simul_param.save_to_disk
        self.bckColor = simul_param.bck_color
        self.blockFigure = simul_param.block_figure
        self.plotTSJump = simul_param.plot_TS_jump
        self.plotTimePeriod = simul_param.plot_time_period
        self.plotVar = simul_param.plot_var
        self.saveTSJump = simul_param.save_TS_jump
        self.saveTimePeriod = simul_param.save_time_period

        # solver type
        self.elastohydrSolver = simul_param.elastohydr_solver
        self.Anderson_parameter = simul_param.m_Anderson
        self.relaxation_factor = simul_param.relaxation_param
        self.set_dryCrack_mechLoading(simul_param.mech_loading)
        self.set_viscousInjection(simul_param.viscous_injection)
        self.set_volumeControl(simul_param.volume_control)
        self.substitutePressure = simul_param.substitute_pressure
        self.solveDeltaP = simul_param.solve_deltaP
        self.solveStagnantTip = simul_param.solve_stagnant_tip
        self.solveTipCorrRib = simul_param.solve_tip_corr_rib
        self.solveSparse = simul_param.solve_sparse

        # miscellaneous
        self.verbosity = simul_param.verbosity
        self.set_tipAsymptote(simul_param.tip_asymptote)
        self.saveRegime = simul_param.save_regime
        self.enableRemeshing = simul_param.enable_remeshing
        self.remeshFactor = simul_param.remesh_factor

        self.meshExtension = simul_param.mesh_extension_direction
        self.meshExtensionFactor = simul_param.mesh_extension_factor

        self.frontAdvancing = simul_param.front_advancing
        self.collectPerfData = simul_param.collect_perf_data
        self.paramFromTip = simul_param.param_from_tip
        if simul_param.param_from_tip:
            raise ValueError("Parameters from tip not yet supported!")
        self.saveReynNumb = simul_param.save_ReyNumb
        self.gravity = simul_param.gravity
        self.TI_KernelExecPath = simul_param.TI_Kernel_exec_path
        self.saveReynNumb = simul_param.save_ReyNumb
        self.saveFluidFlux = simul_param.save_fluid_flux
        self.saveFluidVel = simul_param.save_fluid_vel
        self.saveFluidFluxAsVector = simul_param.save_fluid_flux_as_vector
        self.saveFluidVelAsVector = simul_param.save_fluid_vel_as_vector
        self.saveEffVisc = simul_param.save_effective_viscosity
        self.saveStatisticsPostCoalescence=simul_param.save_statistics_post_coalescence
        self.saveYieldRatio = simul_param.save_yield_ratio
        self.explicitProjection = simul_param.explicit_projection
        self.symmetric = simul_param.symmetric
        self.projMethod = simul_param.proj_method
        self.enableGPU = simul_param.enable_GPU
        self.nThreads = simul_param.n_threads
        if self.projMethod not in ['ILSA_orig', 'LS_grad', 'LS_continousfront']:
            raise ValueError("Projection method is not recognised!")

        # fracture geometry to calculate analytical solution for plotting
        self.height = simul_param.height
        self.aspectRatio = simul_param.aspect_ratio

        # parameter deciding to save the leak-off tip parameter
        self.saveChi = simul_param.save_chi

# ----------------------------------------------------------------------------------------------------------------------

    # setter and getter functions

    def set_tipAsymptote(self, tip_asymptote):
        """
        The function to set up the tip asymptote.

        Arguments:
            tip_asymptote (string):       -- propagation regime. possible options are:

                                            - K  (toughness dominated regime, without leak off)
                                            - M  (viscosity dominated regime, without leak off)
                                            - Mt (viscosity dominated regime , with leak off)
                                            - U  (Universal regime accommodating viscosity, toughness\
                                                 and leak off (see Donstov and Pierce, 2017), 0 order)
                                            - U1  (Universal regime accommodating viscosity, toughness\
                                                 and leak off (see Donstov and Pierce, 2017), delta correction)
                                            - MK (viscosity to toughness transition regime)
                                            - MDR (Maximum drag reduction asymptote, see Lecampion & Zia 2019)
                                            - M_MDR (Maximum drag reduction asymptote in viscosity sotrage \ 
                                                  regime, see Lecampion & Zia 2019)
                                            - HBF or HBF_aprox (Herschel-Bulkley fluid, see Bessmertnykh and \
                                                  Dontsov 2019; the tip volume is evaluated with a fast aproximation)
                                            - HBF_num_quad (Herschel-Bulkley fluid, see Bessmertnykh and \
                                                  Dontsov 2019; the tip volume is evaluated with numerical quadrature of the\ 
                                                  approximate function, which makes it very slow)
                                            - PLF or PLF_aprox (power law fluid, see Dontsov and \
                                                  Kresse 2017; the tip volume is evaluated with a fast aproximation)
                                            - PLF_num_quad (power law fluid, see Dontsov and \
                                                  Kresse 2017; the tip volume is evaluated with numerical quadrature of the\ 
                                                  approximate function, which makes it very slow)
                                            = PLF_M (power law fluid in viscosity storage regime; see Desroche et al.)
        """
        tipAssymptOptions = ["K", "M", "Mt", "U", "U1", "MK", "MDR", "M_MDR", "HBF", "HBF_aprox", 
                             "HBF_num_quad", "PLF", "PLF_aprox", "PLF_num_quad", "PLF_M"]
        if tip_asymptote in tipAssymptOptions:  # check if tip asymptote matches any option
            self.__tipAsymptote = tip_asymptote
        else: # error
            raise ValueError('Invalid tip asymptote. Possible options: ' + repr(tipAssymptOptions))

    def get_tipAsymptote(self):
        return self.__tipAsymptote

    def set_viscousInjection(self, visc_injection):
        self.__viscousInjection = visc_injection
        if visc_injection:
            self.__volumeControl = False
            self.__dryCrack_mechLoading = False

    def get_viscousInjection(self):
        return self.__viscousInjection

    def set_volumeControl(self, vol_control):
        self.__volumeControl = vol_control
        if vol_control:
            self.__viscousInjection = False
            self.__dryCrack_mechLoading = False

    def get_volumeControl(self):
        return self.__volumeControl

    def set_dryCrack_mechLoading(self, mech_loading):
        self.__dryCrack_mechLoading = mech_loading
        if mech_loading:
            self.__viscousInjection = False
            self.__volumeControl = False

    def get_dryCrack_mechLoading(self):
        return self.__dryCrack_mechLoading

    def set_outputFolder(self, output_address):
        # check operating system to get appropriate slash in the address

        slash = '/'
        if output_address is not None:
            self.saveToDisk = True

            if output_address[-1] == slash:
                output_address = output_address[:-1]

            self.__outputAddress = output_address
            self.__outputFolder = output_address + slash + self.get_simulation_name() + slash
        else:
            self.__outputAddress = output_address
            self.__outputFolder = "." + slash + "_simulation_data_PyFrac" + slash + self.get_simulation_name() + slash


    def get_outputFolder(self):
        return self.__outputFolder

    def set_solTimeSeries(self, sol_t_srs):
        if sol_t_srs is None:
            self.__solTimeSeries = None
        elif isinstance(sol_t_srs, np.ndarray):
            self.__solTimeSeries = sol_t_srs
        else:
            raise ValueError("The given solution time series is not a numpy array!")

    def get_solTimeSeries(self):
        return self.__solTimeSeries

    def set_simulation_name(self, simul_name):
        time_stmp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d__%H_%M_%S')
        if simul_name is None:
            self.__simName = 'simulation' + '__' + time_stmp
        else:
            if not isinstance(simul_name, str):
                raise ValueError("The given simulation name is not a string")
            else:
                self.__simName = simul_name + '__' + time_stmp

        self.__timeStamp = time.time()

        try:
            self.set_outputFolder(self.__outputAddress)
        except AttributeError:
            pass


    def get_simulation_name(self):
        return self.__simName

    def get_timeStamp(self):
        return self.__timeStamp

    def get_time_step_prefactor(self, t):
        if isinstance(self.tmStpPrefactor, np.ndarray):
            if self.tmStpPrefactor.shape[0] == 2:
                indxCurTime = max(np.where(t >= self.tmStpPrefactor[0, :])[0])
                return self.tmStpPrefactor[1, indxCurTime]  # current pre-factor
            else:
                raise ValueError("The time step pre-factor array should have two rows, where each column giving the"
                                 " time at which the pre-factor is changed in the first rwo, and the corresponding "
                                 "pre-factor in the second row.")
        else:
            return self.tmStpPrefactor

    def set_mesh_extension_direction(self, direction):
        """
        The function to set up in which directions the mesh should be extended

        Arguments:
            direction (string):       -- direction where the mesh should be extended:

                                            - top  (mesh extension towards positive y)
                                            - bottom  (mesh extension towards negative y)
                                            - Left (mesh extension towards negative x)
                                            - Right (mesh extension towards positive x)
                                            - vertical  (mesh extension up and down)
                                            - horizontal (mesh extension left and right)
        """

        if direction == 'vertical':
            self.meshExtension[0::1] = True
        elif direction == 'horizontal':
            self.meshExtension[2::-1] = True
        elif direction == 'top':
            self.meshExtension[1] = True
        elif direction == 'bottom':
            self.meshExtension[0] = True
        elif direction == 'left':
            self.meshExtension[2] = True
        elif direction == 'right':
            self.meshExtension[3] = True
        else: # error
            raise ValueError('Invalid mesh extension definition Possible options: top, bottom, left, right, vertical'
                             'or horizontal')

    def get_mesh_extension_direction(self):
        return self.meshExtension

    def set_mesh_extension_factor(self, c):
        self.meshExtensionFactor = c

    def get_mesh_extension_factor(self):
        return self.meshExtensionFactor


#-----------------------------------------------------------------------------------------------------------------------


class IterationProperties:
    """
    This class stores performance data in the form of a tree.

    Arguments:
        itr_type (string):      -- currently, the following iterations are profiled:
                                    - 'time step'
                                    - 'time step attempt'
                                    - 'same front'
                                    - 'extended front'
                                    - 'tip inversion'
                                    - 'tip width'
                                    - 'nonlinear system solve'
                                    - 'width constraint iteration'
                                    - 'linear system solve'
                                    - 'Brent method'
    """

    def __init__(self, itr_type="not initialized"):
        self.iterations = 0
        self.norm = None
        self.itrType = itr_type
        self.time = None
        self.CpuTime_start = time.time()
        self.CpuTime_end = None
        self.status = None
        self.failure_cause = None
        self.NumbOfElts = None

        # sub-iterations data
        if itr_type == 'time step':
            self.attempts_data = []
        elif itr_type == 'time step attempt':
            self.sameFront_data = []
            self.extendedFront_data = []
        elif itr_type == 'same front':
            self.nonLinSolve_data = []
        elif itr_type == 'extended front':
            self.tipInv_data = []
            self.tipWidth_data = []
            self.nonLinSolve_data = []
        elif itr_type == 'tip inversion':
            self.brentMethod_data = []
        elif itr_type == 'tip width':
            self.brentMethod_data = []
        elif itr_type == 'nonlinear system solve':
            self.widthConstraintItr_data = []
        elif itr_type == 'width constraint iteration':
            self.linearSolve_data = []
            self.RKL_data = []
        elif itr_type == 'linear system solve':
            pass
        elif itr_type == 'Brent method':
            pass
        else:
            raise ValueError("The given iteration type is not supported!")


#-----------------------------------------------------------------------------------------------------------------------

def instrument_start(itr_type, perfNode):
    if perfNode is not None:
        perfNode_return = IterationProperties(itr_type)
    else:
        perfNode_return = None

    return perfNode_return

def instrument_close(perfNode, perfNode_subItr, norm, n_elem, status, fail_cause, simulated_time):
    perfNode_subItr.CpuTime_end = time.time()
    perfNode_subItr.NumbOfElts = n_elem
    perfNode.iterations += 1
    perfNode_subItr.norm = norm
    perfNode_subItr.status = status
    perfNode_subItr.time = simulated_time
    if not status:
        perfNode_subItr.failure_cause = fail_cause

#-----------------------------------------------------------------------------------------------------------------------


class PlotProperties:
    """
    This class stores the parameters used for plotting of the post-processed results
    """

    def __init__(self, color_map=None, line_color=None, line_style='-', line_width=1., line_style_anal='--',
                 line_color_anal='r', interpolation='none', alpha=0.8, line_width_anal=None, text_size=None,
                 disp_precision=3, mesh_color='yellowgreen', mesh_edge_color='grey', mesh_label_color='black',
                 graph_scaling='linear', color_maps=None, colors_list=None, plot_legend=True, plot_FP_time=True,
                 use_tex=False):

        self.lineStyle = line_style
        self.lineWidth = line_width
        self.lineColor = line_color
        self.colorMap = color_map
        self.lineColorAnal = line_color_anal
        self.lineStyleAnal = line_style_anal
        self.lineWidthAnal = line_width_anal
        self.textSize = text_size
        self.dispPrecision = disp_precision
        self.meshColor = mesh_color
        self.meshEdgeColor = mesh_edge_color
        self.meshLabelColor = mesh_label_color
        self.interpolation = interpolation
        self.alpha = alpha
        self.graphScaling = graph_scaling
        self.plotLegend = plot_legend
        self.PlotFP_Time = plot_FP_time
        self.useTex = use_tex
        if color_maps is None:
            self.colorMaps = ['cool', 'Wistia', 'summer', 'autumn']
        else:
            self.colorMaps = color_maps
        if colors_list is None:
            self.colorsList = ['black', 'firebrick', 'olivedrab', 'royalblue', 'deeppink', 'darkmagenta']
        else:
            self.colorsList = colors_list
        if self.lineColor is None:
            self.lineColor = to_rgb(self.colorsList[0])
        else:
            self.colorsList = [line_color]
        if self.colorMap is None:
            self.colorMap = self.colorMaps[0]
        else:
            self.colorMaps = [color_map]

#-----------------------------------------------------------------------------------------------------------------------


class LabelProperties:
    """
    This class stores the labels of a plot figure.
    """

    def __init__(self, variable, data_subset='whole mesh', projection='2D', use_latex=True):

        if variable not in supported_variables:
            raise ValueError(err_msg_variable)

        if variable in unidimensional_variables:
            projection = '1D'

        if data_subset in ('whole mesh', 'wm'):
            if projection in ('2D_clrmap', '2D_contours', '2D_vectorfield', '3D', '2D'):
                self.yLabel = 'meters'
                self.xLabel = 'meters'
                self.zLabel = var_labels[variable] + units[variable]
            elif projection == '1D':
                self.xLabel = 'time $(s)$'
                self.yLabel = var_labels[variable] + units[variable]
        elif data_subset in ('slice', 's'):
            if '2D' in projection:
                self.xLabel = 'meters'
                self.yLabel = var_labels[variable] + units[variable]
            elif projection == '3D':
                self.yLabel = 'meters'
                self.xLabel = 'meters'
                self.zLabel = var_labels[variable] + units[variable]
        elif data_subset in ('point', 'p'):
            self.xLabel = 'time $(s)$'
            self.yLabel = var_labels[variable] + units[variable]

        self.colorbarLabel = var_labels[variable] + units[variable]
        self.units = units[variable]
        self.unitConversion = unit_conversion[variable]
        self.figLabel = Fig_labels[variable]
        self.legend = var_labels[variable]
        self.useLatex = use_latex
