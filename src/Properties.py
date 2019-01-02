#
# This file is part of PyFrac.
#
# Created by Haseeb Zia on 03.04.17.
# Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2019.
#  All rights reserved. See the LICENSE.TXT file for more details.
#

import math
import numpy as np
import time
import datetime
from matplotlib.colors import to_rgb
from src.Labels import *


class MaterialProperties:
    """
    Class defining the Material properties of the solid

    Instance variables:
        Eprime (float)       -- plain strain modulus
        K1c (ndarray)        -- Linear-Elastic Plane-Strain Toughness for each cell
        Kprime (ndarray)     -- 4*(2/pi)**0.5 * K1c
        Cl (float)           -- Carter's leak off coefficient
        Cprime (ndarray)     -- 2 * Carter's leak off coefficient
        SigmaO (ndarray)     -- in-situ stress field
        grainSize (float)    -- the grain size of the rock; used to calculate the relative roughness.
        anisotropic (bool)   -- if True, the toughness is considered anisotropic.
        K1cFunc (function)   -- the function giving the toughness on the domain. It takes one argument (angle) in
                                case of anisotropic toughness and two arguments (x, y) in case of hetrogenous
                                toughness. The function is also used to get the toughness if the domain is remeshed.
        SigmaOFunc (function)-- the function giving the in-situ stress on the domain. It should takes two arguments
                                (x, y) to give the stress on these coordinates. It is also used to get the
                                stress if the domain is remeshed.
        ClFunc (function)    -- the function giving the in Carter's leak off coefficient on the domain. It should
                                takes two arguments (x, y) to give the coefficient on these coordinates. It is also
                                used to get the leak off coefficient if the domain is remeshed.
    Methods:
    """

    def __init__(self, Mesh, Eprime, Toughness=None, Cl=0., SigmaO=0., grain_size=0., K1c_func=None,
                 anisotropic_K1c=False, SigmaO_func = None, Cl_func = None, TI_elasticity=False, Cij = None,
                 free_surf=False, free_surf_depth=1.e300, TI_plane_angle=0., wc=1e-10):
        """
        Arguments:
            Eprime (float)          -- plain strain modulus.
            Toughness (float)       -- Linear-Elastic Plane-Strain Fracture Toughness.
            Cl (float)              -- Carter's leak off coefficient.
            SigmaO (ndarray)        -- in-situ stress field.
            grainSize (float)       -- the grain size of the rock; used to calculate the relative roughness.
            anisotropic (bool)      -- flag to specify if the fracture toughness is anisotropic.
            K1c_perp (float)        -- the fracture toughness in the direction of x-axis.
            anisotropic_flag(bool)  -- if True, the toughness is considered anisotropic.
            K1c_func (function)     -- the function giving the toughness on the domain. It takes one argument (angle) in
                                       case of anisotropic toughness and two arguments (x, y) in case of hetrogenous
                                       toughness. The function is also used to get the toughness if the domain is remeshed.
            SigmaO_func (function)  -- the function giving the in situ stress on the domain. It should takes two
                                       arguments (x, y) to give the stress on these coordinates. It is also used to get
                                       the stress if the domain is remeshed.
            Cl_func (function)      -- the function giving the in Carter's leak off coefficient on the domain. It
                                       should takes two arguments (x, y) to give the coefficient on these coordinates.
                                       It is also used to get the leak off coefficient if the domain is remeshed.
            TI_elasticity(bool)     -- if True, the medium is elastic transverse isotropic.
            Cij(Ndarray-float)      -- the TI stiffness matrix (in the canonical basis) is provided when
                                       TI_elasticity=true.
            wc (float)              -- minimum width corresponding to the asperity of the material.

        """

        if isinstance(Eprime, np.ndarray):  # check if float or ndarray
            raise ValueError("Eprime can not be an array as input! - homogeneous medium only ")
        else:
            self.Eprime = Eprime

        if isinstance(Toughness, np.ndarray):  # check if float or ndarray
            if Toughness.size == Mesh.NumberOfElts:  # check if size equal to the mesh size
                self.K1c = Toughness
                self.Kprime = (32 / math.pi) ** 0.5 * Toughness
            else:
                # error
                raise ValueError('Error in the size of Toughness input!')
                return
        elif Toughness is not None:
            self.K1c = Toughness * np.ones((Mesh.NumberOfElts,), float)
            self.Kprime = (32 / math.pi) ** 0.5 * Toughness * np.ones((Mesh.NumberOfElts,), float)

        if isinstance(Cl, np.ndarray):  # check if float or ndarray
            if Cl.size == Mesh.NumberOfElts:  # check if size equal to the mesh size
                self.Cl = Cl
                self.Cprime = 2. * Cl
            else:
                raise ValueError('Error in the size of Leak-Off coefficient input!')
                return
        else:
            self.Cl = Cl
            self.Cprime = 2. * Cl * np.ones((Mesh.NumberOfElts,), float)

        if isinstance(SigmaO, np.ndarray):  # check if float or ndarray
            if SigmaO.size == Mesh.NumberOfElts:  # check if size equal to the mesh size
                self.SigmaO = SigmaO
            else:
                raise ValueError('Error in the size of Sigma input!')
                return
        else:
            self.SigmaO = SigmaO * np.ones((Mesh.NumberOfElts,), float)

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
        self.SigmaOFunc = SigmaO_func
        self.ClFunc = Cl_func

        # overriding with the values evaluated by the given functions
        if (K1c_func is not None) or (SigmaO_func is not None) or (Cl_func is not None):
            self.remesh(Mesh)

        self.wc = wc

# ----------------------------------------------------------------------------------------------------------------------
    def remesh(self, mesh):
        """
        This function evaluate the toughness, confining stress and leak off coefficient according to the given mesh

        Arguments:
            mesh (CartesianMesh)        -- the CartesianMesh object describing the new mesh

        Returns:
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

        if self.SigmaOFunc is not None:
            self.SigmaO = np.empty((mesh.NumberOfElts,), dtype=np.float64)
            for i in range(mesh.NumberOfElts):
                self.SigmaO[i] = self.SigmaOFunc(mesh.CenterCoor[i, 0], mesh.CenterCoor[i, 1])

        if self.ClFunc is not None:
            self.Cl = np.empty((mesh.NumberOfElts,), dtype=np.float64)
            self.Cprime = np.empty((mesh.NumberOfElts,), dtype=np.float64)
            for i in range(mesh.NumberOfElts):
                self.Cl[i] = self.ClFunc(mesh.CenterCoor[i, 0], mesh.CenterCoor[i, 1])
            self.Cprime = 2 * self.Cl

#-----------------------------------------------------------------------------------------------------------------------


class FluidProperties:
    """
       Class defining the fluid properties

       Instance variables:
            viscosity (ndarray)     -- Viscosity of the fluid (note its different from local viscosity, see
                                       fracture class for local viscosity)
            rheology (string)       -- string specifying rheology of the fluid. Possible options:
                                         - "Newtonian"
                                         - "non-Newtonian"
            muPrime (float)         -- 12 * viscosity (// plates viscosity factor)
            density (float)         -- density of the fluid
            turbulence (bool)       -- turbulence flag. If true, turbulence will be taken into account
       Methods:
       """

    def __init__(self, viscosity=None, density=1000., rheology="Newtonian", turbulence=False):
        """
        Constructor function.

        Instance variables:
            viscosity (ndarray)     -- viscosity of the fluid (note its different from local viscosity, see
                                       fracture class for local viscosity)
            density (float)         -- density of the fluid
            rheology (string)       -- string specifying rheology of the fluid. Possible options:
                                         - "Newtonian"
                                         - "non-Newtonian"
            turbulence (bool)       -- turbulence flag. If true, turbulence will be taken into account
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

        rheologyOptions = ("Newtonian", "non-Newtonian")
        if rheology in rheologyOptions:  # check if rheology match to any rheology option
            if rheology is "Newtonian":
                self.rheology = rheology
            elif rheology is "non-Newtonian":
                raise ValueError("Non-Newtonian rheology not yet supported")
        else:# error
            raise ValueError('Invalid input for rheology. Possible options: ' + repr(rheologyOptions))

        self.density = density

        if isinstance(turbulence, bool):
            self.turbulence = turbulence
        else:
            # error
            raise ValueError('Invalid turbulence flag. Can be either True or False')


# --------------------------------------------------------------------------------------------------------
class InjectionProperties:
    """
        Class defining the injection schedule

        instance variables:
            injectionRate (ndarray)      -- array specifying the time series (row 0) and the corresponding injection
                                            rates (row 1).
            source_coordinates (ndarray) -- array with a single row and two columns specifying the x and y coordinate
                                            of the injection point coordinates.
            source_location (ndarray)    -- the element(s) where the fluid is injected in the cartesian mesh.
    """

    def __init__(self, rate, Mesh, source_coordinates=None ):
        """
        The constructor of the InjectionProperties class.

        Arguments:
            injectionRate (ndarray)      -- array specifying the time series (row 0) and the corresponding injection
                                            rates (row 1).
            Mesh (CartesianMesh)         -- the CartesianMesh object defining mesh.
            source_coordinates (ndarray) -- array with a single row and two columns specifying the x and y coordinate
                                            of the injection point coordinates.
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

        if not source_coordinates is None:
            raise ValueError("Variable source location not yet supported!")
            # if len(source_coordinates) == 2:
            #     self.source_coordinates = source_coordinates
            # else:
            #     # error
            #     raise ValueError('Invalid source coordinates. Correct format: a numpy array with a single row'
            #                      ' and two columns to \n specify x and y coordinate of the source e.g.'
            #                      ' np.array([x_coordinate, y_coordinate])')
        source_coordinates = [0., 0.]
        self.source_location = Mesh.locate_element(source_coordinates[0], source_coordinates[1])

# ----------------------------------------------------------------------------------------------------------------------

class LoadingProperties:
    """
        Class defining the mechanical loading properties

        Instance variables:
            EltLoaded (ndarray)   -- array of elements that are loaded.
            displ_rate (float):   -- the rate at which the elements in the EltLoaded list are displaced due to the
                                     applied mechanical loading
    """

    def __init__(self,  displ_rate=0., loaded_elts=None):
        """
        The constructor of the InjectionProperties class.

        Arguments:
            EltLoaded (ndarray)   -- array of elements that are loaded.
            displ_rate (float):   -- the rate at which the elements in the EltLoaded list are displaced due to the
                                     applied mechanical loading
        """

        self.displRate = displ_rate

        if isinstance(loaded_elts, np.ndarray):
            self.EltLoaded = loaded_elts
        else:
            raise ValueError("The loaded elements should be given in the form an ndarray of integers.")



# --------------------------------------------------------------------------------------------------------
class SimulationProperties:
    """
    Class defining the simulation properties.

        instance variables
            tolFractFront (float)       -- tolerance for the fracture front loop.
            toleranceEHL (float)        -- tolerance for the Elastohydrodynamic solver.
            toleranceProjection (float) -- tolerance for projection iteration for anisotropic case
            maxFrontItr (int)           -- maximum iterations to for the fracture front loop.
            maxSolverItr (int)          -- maximum iterations for the EHL iterative solver (Picard-Newton
                                           hybrid) in this case.
            maximumItrEHL (int)         -- maximum number of iterations for the Elastohydrodynamic solver.
            maxToughnessItr (int)       -- maximum toughness loop iterations.
            tmStpPrefactor (float)      -- factor for time-step adaptivity.
            maxTimeSteps (integer)      -- maximum number of time steps.
            tmStpPrefactor_max (float)  -- used in the case of re-attempt from five steps back.
            finalTime (float)           -- time where the simulation ends.
            timeStepLimit (float)       -- limit above which time step will not exceed.
            maxReattempts (int)         -- maximum number of reattempts in case of failure of a time step. A smaller
                                           time step will be attempted the given number of times.
            reAttemptFactor (float)     -- the factor by which time step is reduced on reattempts.
            plotFigure (boolean)        -- flag specifying to plot fracture trace after the given time period.
            saveToDisk (boolean)        -- flag specifying to save fracture to dist after the given time period.
            plotAnalytical (boolean)    -- if true, analytical solution will also be plotted along with the computed
                                           solution.
            analyticalSol (String)      -- the analytical solution of the radial fracture to be plotted on the
                                           fracture. Possible options:
                                                - K  (toughness dominated regime, without leak off)
                                                - Kt (toughness dominated regime , with leak off)
                                                - M  (viscosity dominated regime, without leak off)
                                                - Mt (viscosity dominated regime , with leak off)
                                                - E  (elliptical, toughness dominated without leak off)
            bckColor (String)           -- the string specifying the parameter according to which the background of the
                                           domain is color coded. Possible options
                                                - sigma0 (confining stress)
                                                - K1c (fracture toughness)
                                                - Cl (leak-off coefficient)
            plotEltType (boolean)       -- if True, the type of element will be spedified with color coded dots(channel,
                                           ribbon or tip).
            utputTimePeriod (float)    -- the time period after which the output file is written or the
                                           figures are plotted.
            tipAsymptote (string)       -- propagation regime. Possible options:
                                                - K  (toughness dominated regime, without leak off)
                                                - M  (viscosity dominated regime, without leak off)
                                                - Mt (viscosity dominated regime , with leak off)
                                                - U  (Universal regime accommodating viscosity, toughness
                                                     and leak off (see Donstov and Pierce, 2017))
                                                - MK (viscosity to toughness transition regime)
            saveRegime (boolean)        -- if True, the regime of the propagation (see Zia and Lecampion 2018) will be
                                           saved.
            verbosity (int)             -- the level of details about the ongoing simulation to be plotted (currently
                                           two levels 1 and 2 are supported).
            enableRemeshing (bool)      -- if True, the computational domain will be compressed by the factor given by
                                           by the variable remeshFactor after the fracture front reaches the end of the
                                           domain.
            remeshFactor (float)        -- the factor by which the domain is compressed on re-meshing.
            frontAdvancing (string)     -- The type of front advancing to be done. Possible options are:
                                                -- explicit
                                                -- semi-implicit
                                                -- implicit
            gravity (bool)              -- if True, the effect of gravity will be taken into account.
            collectPerfData (bool)      -- if True, the performance data will be collected in the form of a tree.
            paramFromTip (bool)     -- if True, the space dependant parameters such as toughness and leak-off
                                           coefficients will be taken from the tip by projections instead of taking them
                                           from the ribbon cell center. The numerical scheme as a result will become
                                           unstable due to the complexities in finding the projection
            saveReynNumb (boolean)      -- if True, the Reynold's number at each edge of the cells inside the fracture
                                           will be saved.
            saveFluidFlux (boolean)     -- if True, the fluid flux at each edge of the cells inside the fracture
                                           will be saved.
            saveFluidVel (boolean)     -- if True, the fluid velocity at each edge of the cells inside the fracture
                                           will be saved.
            TI_KernelExecPath (string)  -- the folder containing the executable to calculate transverse isotropic
                                           kernel or kernel with free surface.
            projMethod (string)         -- the method by which the angle prescribed by the projections onto the front
                                            are evaluated. Possible options are:
                                                -- 'ILSA_orig' (the method described in the original ILSA scheme)
                                                -- 'LS_grad' (using gradient of the level set)

        private variables:
            __out_file_address (string) -- disk address of the files to be saved. If not given, a new
                                           ./Data/"tim stamp" folder will be automatically created.
            __solTimeSeries (ndarray)   -- time series where the solution is required. The time stepping would be
                                           adjusted to get solution exactly at the given times.
            __dryCrack_mechLoading(bool)-- if True, the mechanical loading solver will be used.
            __viscousInjection (bool)   -- if True, the the solver will also take the fluid viscosity into account.
            __volumeControl (bool)      -- if True, the the volume control solver will be used.
            __simName (str)             -- the name of the simulation.
            __timeStamp (str)           -- the time at which the simulation properties was created.

            
    """

    def __init__(self, address = None):
        """
        The constructor of the SimulationParameters class. See documentation of the class.

        Arguments:
            address (str)               -- the folder where the simulation parameters file is located. The file must be
                                           name 'simul_param'
        """

        import sys
        if "win32" in sys.platform or "win64" in sys.platform:
            slash = "\\"
        else:
            slash = "/"
        if not '..' + slash + 'src' in sys.path:
            sys.path.append('..' + slash + 'src')
        if not '.' + slash + 'src' in sys.path:
            sys.path.append('.' + slash + 'src')

        if address is None:
            import src.default_SimParam as simul_param
        else:
            import sys
            sys.path.append(address)
            import simul_param
            sys.path.remove(address)


        # tolerances
        self.tolFractFront = simul_param.toleranceFractureFront
        self.toleranceEHL = simul_param.toleranceEHL
        self.toleranceProjection = simul_param.tol_projection

        # max iterations
        self.maxFrontItrs = simul_param.max_front_itrs
        self.maxSolverItrs = simul_param.max_solver_itrs
        self.maxToughnessItrs = simul_param.max_toughness_Itrs

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
        self.outputTimePeriod = simul_param.output_time_period
        self.plotFigure = simul_param.plot_figure
        self.plotAnalytical = simul_param.plot_analytical
        self.analyticalSol = simul_param.analytical_sol
        self.set_simulation_name(simul_param.sim_name)
        self.set_outputFolder(simul_param.output_folder)
        self.saveToDisk = simul_param.save_to_disk
        self.bckColor = simul_param.bck_color
        self.plotEltType = simul_param.plot_eltType
        self.blockFigure = simul_param.block_figure
        self.outputEveryTS = simul_param.output_every_TS
        self.plotVar = simul_param.plot_var
        self.plotProj = simul_param.plot_proj

        # solver type
        self.set_dryCrack_mechLoading(simul_param.mech_loading)
        self.set_viscousInjection(simul_param.viscous_injection)
        self.set_volumeControl(simul_param.volume_control)
        self.substitutePressure = simul_param.substitute_pressure

        # miscellaneous
        self.verbosity = simul_param.verbosity
        self.set_tipAsymptote(simul_param.tip_asymptote)
        self.saveRegime = simul_param.save_regime
        self.enableRemeshing = simul_param.enable_remeshing
        self.remeshFactor = simul_param.remesh_factor
        self.frontAdvancing = simul_param.front_advancing
        self.collectPerfData = simul_param.collect_perf_data
        self.paramFromTip = simul_param.param_from_tip
        self.saveReynNumb = simul_param.save_ReyNumb
        self.gravity = simul_param.gravity
        self.TI_KernelExecPath = simul_param.TI_Kernel_exec_path
        self.saveReynNumb = simul_param.save_ReyNumb
        self.saveFluidFlux = simul_param.save_fluid_flux
        self.saveFluidVel = simul_param.save_fluid_vel
        self.explicitProjection = simul_param.explict_projection
        self.symmetric = simul_param.symmetric
        self.projMethod = simul_param.proj_method
        if self.projMethod not in ['ILSA_orig', 'LS_grad']:
            raise ValueError("Projection method is not recognised!")

        # fracture geometry to calculate analytical solution for plotting
        self.height = simul_param.height
        self.aspectRatio = simul_param.aspect_ratio

# ----------------------------------------------------------------------------------------------------------------------

    # setter and getter functions

    def set_tipAsymptote(self, tip_asymptote):
        tipAssymptOptions = ("K", "M", "Mt", "U", "MK", "MDR", "M_MDR")
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

        import sys
        if "win32" in sys.platform or "win64" in sys.platform:
            slash = "\\"
        else:
            slash = "/"

        if output_address is not None:
            self.saveToDisk = True
            if "\\" in output_address:
                if slash != "\\":
                    raise SystemExit('Windows style slash in the given address on linux system.')
            elif "/" in output_address:
                if slash != "/":
                    raise SystemExit('linux style slash in the given address on windows system!')

            if output_address[-1] is slash:
                output_address = output_address[:-1]

            self.__outputAdress = output_address
            self.__outputFolder = output_address + slash + self.get_simulation_name() + slash
        else:
            self.__outputAdress = output_address
            self.__outputFolder = "." + slash + "_simulation_data_PyFrac" + slash + self.get_simulation_name() + slash


    def get_outputFolder(self):
        return self.__outputFolder

    def set_solTimeSeries(self, sol_t_srs):
        if isinstance(sol_t_srs, np.ndarray):
            self.__solTimeSeries = sol_t_srs
        elif sol_t_srs is None:
            self.__solTimeSeries = None
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
            self.set_outputFolder(self.__outputAdress)
        except AttributeError:
            pass


    def get_simulation_name(self):
        return self.__simName

    def get_timeStamp(self):
        return self.__timeStamp

#-----------------------------------------------------------------------------------------------------------------------


class IterationProperties:
    """
    This class stores performance data in the form of a tree
    """

    def __init__(self, itr_type="not initialized"):
        self.iterations = 0
        self.normList = []
        self.itrType = itr_type
        self.time = None
        self.CpuTime_start = time.time()
        self.CpuTime_end = None
        self.status = None
        self.failure_cause = None
        self.NumbOfElts = None
        self.subIterations = []

#-----------------------------------------------------------------------------------------------------------------------


class PlotProperties:
    """
    This class stores the parameters used for plotting of the post-processed results
    """

    def __init__(self, color_map=None, line_color=None, line_style='-', line_width=1., line_style_anal='--',
                 line_color_anal='r', interpolation='none', alpha=0.8, line_width_anal=None, text_size=None,
                 disp_precision=3, mesh_color='yellowgreen', mesh_edge_color='grey', mesh_label_color='black',
                 graph_scaling='linear', color_maps=None, colors_list=None, plot_legend=True, plot_FP_time=True):

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

    def __init__(self, variable, data_subset, projection='2D', use_latex=True):

        if variable not in supported_variables:
            raise ValueError(err_msg_variable)

        if data_subset in ('whole mesh', 'wm'):
            if projection in ('2D_clrmap', '2D_contours', '3D', '2D'):
                self.yLabel = 'meters'
                self.xLabel = 'meters'
                self.zLabel = labels[variable] + units[variable]
            elif projection is '1D':
                self.xLabel = 'time $(s)$'
                self.yLabel = labels[variable] + units[variable]
        elif data_subset in ('slice', 's'):
            if '2D' in projection:
                self.xLabel = 'meters'
                self.yLabel = labels[variable] + units[variable]
            elif projection is '3D':
                self.yLabel = 'meters'
                self.xLabel = 'meters'
                self.zLabel = labels[variable] + units[variable]
        elif data_subset in ('point', 'p'):
            self.xLabel = 'time $(s)$'
            self.yLabel = labels[variable] + units[variable]

        self.colorbarLabel = labels[variable] + units[variable]
        self.units = units[variable]
        self.unitConversion = unit_conversion[variable]
        self.figLabel = Fig_labels[variable]
        self.legend = labels[variable]
        self.useLatex = use_latex