#
# This file is part of PyFrac.
#
# Created by Brice Lecampion on 03.04.17.
# Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2017.  All rights reserved.
# See the LICENSE.TXT file for more details. 
#

import math
import numpy as np
import time


from src.CartesianMesh import *


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
                 anisotropic_flag=False, SigmaO_func = None, Cl_func = None):
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
        self.anisotropic = anisotropic_flag
        if anisotropic_flag:
            try:
                self.K1c_perp = K1c_func(0)
            except TypeError:
                raise SystemExit('The given Kprime function is not correct for anisotropic case! It should take one'
                                 ' argument, i.e. the angle and return a toughness value.')
        else:
            self.K1c_perp = None

        if K1c_func is not None and not self.anisotropic:
            try:
                K1c_func(0.,0.)
            except TypeError:
                raise SystemExit('The  given Kprime function is not correct! It should take two arguments, '
                           'i.e. the x and y coordinates of a point and return the toughness at this point.')

        self.K1cFunc = K1c_func
        self.SigmaOFunc = SigmaO_func
        self.ClFunc = Cl_func

        # overriding with the values evaluated by the given functions
        if (K1c_func is not None) or (SigmaO_func is not None) or (Cl_func is not None):
            self.remesh(Mesh)


# ----------------------------------------------------------------------------------------------------------------------
    def remesh(self, mesh):
        """
        This function evaluate the toughness, confining stress and leak off coefficient according to the given mesh

        Arguments:
            mesh (CartesianMesh)        -- the CartesianMesh object describing the new mesh

        Returns:
        """

        if self.K1cFunc is not None and not self.anisotropic:
            self.Kprime = np.empty((mesh.NumberOfElts, ), dtype=np.float64)
            self.K1c = np.empty((mesh.NumberOfElts,), dtype=np.float64)
            for i in range(mesh.NumberOfElts):
                self.K1c[i] = self.K1cFunc(mesh.CenterCoor[i, 0], mesh.CenterCoor[i, 1])
            self.Kprime = self.K1c * ((32 / math.pi) ** 0.5)
        elif self.K1cFunc is not None and self.anisotropic:
            self.Kprime = np.empty((mesh.NumberOfElts,), dtype=np.float64)
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
                raise ValueError('Invalid injection rate. The list should have 2 rows (to specify time and corresponding '
                      'injection rate) for each entry')
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
class SimulationParameters:
    """
        Class defining the simulation parameters

        instance variables
            tolFractFront (float)       -- tolerance for the fracture front loop.
            toleranceEHL (float)        -- tolerance for the Elastohydrodynamic solver.
            toleranceToughness (float)  -- tolerance for toughness iteration
            maxFrontItr (int)           -- maximum iterations to for the fracture front loop.
            maxSolverItr (int)          -- maximum iterations for the EHL iterative solver (Picard-Newton
                                           hybrid) in this case.
            maximumItrEHL (int)         -- maximum number of iterations for the Elastohydrodynamic solver.
            maxToughnessItr (int)       -- maximum toughness loop iterations.
            tmStpPrefactor (float)      -- factor for time-step adaptivity.
            maxTimeSteps (integer)      -- maximum number of time steps.
            tmStpPrefactor_max (float)  -- used in the case of re-attempt from five steps back.
            FinalTime (float)           -- time where the simulation ends.
            timeStepLimit (float)       -- limit above which time step will not exceed.
            tmStpFactLimit (float)      -- limit on to what factor the time step can increase between two successive
                                           time steps.
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
                                                - Kprime (fracture toughness)
                                                - Cprime (leak-off coefficient)
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
            remeshFactor (float)        -- the factor by which the domain is compressed on re-meshing.
            frontAdvancing (string)     -- The type of front advancing to be done. Possible options are:
                                                -- explicit
                                                -- semi-implicit
                                                -- implicit
            collectPerfData (bool)      -- if True, the performance data will be collected in the form of a tree.

        private variables:
            __out_file_address (string) -- disk address of the files to be saved. If not given, a new
                                           ./Data/"tim stamp" folder will be automatically created.
            __solTimeSeries (ndarray)   -- time series where the solution is required. The time stepping would be
                                           adjusted to get solution exactly at the given times.
            __dryCrack_mechLoading(bool)-- if True, the mechanical loading solver will be used.
            __viscousInjection (bool)   -- if True, the the solver will also take the fluid viscosity into account.
            __volumeControl (bool)      -- if True, the the volume control solver will be used.

            
    """

    def __init__(self, address = None):
        """
        The constructor of the SimulationParameters class. See documentation of the class.

        Arguments:

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
            import default_SimParam as simul_param
        else:
            import sys
            sys.path.append(address)
            import simul_param
            sys.path.remove(address)


        # tolerances
        self.tolFractFront = simul_param.toleranceFractureFront
        self.toleranceEHL = simul_param.toleranceEHL
        self.toleranceToughness = simul_param.tol_toughness

        # max iterations
        self.maxFrontItr = simul_param.maxfront_its
        self.maxSolverItr = simul_param.max_itr_solver
        self.maxToughnessItr = simul_param.max_toughnessItr

        # time and time stepping
        self.maxTimeSteps = simul_param.maximum_steps
        self.tmStpPrefactor = simul_param.tmStp_prefactor
        self.tmStpPrefactor_max = simul_param.tmStp_prefactor
        self.FinalTime = simul_param.final_time
        self.set_solTimeSeries(simul_param.req_sol_at)
        self.timeStepLimit = simul_param.timeStep_limit
        self.tmStpFactLimit = simul_param.tmStp_fact_limit

        # time step re-attempt
        self.maxReattempts = simul_param.max_reattemps
        self.reAttemptFactor = simul_param.reattempt_factor

        # output parameters
        self.outputTimePeriod = simul_param.output_time_period
        self.plotFigure = simul_param.plot_figure
        self.plotAnalytical = simul_param.plot_analytical
        self.analyticalSol = simul_param.analytical_sol
        self.set_outFileAddress(simul_param.out_file_folder)
        self.saveToDisk = simul_param.save_to_disk
        self.bckColor = simul_param.bck_color
        self.plotEltType = simul_param.plot_eltType

        # solver type
        self.set_dryCrack_mechLoading(simul_param.mech_loading)
        self.set_viscousInjection(simul_param.viscous_injection)
        self.set_volumeControl(simul_param.volume_control)

        # miscellaneous
        self.verbosity = simul_param.verbosity
        self.set_tipAsymptote(simul_param.tip_asymptote)
        self.saveRegime = simul_param.save_regime
        self.remeshFactor = simul_param.remesh_factor
        self.frontAdvancing = simul_param.front_advancing
        self.collectPerfData = simul_param.collect_perf_data

# ----------------------------------------------------------------------------------------------------------------------

    # setter and getter functions

    def set_tipAsymptote(self, tip_asymptote):
        tipAssymptOptions = ("K", "M", "Mt", "U", "MK")
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

    def set_outFileAddress(self, out_file_folder):
        # check operating system to get appropriate slash in the address

        import sys
        if "win32" in sys.platform or "win64" in sys.platform:
            slash = "\\"
        else:
            slash = "/"

        if out_file_folder is not 'None':
            self.saveToDisk = True
            if "\\" in out_file_folder:
                if slash != "\\":
                    raise SystemExit('Windows style slash in the given address on linux system.')
            elif "/" in out_file_folder:
                if slash != "/":
                    raise SystemExit('linux style slash in the given address on windows system!')

            if out_file_folder[-1] is slash:
                out_file_folder = out_file_folder[:-1]

            import os
            if not os.path.exists(out_file_folder):
                os.makedirs(out_file_folder)

            self.__outFileAddress = out_file_folder + slash
            self.lastSavedFile = 0

        else:
            out_file_folder = "." + slash + "_simulation_data_PyFrac"

            import os
            if not os.path.exists(out_file_folder):
                os.makedirs(out_file_folder)

            self.__outFileAddress = out_file_folder + slash
            self.lastSavedFile = 0


    def get_outFileAddress(self):
        return self.__outFileAddress

    def set_solTimeSeries(self, sol_t_srs):
        if isinstance(sol_t_srs, np.ndarray):
            self.__solTimeSeries = sol_t_srs
            self.FinalTime = max(sol_t_srs)
        elif sol_t_srs is None:
            self.__solTimeSeries = None
        else:
            raise ValueError("The given solution time series is not a numpy array!")

    def get_solTimeSeries(self):
        return self.__solTimeSeries

#-----------------------------------------------------------------------------------------------------------------------


class IterationProperties:
    """
    This class store performance data in the form of a tree
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
        self.subIterations = []

