#
# This file is part of PyFrac.
#
# Created by Brice Lecampion on 03.04.17.
# Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2017.  All rights reserved.
# See the LICENSE.TXT file for more details. 
#

import math
import numpy as np


from src.CartesianMesh import *


class MaterialProperties:
    """
    Class defining the Material properties of the solid

    Instance variables:
        Eprime (float)       -- plain strain modulus
        K1c (ndarray)        -- Linear-Elastic Plane-Strain Toughness for each cell
        Kprime (ndarray)     -- 4*(2/pi)**0.5 * K1c
        Cprime (ndarray)     -- 2 * Carter's leak off coefficient
        SigmaO (ndarray)     -- in-situ stress field
        grainSize (float)    -- the grain size of the rock; used to calculate the relative roughness
    Methods:
    """

    def __init__(self, Mesh, Eprime, Toughness, Cl=0., SigmaO=0., grain_size=0., Kprime_func=None,
                 anisotropic_flag=False, Toughness_min = None):
        """
        Arguments:
            Eprime (float)      -- plain strain modulus
            Toughness (float)   -- Linear-Elastic Plane-Strain Fracture Toughness
            Cl (float)          -- Carter's leak off coefficient
            SigmaO (ndarray)    -- in-situ stress field
            grainSize (float)   -- the grain size of the rock; used to calculate the relative roughness
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
        else:
            self.K1c = Toughness * np.ones((Mesh.NumberOfElts,), float)
            self.Kprime = (32 / math.pi) ** 0.5 * Toughness * np.ones((Mesh.NumberOfElts,), float)

        self.KprimeFunc = Kprime_func
        if not Kprime_func is None:
            #todo: serialize and dump Kprime function
            # import marshal
            # self.KpFunString = marshal.dumps(Kprime_func.func_code)
            pass

        if isinstance(Cl, np.ndarray):  # check if float or ndarray
            if Cl.size == Mesh.NumberOfElts:  # check if size equal to the mesh size
                self.Cprime = 2. * Cl
            else:
                raise ValueError('Error in the size of Leak-Off coefficient input!')
                return
        else:
            self.Cprime = 2. * Cl * np.ones((Mesh.NumberOfElts,), float)

        if isinstance(SigmaO, np.ndarray):  # check if float or ndarray
            if SigmaO.size == Mesh.NumberOfElts:  # check if size equal to the mesh size
                self.SigmaO = SigmaO
            else:
                raise ValueError('Error in the size of Sigma input ')
                return
        else:
            self.SigmaO = SigmaO * np.ones((Mesh.NumberOfElts,), float)

        self.grainSize = grain_size
        self.anisotropic = anisotropic_flag
        if anisotropic_flag:
            try:
                self.KprimeFunc(0)
            except TypeError:
                raise SystemExit('The given Kprime function is not correct for anisotropic case! It should take one'
                                 ' argument, i.e. the angle and return a toughness value.')
            if Toughness_min is None:
                raise SystemExit('Two toughnesses in the orthogonal directions should be provided.')
        self.K1c_perp = Toughness_min

        if not Kprime_func is None and not self.anisotropic:
            try:
                self.KprimeFunc(0.,0.)
            except TypeError:
                raise SystemExit('The  given Kprime function is not correct! It should take two arguments, '
                           'i.e. the x and y coordinates of a point and return the toughness at this point.')

# --------------------------------------------------------------------------------------------------------

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

    def __init__(self, viscosity, density=1000., rheology="Newtonian", turbulence=False):
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
        if isinstance(viscosity, np.ndarray):  # check if float or ndarray
            raise ValueError(' viscosity of the fluid is not an array. Note that its different from local viscosity (See local\n'
                  ' viscosity variable in the fracture class')
            return
        else:
            # uniform viscosity
            self.viscosity = viscosity

        rheologyOptions = ("Newtonian", "non-Newtonian")
        if rheology in rheologyOptions:  # check if rheology match to any rheology option
            self.rheology = rheology
        else:
            # error
            raise ValueError('Invalid input for rheology. Possible options: ' + repr(rheologyOptions))
            return

        self.muPrime = 12. * self.viscosity  # the geometric viscosity in the parallel plate solution
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
            maxTimeSteps (integer)      -- maximum number of time steps.
            tolFractFront (float)       -- tolerance for the fracture front loop.
            toleranceEHL (float)        -- tolerance for the Elastohydrodynamic solver.
            maximumItrEHL (int)         -- maximum number of iterations for the Elastohydrodynamic solver.
            tmStpPrefactor (float)      -- factor for time-step adaptivity.
            tmStpPrefactor_max (float)  -- used in the case of re-attempt from five steps back.
            FinalTime (float)           -- time where the simulation ends.
            maxFrontItr (int)           -- maximum iterations to for the fracture front loop.
            tipAsymptote (string)       -- propagation regime. Possible options:
                                                - K  (toughness dominated regime, without leak off)
                                                - M  (viscosity dominated regime, without leak off)
                                                - Mt (viscosity dominated regime , with leak off)
                                                - U  (Universal regime accommodating viscosity, toughness
                                                     and leak off (see Donstov and Pierce, 2017))
                                                - MK (viscosity to toughness transition regime)
            maxSolverItr (int)          -- maximum iterations for the EHL iterative solver (Picard-Newton
                                           hybrid) in this case.
            maxReattempts (int)         -- maximum number of reattempts in case of failure of a time step. A smaller
                                           time step will be attempted the given number of times.
            reAttemptFactor (float)     -- the factor multiplied with the time step before reattempt.
            outputTimePeriod (float)    -- the time period after which the output file is written or the
                                           figures are plotted.
            plotFigure (boolean)        -- flag specifying to plot fracture trace after the given time period.
            plotAnalytical (boolean)    -- if true, analytical solution will also be plotted along with the computed
                                           solution.
            analyticalSol (String)      -- the analytical solution of the radial fracture to be plotted on the
                                           fracture. Possible options:
                                                - "M" (viscosity dominated)
                                                - "K" (toughness dominated)
            saveToDisk (boolean)        -- flag specifying to save fracture to dist after the given time period.
            out_file_address (string)   -- disk address of the files to be saved. If not given, a new
                                           ./Data/"tim stamp" folder will be automatically created.

            toleranceToughness (float)  -- tolerance for toughness iteration
            solTimeSeries (ndarray)     -- time series where the solution is required. The time stepping would be
                                           adjusted to get solution exactly at the given times.
            plotEltType (boolean)       -- if True, the type of element (tip, ribbon or channel) will be depicted with
                                            color coded dots
            saveRegime (boolean)        -- if True, the regime of the propagation (see Zia and Lecampion 2018) will be
                                           saved.
            
    """

    def __init__(self, address = None):
        """
        The constructor of the SimulationParameters class. See documentation of the class.

        Arguments:

        """

        if address is None:
            import default_SimParam as simul_param
        else:
            import sys
            sys.path.append(address)
            import simul_param
            sys.path.remove(address)


        self.maxTimeSteps = simul_param.maximum_steps
        self.tolFractFront = simul_param.toleranceFractureFront
        self.toleranceEHL = simul_param.toleranceEHL
        self.tmStpPrefactor = simul_param.tmStp_prefactor
        self.tmStpPrefactor_max = simul_param.tmStp_prefactor
        self.FinalTime = simul_param.final_time

        # todo: all the option structures can be put into one file
        tipAssymptOptions = ("K", "M", "Mt", "U", "MK")
        if simul_param.tip_asymptote in tipAssymptOptions:  # check if tip asymptote matches any option
            self.tipAsymptote = simul_param.tip_asymptote
        else:
            # error
            raise ValueError('Invalid tip asymptote. Possible options: ' + repr(tipAssymptOptions))

        self.maxFrontItr = simul_param.maxfront_its
        self.maxSolverItr = simul_param.max_itr_solver
        self.maxReattempts = simul_param.max_reattemps
        self.reAttemptFactor = simul_param.reattempt_factor

        # output parameters
        if isinstance(simul_param.req_sol_at, np.ndarray):
            self.solTimeSeries = simul_param.req_sol_at
            self.FinalTime = max(simul_param.req_sol_at)
        else:
            # self.solTimeSeries = np.asarray([self.FinalTime], dtype=np.float64)
            self.solTimeSeries = simul_param.req_sol_at
        self.outputTimePeriod = simul_param.output_time_period
        self.plotFigure = simul_param.plot_figure
        self.plotAnalytical = simul_param.plot_analytical
        self.analyticalSol = simul_param.analytical_sol
        self.saveToDisk = simul_param.save_to_disk

        # toughness anisotropy
        self.toleranceToughness = simul_param.tol_toughness
        self.maxToughnessItr = simul_param.max_toughnessItr

        self.dryCrack_mechLoading = simul_param.mech_loading
        self.viscousInjection = simul_param.viscous_injection
        self.volumeControl = simul_param.volume_control
        self.timeStepLimit = simul_param.timeStep_limit
        self.tmStpFactLimit = simul_param.tmStp_fact_limit
        self.verbosity = simul_param.verbosity

        if simul_param.mech_loading or simul_param.volume_control:
            self.viscousInjection = False

        if simul_param.mech_loading:
            self.plotAnalytical = False

        self.bckColor = simul_param.bck_color
        self.plotEltType = simul_param.plot_eltType
        self.saveRegime = simul_param.save_regime
        self.remeshFactor = simul_param.remesh_factor

        # check operating system to get appropriate slash in the address
        import sys
        if "win32" in sys.platform or "win64" in sys.platform:
            slash = "\\"
        else:
            slash = "/"

        if simul_param.out_file_folder == "None" and simul_param.save_to_disk:
            # time stamp as the folder address
            from time import gmtime, strftime
            timeStamp = "runDate_"+ strftime("%Y-%m-%d_time_%Hh-%Mm-%Ss", gmtime())

            # output folder address
            address = "." + slash + "Data" + slash + timeStamp
            # check if folder exists
            import os
            if not os.path.exists(address):
                os.makedirs(address)

            self.outFileAddress = address + slash
            self.lastSavedFile = 0
        elif simul_param.save_to_disk:
            if "\\" in simul_param.out_file_folder:
                if slash != "\\":
                    raise SystemExit('Windows style slash in the given address on linux system.')
            elif "/" in simul_param.out_file_folder:
                if slash != "/":
                    raise SystemExit('linux style slash in the given address on windows system')

            import os
            if not os.path.exists(simul_param.out_file_folder):
                os.makedirs(simul_param.out_file_folder)

            self.outFileAddress = simul_param.out_file_folder + slash
            self.lastSavedFile = 0

# ----------------------------------------------------------------------------------------------------------------------
