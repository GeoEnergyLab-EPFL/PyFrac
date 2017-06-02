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


# todo !!! decide where to use system exit
class MaterialProperties:
    """
    Class defining the solid Material properties

    instance variables:
        Eprime (float)          : plain strain modulus
        K1c (ndarray-float)     : Linear-Elastic Plane-Strain Fracture Toughness for each cell
        Kprime (ndarray-float)  : 4*(2/pi)**0.5 * K1c 
        Cprime (ndarray-float)  : 2 * Carter's leak off coefficient
        SigmaO (ndarray-float)  : in-situ stress field
        grainSize (float)       : the grain size of the rock; used to calculate the relative roughness
    methods:
    """

    def __init__(self, Eprime, Toughness, Cl, SigmaO, grain_size, Mesh):
        """
        Arguments:
            Eprime (float)          : plain strain modulus
            Toughness (float)       : Linear-Elastic Plane-Strain Fracture Toughness
            Cl (float)              : Carter's leak off coefficient
            SigmaO (ndarray-float)  : in-situ stress field
            grainSize (float)       : the grain size of the rock; used to calculate the relative roughness
        """
        if isinstance(Eprime, np.ndarray):  # check if float or ndarray
            raise SystemExit("Eprime  can not be an array as input ! - homogeneous medium only ")
        else:
            self.Eprime = Eprime

        if isinstance(Toughness, np.ndarray):  # check if float or ndarray
            if Toughness.size == Mesh.NumberOfElts:  # check if size equal to the mesh size
                self.K1c = Toughness
                self.Kprime = (32 / math.pi) ** 0.5 * Toughness
            else:
                # error
                raise SystemExit('Error in the size of Toughness input ')
                return
        else:
            self.K1c = Toughness * np.ones((Mesh.NumberOfElts,), float)
            self.Kprime = (32 / math.pi) ** 0.5 * Toughness * np.ones((Mesh.NumberOfElts,), float)

        if isinstance(Cl, np.ndarray):  # check if float or ndarray
            if Cl.size == Mesh.NumberOfElts:  # check if size equal to the mesh size
                self.Cprime = 2. * Cl
            else:
                raise SystemExit('Error in the size of Leak-Off coefficient input ')
                return
        else:
            self.Cprime = 2. * Cl * np.ones((Mesh.NumberOfElts,), float)

        if isinstance(SigmaO, np.ndarray):  # check if float or ndarray
            if SigmaO.size == Mesh.NumberOfElts:  # check if size equal to the mesh size
                self.SigmaO = SigmaO
            else:
                raise SystemExit('Error in the size of Sigma input ')
                return
        else:
            self.SigmaO = SigmaO * np.ones((Mesh.NumberOfElts,), float)

        self.grainSize = grain_size


# --------------------------------------------------------------------------------------------------------

class FluidProperties:
    """
       Class defining the fluid properties

       instance variables:
            viscosity (ndarray-float):      Viscosity of the fluid (note its different from local viscosity, see
                                            fracture class for local viscosity) 
            rheology (string):              string specifying rheology of the fluid. Possible options:
                                                -- "Newtonian"
                                                -- "non-Newtonian"
            muPrime (float):                12 * viscosity (// plates viscosity factor)
            density (float, default 1000):  density of the fluid
            turbulence (bool, default False):turbulence flag. If true, turbulence will be taken into account
       methods:
       """

    def __init__(self, viscosity, mesh, density=1000., rheology="Newtonian", turbulence=False):

        if isinstance(viscosity, np.ndarray):  # check if float or ndarray
            raise SystemExit(' viscosity of the fluid is not an array. Note that its different from local viscosity (See local\n'
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
            raise SystemExit('Invalid input for rheology. Possible options: ' + repr(rheologyOptions))
            return

        self.muPrime = 12. * self.viscosity  # the geometric viscosity in the parallel plate solution
        self.density = density

        if isinstance(turbulence, bool):
            self.turbulence = turbulence
        else:
            # error
            raise SystemExit('Invalid turbulence flag. Can be either True or False')


# --------------------------------------------------------------------------------------------------------
class InjectionProperties:
    """
        Class defining the injection schedule

        instance variables:
            injectionRate (ndarray-float):      Array specifying the time series (row 0) and the corresponding injection
                                                rates (row 1).   
            source_coordinates (ndarray-float): Array with a single row and two columns specifying the x and y coordinate
                                                of the injection point coordinates.
            source_location (ndarray-int):      The element(s) where the fluid is injected in the cartesian mesh.                             
    """

    def __init__(self, rate, source_coordinates, Mesh):  # add Mesh as input directly here to ensure consistency
        """
        The constructor of the InjectionProperties class. See documentation of the class.
        """

        if isinstance(rate, np.ndarray):
            if rate.shape[0] != 2:
                raise SystemExit('Invalid injection rate. The list should have 2 rows (to specify time and corresponding '
                      'injection rate) for each entry')
            else:
                self.injectionRate = rate
        else:
            self.injectionRate = np.asarray([[0], [rate]])

        if len(source_coordinates) == 2:
            self.source_coordinates = source_coordinates
        else:
            # error
            raise SystemExit('Invalid source coordinates. Correct format: a numpy array with a single row and two columns to \n'
                  'specify x and y coordinate of the source e.g. np.array([x_coordinate, y_coordinate])')

        self.source_location = Mesh.locate_element(source_coordinates[0], source_coordinates[1])




# --------------------------------------------------------------------------------------------------------
class SimulationParameters:
    """
        Class defining the simulation parameters

        instance variables
            maxTimeSteps (integer, default 10):     maximum number of time steps.
            tolFractFront (float, default 1.e-3):   tolerance for the fracture front loop.
            toleranceEHL (float, default 1.e-5):    tolerance for the Elastohydrodynamic solver.
            maximumItrEHL (int, default 100):       maximum number of iterations for the Elastohydrodynamic solver.
            tmStpPrefactor (float, default 0.8):    factor for time-step adaptivity. 
            FinalTime (float, default 1000):        time where the simulation ends.
            maxFrontItr (int, default 30):          maximum iterations to for the fracture front loop.
            tipAsymptote (string, default "U"):     propagation regime. Possible options:
                                                        regime -- K  toughness dominated regime, without leak off
                                                        regime -- M  viscosity dominated regime, without leak off
                                                        regime -- Mt viscosity dominated regime , with leak off
                                                        regime -- U  Universal regime accommodating viscosity, toughness 
                                                                     and leak off (see Donstov and Pierce, 2017)
                                                        regime -- M-K transition regime
            maxSolverItr (int, default 100):        maximum iterations for the EHL iterative solver (Picard-Newton
                                                    hybrid) in this case.
            maxReattempts (int, default 5):         maximum number of reattempts in case of failure of a time step. A
                                                    smaller time step will be attempted the given number of times. 
            reAttemptFactor (float, default 0.8):   the factor multiplied with the time step before reattempt.
            outputTimePeriod (float, default inf):  the time period after which the output file is written or the
                                                    figures are plotted. 
            plotFigure (boolean, default False):    flag specifying to plot fracture trace after the given time period
            saveToDisk (boolean, default False):    flag specifying to save fracture to dist after the given time period
            out_file_address (string, default "None"): disk address of the files to be saved. If not given, a new
                                                    ./Data/"tim stamp" folder will be automatically created.
            plot_analytical (boolean, default False): flag specifying to plot the analytical solution
            analyticalSol (String, default "M"):    the analytical solution of the radial fracture to be plotted on the
                                                    fracture. Possible options:
                                                        "M" -- viscosity dominated
                                                        "K" -- toughness dominated
            plot_evolution (boolean, default False):if True, the fracture footprint plots will be superimposed on the 
                                                    previous footprint plots i.e. evolution of fracture with time will
                                                    be shown
            
    """

    def __init__(self, toleranceFractureFront=1.0e-3, toleranceEHL=1.0e-5, maxfront_its=30, max_itr_solver=100,
                 tmStp_prefactor=0.4, tip_asymptote='U', final_time=1000., maximum_steps=1000, max_reattemps = 5,
                 reattempt_factor = 0.8, output_time_period = np.inf, plot_figure = False, save_to_disk = False,
                 out_file_folder = "None", plot_analytical = False, analytical_sol = "M", plot_evolution=True):
        """
        
        The constructor of the SimulationParameters class. See documentation of the class.
        """
        self.maxTimeSteps = maximum_steps
        self.tolFractFront = toleranceFractureFront
        self.toleranceEHL = toleranceEHL
        self.tmStpPrefactor = tmStp_prefactor
        self.FinalTime = final_time

        # todo: all the option structures can be put into one file
        tipAssymptOptions = ("K", "M", "Mt", "U", "M-K")
        if tip_asymptote in tipAssymptOptions:  # check if tip asymptote matches any option
            self.tipAsymptote = tip_asymptote
        else:
            # error
            raise SystemExit('Invalid tip asymptote. Possible options: ' + repr(tipAssymptOptions))
            return

        self.maxFrontItr = maxfront_its
        self.maxSolverItr = max_itr_solver
        self.maxReattempts = max_reattemps
        self.reAttemptFactor = reattempt_factor

        # output parameters
        self.outputTimePeriod = output_time_period
        self.plotFigure = plot_figure
        if plot_figure:
            self.plotAnalytical = plot_analytical
            self.analyticalSol = analytical_sol

        self.saveToDisk = save_to_disk
        self.plotEvolution = plot_evolution
        # check operating system to get appropriate slash in the address
        import sys
        if "win" in sys.platform:
            slash = "\\"
        else:
            slash = "/"

        if out_file_folder == "None" and save_to_disk:
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
        elif save_to_disk:
            import os
            if not os.path.exists(out_file_folder):
                os.makedirs(out_file_folder)

            self.outFileAddress = out_file_folder + slash
            self.lastSavedFile = 0

# ----------------------------------------------------------------------------------------------------------------------
