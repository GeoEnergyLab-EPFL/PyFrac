# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on 11.05.17.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""
import logging
import copy
import matplotlib.pyplot as plt
import dill
import os
import numpy as np
import time
from time import gmtime, strftime
import warnings

# local imports
from properties import LabelProperties, IterationProperties, PlotProperties
from properties import instrument_start, instrument_close
from elasticity import load_isotropic_elasticity_matrix, load_TI_elasticity_matrix, mapping_old_indexes
from elasticity import load_isotropic_elasticity_matrix_toepliz
from mesh import CartesianMesh
from time_step_solution import attempt_time_step
from visualization import plot_footprint_analytical, plot_analytical_solution,\
                          plot_injection_source, get_elements
from symmetry import load_isotropic_elasticity_matrix_symmetric, symmetric_elasticity_matrix_from_full
from labels import TS_errorMessages, supported_projections, suitable_elements


class Controller:
    """
    This class describes the controller which takes the given material, fluid, injection and loading properties and
    advances a given fracture according to the provided simulation properties.
    """

    errorMessages = TS_errorMessages

    def __init__(self, Fracture, Solid_prop, Fluid_prop, Injection_prop, Sim_prop, Load_prop=None, C=None):
        """ The constructor of the Controller class.

        Args:
           Fracture (Fracture):                     -- the fracture to be propagated.
           Solid_prop (MaterialProperties):         -- the MaterialProperties object giving the material properties.
           Fluid_prop (FluidProperties):            -- the FluidProperties object giving the fluid properties.
           Injection_prop (InjectionProperties):    -- the InjectionProperties object giving the injection.
                                                       properties.
           Sim_prop (SimulationProperties):         -- the SimulationProperties object giving the numerical
                                                       parameters to be used in the simulation.
           Load_prop (LoadingProperties):           -- the LoadingProperties object specifying how the material is
                                                       mechanically loaded.
           C (ndarray):                             -- the elasticity matrix.

        """
        log = logging.getLogger('PyFrac.controller')
        self.fracture = Fracture
        self.solid_prop = Solid_prop
        self.fluid_prop = Fluid_prop
        self.injection_prop = Injection_prop
        self.sim_prop = Sim_prop
        self.load_prop = Load_prop
        self.C = C
        self.fr_queue = [None, None, None, None, None]  # queue of fractures from the last five time steps
        self.stepsFromChckPnt = 0
        self.tmStpPrefactor_copy = copy.copy(Sim_prop.tmStpPrefactor) # should be in simulation properties
        self.stagnant_TS = None     # time step if the front is stagnant. It is increased exponentialy to avoid uneccessary small steps.
        self.perfData = []
        self.lastSavedFile = 0
        self.lastSavedTime = np.NINF
        self.lastPlotTime = np.NINF
        self.TmStpCount = 0
        self.chkPntReattmpts = 0    # the number of re-attempts done from the checkpoint. Simulation is declared failed after 5 attempts.
        self.TmStpReductions = 0    # the number of times the time step has been reattempted because the fracture it was advancing too more than two cells in a row
        self.delta_w = None         # change in width between successive time steps. Used to limit time step.
        self.lstTmStp = None
        self.solveDetlaP_cp = self.sim_prop.solveDeltaP # copy of the flag indicating the solver to solve for pressure or delta p
        self.PstvInjJmp = None      # flag specifyung if the jump to the time of the next positive injection after the fracture is
                                        # fully closed is to be taken or not. Asked from user if it is None.
        self.fullyClosed = False    # should be related to the fracture state (thus in fracture class)
        self.setFigPos = True
        self.lastSuccessfulTS = Fracture.time
        self.maxTmStp = 0           # the maximum time step taken uptil now by the controller.


        # make a list of Nones with the size of the number of variables to plot during simulation
        self.Figures = [None for i in range(len(self.sim_prop.plotVar))]

        # Find the times where any parameter changes. These times will be added to the time series where the solution is
        # required to ensure the time is hit during time stepping and the change is applied at the exact time.
        param_change_at = np.array([], dtype=np.float64)
        if Injection_prop.injectionRate.shape[1] > 1:
           param_change_at = np.hstack((param_change_at, Injection_prop.injectionRate[0]))
        if isinstance(Sim_prop.fixedTmStp, np.ndarray):
           param_change_at = np.hstack((param_change_at, Sim_prop.fixedTmStp[0]))
        if isinstance(Sim_prop.tmStpPrefactor, np.ndarray):
           param_change_at = np.hstack((param_change_at, Sim_prop.tmStpPrefactor[0]))


        if len(param_change_at) > 0:
            if self.sim_prop.get_solTimeSeries() is not None:
                # add the times where any parameter changes to the required solution time series
                sol_time_srs = np.hstack((self.sim_prop.get_solTimeSeries(), param_change_at))
            else:
                sol_time_srs = param_change_at
            sol_time_srs = np.unique(sol_time_srs)
            if sol_time_srs[0] == 0:
                sol_time_srs = np.delete(sol_time_srs, 0)
        else:
           sol_time_srs = self.sim_prop.get_solTimeSeries()
        self.timeToHit = sol_time_srs

        if self.sim_prop.finalTime is None:
           if self.sim_prop.get_solTimeSeries() is None:
               ## Not necessarily an error
                raise ValueError("The final time to stop the simulation is not provided!")
           else:
               self.sim_prop.finalTime = np.max(self.sim_prop.get_solTimeSeries())
        else:
            if self.timeToHit is not None:
                greater_finalTime = np.where(self.timeToHit > self.sim_prop.finalTime)[0]
                self.timeToHit = np.delete(self.timeToHit, greater_finalTime)

        # Setting to volume control solver if viscosity is zero
        if self.fluid_prop.viscosity < 1e-15:
           log.info('Fluid viscosity is zero. Setting solver to volume control...')
           self.sim_prop.set_volumeControl(True)

        if not all(elem in self.fracture.EltChannel for elem in Injection_prop.sourceElem):
            message = 'INJECTION LOCATION ERROR: \n' \
                      'injection points are located outisde of the fracture footprints'
            raise SystemExit(message)

        # Setting whether sparse matrix is used to make fluid conductivity matrix
        if Sim_prop.solveSparse is None:
           if Fracture.mesh.NumberOfElts < 2500:
               Sim_prop.solveSparse = False
           else:
               Sim_prop.solveSparse = True

        # basic performance data
        self.remeshings = 0
        self.successfulTimeSteps = 0
        self.failedTimeSteps = 0

        # setting front advancing scheme to implicit if velocity is not available for the first time step.
        self.frontAdvancing = copy.copy(Sim_prop.frontAdvancing)
        if Sim_prop.frontAdvancing in ['explicit', 'predictor-corrector']:
            if np.nanmax(Fracture.v) <= 0 or np.isnan(Fracture.v).any():
                Sim_prop.frontAdvancing = 'implicit'

        if self.sim_prop.saveToDisk:
            self.logAddress = copy.copy(Sim_prop.get_outputFolder())
        else:
            self.logAddress = './'

        # setting up tip asymptote
        if self.fluid_prop.rheology in ["Herschel-Bulkley", "HBF"]:
            if self.sim_prop.get_tipAsymptote() not in ["HBF", "HBF_aprox", "HBF_num_quad"]:
                warnings.warn("Fluid rhelogy and tip asymptote does not match. Setting tip asymptote to \'HBF\'")
                self.sim_prop.set_tipAsymptote('HBF')
        if self.fluid_prop.rheology in ["power-law", "PLF"]:
            if self.sim_prop.get_tipAsymptote() not in ["PLF", "PLF_aprox", "PLF_num_quad", "PLF_M"]:
                warnings.warn("Fluid rhelogy and tip asymptote does not match. Setting tip asymptote to \'PLF\'")
                self.sim_prop.set_tipAsymptote('PLF')
        if self.fluid_prop.rheology == 'Newtonian':
            if self.sim_prop.get_tipAsymptote() not in ["K", "M", "Mt", "U", "U1", "MK", "MDR", "M_MDR"]:
                warnings.warn("Fluid rhelogy and tip asymptote does not match. Setting tip asymptote to \'U\'")
                self.sim_prop.set_tipAsymptote('U1')

        # if you set the code to advance max 1 cell then remove the SimulProp.timeStepLimit
        if self.sim_prop.timeStepLimit is not None and self.sim_prop.limitAdancementTo2cells is True:
            if self.sim_prop.forceTmStpLmtANDLmtAdvTo2cells == False:
                warnings.warn("You have set sim_prop.limitAdancementTo2cells = True. This imply that sim_prop.timeStepLimit will be deactivated.")
                self.sim_prop.timeStepLimit = None
            else:
                warnings.warn(
                    "You have forced <limitAdancementTo2cells> to be True and set <timeStepLimit> - the first one might be uneffective onto the second one until the prefactor has been reduced to produce a time step < timeStepLimit")
#-----------------------------------------------------------------------------------------------------------------------

    def run(self):
        """
        This function runs the simulation according to the parameters given in the properties classes. See especially
        the documentation of the :py:class:`properties.SimulationProperties` class to get details of the parameters
        controlling the simulation run.
        """
        log = logging.getLogger('PyFrac.controller.run')
        log_only_to_logfile = logging.getLogger('PyFrac_LF.controller.run')

        # output initial fracture
        if self.sim_prop.saveToDisk:
            # save properties
            if not os.path.exists(self.sim_prop.get_outputFolder()):
                os.makedirs(self.sim_prop.get_outputFolder())

            prop = (self.solid_prop, self.fluid_prop, self.injection_prop, self.sim_prop)
            with open(self.sim_prop.get_outputFolder() + "properties", 'wb') as output:
                dill.dump(prop, output, -1)

        if self.sim_prop.plotFigure or self.sim_prop.saveToDisk:
            # save or plot fracture
            self.output(self.fracture)
            self.lastSavedTime = self.fracture.time

        if self.sim_prop.log2file:
            self.sim_prop.set_logging_to_file(self.logAddress)

        # deactivate the block_toepliz_compression functions
        # DO THIS CHECK BEFORE COMPUTING C!
        if self.C is not None: # in the case C is provided
            self.sim_prop.useBlockToeplizCompression = False
        elif self.solid_prop.TI_elasticity: # in case of TI_elasticity
            self.sim_prop.useBlockToeplizCompression = False
        elif not self.solid_prop.TI_elasticity and self.sim_prop.symmetric:  # in case you save 1/4 of the elasticity due to domain symmetry
            self.sim_prop.useBlockToeplizCompression = False

        # load elasticity matrix
        if self.C is None:
            log.info("Making elasticity matrix...")
            if self.sim_prop.symmetric:
                if not self.sim_prop.get_volumeControl():
                    raise ValueError("Symmetric fracture is only supported for inviscid fluid yet!")

            if not self.solid_prop.TI_elasticity:
                if self.sim_prop.symmetric:
                    self.C = load_isotropic_elasticity_matrix_symmetric(self.fracture.mesh,
                                                                        self.solid_prop.Eprime)
                else:
                    if not self.sim_prop.useBlockToeplizCompression:
                        self.C = load_isotropic_elasticity_matrix(self.fracture.mesh,
                                                                  self.solid_prop.Eprime)
                    else:
                        self.C = load_isotropic_elasticity_matrix_toepliz(self.fracture.mesh,
                                                                          self.solid_prop.Eprime)
            else:
                C = load_TI_elasticity_matrix(self.fracture.mesh,
                                                   self.solid_prop,
                                                   self.sim_prop)
                # compressing the elasticity matrix for symmetric fracture
                if self.sim_prop.symmetric:
                    self.C = symmetric_elasticity_matrix_from_full(C, self.fracture.mesh)
                else:
                    self.C = C
            log.info('Done!')

        # # perform first time step with implicit front advancing due to non-availability of velocity
        # if not self.sim_prop.symmetric:
        #     if self.sim_prop.frontAdvancing == "predictor-corrector":
        #         self.sim_prop.frontAdvancing = "implicit"

        log.info("Starting time = " + repr(self.fracture.time))
        # starting time stepping loop
        while self.fracture.time < 0.999 * self.sim_prop.finalTime and self.TmStpCount < self.sim_prop.maxTimeSteps:

            timeStep = self.get_time_step()

            if self.sim_prop.collectPerfData:
                tmStp_perf = IterationProperties(itr_type="time step")
            else:
                tmStp_perf = None

            # advancing time step
            status, Fr_n_pls1 = self.advance_time_step(self.fracture,
                                                         self.C,
                                                         timeStep,
                                                         tmStp_perf)

            if self.sim_prop.collectPerfData:
                tmStp_perf.CpuTime_end = time.time()
                tmStp_perf.status = status == 1
                tmStp_perf.failure_cause = self.errorMessages[status]
                tmStp_perf.time = self.fracture.time
                tmStp_perf.NumbOfElts = len(self.fracture.EltCrack)
                self.perfData.append(tmStp_perf)

            if status == 1:
            # Successful time step
                log.info("Time step successful!")
                log.debug("Element in the crack: "+str(len(Fr_n_pls1.EltCrack)))
                log.debug("Nx: " + str(Fr_n_pls1.mesh.nx))
                log.debug("Ny: " + str(Fr_n_pls1.mesh.nx))
                log.debug("hx: " + str(Fr_n_pls1.mesh.hx))
                log.debug("hy: " + str(Fr_n_pls1.mesh.hy))
                self.delta_w = Fr_n_pls1.w - self.fracture.w
                self.lstTmStp = Fr_n_pls1.time - self.fracture.time
                # output
                if self.sim_prop.plotFigure or self.sim_prop.saveToDisk:
                    if Fr_n_pls1.time > self.lastSavedTime:
                        self.output(Fr_n_pls1)

                # add the advanced fracture to the last five fractures list
                self.fracture = copy.deepcopy(Fr_n_pls1)
                self.fr_queue[self.successfulTimeSteps % 5] = copy.deepcopy(Fr_n_pls1)

                if self.fracture.time > self.lastSuccessfulTS:
                    self.lastSuccessfulTS = self.fracture.time
                if self.maxTmStp < self.lstTmStp:
                    self.maxTmStp = self.lstTmStp
                # put check point reattempts to zero if the simulation has advanced past the time where it failed
                if Fr_n_pls1.time > self.lastSuccessfulTS + 2 * self.maxTmStp:
                    self.chkPntReattmpts = 0
                    # set the prefactor to the original value after four time steps (after the 5 time steps back jump)
                    self.sim_prop.tmStpPrefactor = self.tmStpPrefactor_copy
                self.successfulTimeSteps += 1
                # set to 0 the counter of time step reductions
                if self.TmStpReductions > 0:
                    self.TmStpReductions = 0
                    self.sim_prop.tmStpPrefactor = self.tmStpPrefactor_copy
                # resetting the parameters for closure
                if self.fullyClosed:
                    # set to solve for pressure if the fracture was fully closed in last time step and is open now
                    self.sim_prop.solveDeltaP = False
                else:
                    self.sim_prop.solveDeltaP = self.solveDetlaP_cp
                self.PstvInjJmp = None
                self.fullyClosed = False

                # set front advancing back as set in simulation properties originally if velocity becomes available.
                if np.max(Fr_n_pls1.v) > 0 or not np.isnan(Fr_n_pls1.v).any():
                    self.sim_prop.frontAdvancing = copy.copy(self.frontAdvancing)
                else:
                    self.sim_prop.frontAdvancing = 'implicit'

                if self.TmStpCount == self.sim_prop.maxTimeSteps:
                    log.warning("Max time steps reached!")

            elif status == 12 or status == 16:
                # re-meshing required
                if self.sim_prop.enableRemeshing:
                    # we need to decide which remeshings are to be considered
                    compress = False
                    if status == 16:
                        # we reached cell number limit so we adapt by compressing the domain accordingly

                        # calculate the new number of cells
                        new_elems = [int((self.fracture.mesh.nx + np.round(self.sim_prop.meshReductionFactor, 0))
                                         / self.sim_prop.meshReductionFactor),
                                     int((self.fracture.mesh.ny + np.round(self.sim_prop.meshReductionFactor, 0))
                                         / self.sim_prop.meshReductionFactor)]
                        if new_elems[0] % 2 == 0:
                            new_elems[0] = new_elems[0] + 1
                        if new_elems[1] % 2 == 0:
                            new_elems[1] = new_elems[1] + 1

                        # Decide if we still can reduce the number of elements
                        if (2 * self.fracture.mesh.Lx / new_elems[0] > self.sim_prop.maxCellSize) or (2 *
                            self.fracture.mesh.Ly / new_elems[1] > self.fracture.mesh.hy / self.fracture.mesh.hx *
                            self.sim_prop.maxCellSize):
                            log.warning("Reduction of cells not possible as minimal cell size would be violated!")
                            self.sim_prop.meshReductionPossible = False
                        else:

                            log.info("Reducing cell number...")

                            # We need to make sure the injection point stays where it is. We also do this for two points
                            # on same x or y
                            if len(self.fracture.source) == 1:
                                index = self.fracture.source[0]
                                cent_point = self.fracture.mesh.CenterCoor[self.fracture.source[0]]

                                reduction_factor = self.sim_prop.meshReductionFactor
                            elif len(self.fracture.source) == 2:
                                index = self.fracture.source[0]
                                cent_point = self.fracture.mesh.CenterCoor[self.fracture.source[0]]

                                if self.fracture.mesh.CenterCoor[self.fracture.source[0]][0] == \
                                        self.fracture.mesh.CenterCoor[self.fracture.source[1]][0]:
                                    elems_inter = int(abs(self.fracture.mesh.CenterCoor[self.fracture.source[0]][1] -
                                        self.fracture.mesh.CenterCoor[self.fracture.source[1]][1]) / \
                                                  self.fracture.mesh.hy)
                                    new_inter = int(np.ceil(elems_inter/self.sim_prop.meshReductionFactor))
                                    reduction_factor = elems_inter / new_inter

                                elif self.fracture.mesh.CenterCoor[self.fracture.source[0]][1] == \
                                        self.fracture.mesh.CenterCoor[self.fracture.source[1]][1]:
                                    elems_inter = int(abs(self.fracture.mesh.CenterCoor[self.fracture.source[0]][0] -
                                        self.fracture.mesh.CenterCoor[self.fracture.source[1]][0]) / \
                                                  self.fracture.mesh.hx)
                                    new_inter = int(np.ceil(elems_inter / self.sim_prop.meshReductionFactor))
                                    reduction_factor = elems_inter / new_inter

                                else:
                                    reduction_factor = self.sim_prop.meshReductionFactor

                                log.info("The real reduction factor used is " + repr(reduction_factor))

                            else:
                                index = self.fracture.mesh.locate_element(0., 0.)
                                cent_point = np.asarray([0., 0.])

                                reduction_factor = self.sim_prop.meshReductionFactor

                            row = int(index/self.fracture.mesh.nx)
                            column = index - self.fracture.mesh.nx * row

                            row_frac = (self.fracture.mesh.ny - (row + 1))/row
                            col_frac = column/(self.fracture.mesh.nx - (column + 1))

                            # calculate the new number of cells
                            new_elems = [int((self.fracture.mesh.nx + np.round(reduction_factor, 0))
                                             / reduction_factor),
                                         int((self.fracture.mesh.ny + np.round(reduction_factor, 0))
                                             / reduction_factor)]
                            if new_elems[0] % 2 == 0:
                                new_elems[0] = new_elems[0] + 1
                            if new_elems[1] % 2 == 0:
                                new_elems[1] = new_elems[1] + 1


                            # We calculate the new dimension of the meshed area
                            new_limits = [[cent_point[0] - round((new_elems[0] - 1)/(1 / col_frac + 1)) *
                                           self.fracture.mesh.hx * reduction_factor,
                                           cent_point[0] + (new_elems[0] - round((new_elems[0] - 1)/(1 / col_frac + 1))
                                                            - 1) * self.fracture.mesh.hx *
                                           reduction_factor],
                                          [cent_point[1] - round((new_elems[1] - 1) / (row_frac + 1)) *
                                           self.fracture.mesh.hy * reduction_factor,
                                           cent_point[1] + (new_elems[1] - round((new_elems[1] - 1) / (row_frac + 1))
                                                            - 1) * self.fracture.mesh.hy *
                                           reduction_factor]]

                            elems = new_elems

                            self.remesh(new_limits, elems, 'reduce')

                            # set all other to zero
                            side_bools = [False, False, False, False]

                    elif status == 12:
                        if self.sim_prop.meshExtensionAllDir:
                            # we extend no matter how many boundaries we have hit
                            # ensure all directions to extend are true
                            self.sim_prop.set_mesh_extension_direction(['all'])

                        front_indices = \
                        np.intersect1d(self.fracture.mesh.Frontlist, Fr_n_pls1.EltTip, return_indices=True)[1]
                        side_bools = [(front_indices <= Fr_n_pls1.mesh.nx - 3).any(),
                                      (front_indices[front_indices > Fr_n_pls1.mesh.nx - 3]
                                       <= 2 * (Fr_n_pls1.mesh.nx - 3) + 1).any(),
                                      (front_indices[front_indices >= 2 * (Fr_n_pls1.mesh.nx - 2)] % 2 == 0).any(),
                                      (front_indices[front_indices >= 2 * (Fr_n_pls1.mesh.nx - 2)] % 2 != 0).any()]
                        # side_bools is a set of booleans telling us which sides are touched by the remeshing.
                        # First boolean represents bottom, top, left, right

                        if not self.sim_prop.meshExtensionAllDir:
                            compress = \
                                not np.asarray(np.asarray(self.sim_prop.meshExtension) * np.asarray(side_bools)).any() \
                                or (len(np.asarray(side_bools)[np.asarray(side_bools) == True]) > 3)


                    # This is the classical remeshing where the sides of the elements are multiplied by a constant.
                    if compress:
                        log.info("Remeshing by compressing the domain...")

                        # We need to make sure the injection point stays where it is. We also do this for two points
                        # on same x or y
                        if len(self.fracture.source) == 1:
                            index = self.fracture.source[0]
                            cent_point = self.fracture.mesh.CenterCoor[self.fracture.source[0]]

                            compression_factor = self.sim_prop.remeshFactor
                        elif len(self.fracture.source) == 2:
                            index = self.fracture.source[0]
                            cent_point = self.fracture.mesh.CenterCoor[self.fracture.source[0]]

                            if self.fracture.mesh.CenterCoor[self.fracture.source[0]][0] == \
                                    self.fracture.mesh.CenterCoor[self.fracture.source[1]][0]:
                                elems_inter = int(abs(self.fracture.mesh.CenterCoor[self.fracture.source[0]][1] -
                                                      self.fracture.mesh.CenterCoor[self.fracture.source[1]][1]) / \
                                                  self.fracture.mesh.hy)
                                new_inter = int(np.ceil(elems_inter / self.sim_prop.remeshFactor))
                                compression_factor = elems_inter / new_inter

                            elif self.fracture.mesh.CenterCoor[self.fracture.source[0]][1] == \
                                    self.fracture.mesh.CenterCoor[self.fracture.source[1]][1]:
                                elems_inter = int(abs(self.fracture.mesh.CenterCoor[self.fracture.source[0]][0] -
                                                      self.fracture.mesh.CenterCoor[self.fracture.source[1]][0]) / \
                                                  self.fracture.mesh.hx)
                                new_inter = int(np.ceil(elems_inter / self.sim_prop.remeshFactor))
                                compression_factor = elems_inter / new_inter

                            else:
                                compression_factor = self.sim_prop.remeshFactor

                            log.info("The real reduction factor used is " + repr(compression_factor))

                        else:
                            index = self.fracture.mesh.locate_element(0., 0.)
                            cent_point = np.asarray([0., 0.])

                            compression_factor = self.sim_prop.remeshFactor

                        row = int(index / self.fracture.mesh.nx)
                        column = index - self.fracture.mesh.nx * row

                        row_frac = (self.fracture.mesh.ny - (row + 1)) / row
                        col_frac = column / (self.fracture.mesh.nx - (column + 1))

                        # We calculate the new dimension of the meshed area
                        new_limits = [[cent_point[0] - round((self.fracture.mesh.nx - 1) / (1 / col_frac + 1)) *
                                       self.fracture.mesh.hx * compression_factor,
                                       cent_point[0] + (self.fracture.mesh.nx - round((self.fracture.mesh.nx - 1) /
                                                                                      (1 / col_frac + 1)) - 1) *
                                       self.fracture.mesh.hx * compression_factor],
                                      [cent_point[1] - round((self.fracture.mesh.ny - 1) / (row_frac + 1)) *
                                       self.fracture.mesh.hy * compression_factor,
                                       cent_point[1] + (self.fracture.mesh.ny - round((self.fracture.mesh.ny - 1) /
                                                                                       (row_frac + 1)) - 1) *
                                       self.fracture.mesh.hy * compression_factor]]

                        # # We calculate the new dimension of the meshed area
                        # new_dimensions = 2 * self.sim_prop.remeshFactor * np.asarray([self.fracture.mesh.Lx,
                        #                                                           self.fracture.mesh.Ly])
                        # new_limits = [[(self.fracture.mesh.domainLimits[2]+self.fracture.mesh.domainLimits[3]) / 2
                        #                - new_dimensions[0]/2, (self.fracture.mesh.domainLimits[2] +
                        #                                        self.fracture.mesh.domainLimits[3]) / 2
                        #                + new_dimensions[0]/2],
                        #               [(self.fracture.mesh.domainLimits[0]+self.fracture.mesh.domainLimits[1]) / 2
                        #                - new_dimensions[1]/2, (self.fracture.mesh.domainLimits[0] +
                        #                                        self.fracture.mesh.domainLimits[1]) / 2
                        #                + new_dimensions[1]/2]]

                        elems = [self.fracture.mesh.nx, self.fracture.mesh.ny]

                        if len(np.intersect1d(self.fracture.mesh.CenterElts, cent_point)) == 0:
                            compression_factor = 10

                        self.remesh(new_limits, elems, rem_factor=compression_factor)

                        side_bools = [False, False, False, False]

                    else:
                        nx_init = self.fracture.mesh.nx
                        ny_init = self.fracture.mesh.ny
                        for side in range(4):
                            if np.asarray(np.asarray(self.sim_prop.meshExtension) * np.asarray(side_bools))[side]:
                                if side == 0:

                                    elems_add = int(ny_init * (self.sim_prop.meshExtensionFactor[side] - 1))
                                    if elems_add % 2 != 0:
                                        elems_add = elems_add + 1

                                    if not self.sim_prop.symmetric:
                                        log.info("Remeshing by extending towards negative y...")
                                        new_limits = [[self.fracture.mesh.domainLimits[2],
                                                       self.fracture.mesh.domainLimits[3]],
                                                      [self.fracture.mesh.domainLimits[0] -
                                                       elems_add * self.fracture.mesh.hy,
                                                       self.fracture.mesh.domainLimits[1]]]
                                    else:
                                        log.info("Remeshing by extending in vertical direction to keep symmetry...")
                                        new_limits = [[self.fracture.mesh.domainLimits[2],
                                                       self.fracture.mesh.domainLimits[3]],
                                                      [self.fracture.mesh.domainLimits[0] -
                                                       elems_add * self.fracture.mesh.hy,
                                                       self.fracture.mesh.domainLimits[1] +
                                                       elems_add * self.fracture.mesh.hy]]
                                        side_bools[1] = False

                                    direction = 'bottom'

                                    elems = [self.fracture.mesh.nx, self.fracture.mesh.ny + elems_add]


                                if side == 1:

                                    elems_add = int(ny_init * (self.sim_prop.meshExtensionFactor[side] - 1))
                                    if elems_add % 2 != 0:
                                        elems_add = elems_add + 1

                                    if not self.sim_prop.symmetric:
                                        log.info("Remeshing by extending towards positive y...")
                                        new_limits = [[self.fracture.mesh.domainLimits[2],
                                                       self.fracture.mesh.domainLimits[3]],
                                                      [self.fracture.mesh.domainLimits[0],
                                                       self.fracture.mesh.domainLimits[1] +
                                                       elems_add * self.fracture.mesh.hy]]
                                    else:
                                        log.info("Remeshing by extending in vertical direction to keep symmetry...")
                                        new_limits = [[self.fracture.mesh.domainLimits[2],
                                                       self.fracture.mesh.domainLimits[3]],
                                                      [self.fracture.mesh.domainLimits[0] -
                                                       elems_add * self.fracture.mesh.hy,
                                                       self.fracture.mesh.domainLimits[1] +
                                                       elems_add * self.fracture.mesh.hy]]
                                        side_bools[0] = False

                                    direction = 'top'

                                    elems = [self.fracture.mesh.nx, self.fracture.mesh.ny + elems_add]

                                if side == 2:

                                    elems_add = int(nx_init * (self.sim_prop.meshExtensionFactor[side] - 1))
                                    if elems_add % 2 != 0:
                                        elems_add = elems_add + 1

                                    if not self.sim_prop.symmetric:
                                        log.info("Remeshing by extending towards negative x...")
                                        new_limits = [
                                            [self.fracture.mesh.domainLimits[2] - elems_add * self.fracture.mesh.hx,
                                             self.fracture.mesh.domainLimits[3]],
                                            [self.fracture.mesh.domainLimits[0],
                                             self.fracture.mesh.domainLimits[1]]]
                                    else:
                                        log.info("Remeshing by extending in horizontal direction to keep symmetry...")
                                        new_limits = [
                                            [self.fracture.mesh.domainLimits[2] - elems_add * self.fracture.mesh.hx,
                                             self.fracture.mesh.domainLimits[3] + elems_add * self.fracture.mesh.hx],
                                            [self.fracture.mesh.domainLimits[0],
                                             self.fracture.mesh.domainLimits[1]]]
                                        side_bools[3] = False

                                    direction = 'left'

                                    elems = [self.fracture.mesh.nx + elems_add, self.fracture.mesh.ny]

                                if side == 3:

                                    elems_add = int(nx_init * (self.sim_prop.meshExtensionFactor[side] - 1))
                                    if elems_add % 2 != 0:
                                        elems_add = elems_add + 1

                                    if not self.sim_prop.symmetric:
                                        log.info("Remeshing by extending towards positive x...")
                                        new_limits = [[self.fracture.mesh.domainLimits[2],
                                                       self.fracture.mesh.domainLimits[
                                                           3] + elems_add * self.fracture.mesh.hx],
                                                      [self.fracture.mesh.domainLimits[0],
                                                       self.fracture.mesh.domainLimits[1]]]
                                    else:
                                        log.info("Remeshing by extending in horizontal direction to keep symmetry...")
                                        new_limits = [
                                            [self.fracture.mesh.domainLimits[2] - elems_add * self.fracture.mesh.hx,
                                             self.fracture.mesh.domainLimits[3] + elems_add * self.fracture.mesh.hx],
                                            [self.fracture.mesh.domainLimits[0],
                                             self.fracture.mesh.domainLimits[1]]]
                                        side_bools[2] = False

                                    direction = 'right'

                                    elems = [self.fracture.mesh.nx + elems_add, self.fracture.mesh.ny]

                                self.remesh(new_limits, elems, direction=direction)
                                side_bools[side] = False

                    if np.asarray(side_bools).any():
                        log.info("Remeshing by compressing the domain...")

                        # We need to make sure the injection point stays where it is. We also do this for two points
                        # on same x or y
                        if len(self.fracture.source) == 1:
                            index = self.fracture.source[0]
                            cent_point = self.fracture.mesh.CenterCoor[self.fracture.source[0]]

                            compression_factor = self.sim_prop.remeshFactor
                        elif len(self.fracture.source) == 2:
                            index = self.fracture.source[0]
                            cent_point = self.fracture.mesh.CenterCoor[self.fracture.source[0]]

                            if self.fracture.mesh.CenterCoor[self.fracture.source[0]][0] == \
                                    self.fracture.mesh.CenterCoor[self.fracture.source[1]][0]:
                                elems_inter = int(abs(self.fracture.mesh.CenterCoor[self.fracture.source[0]][1] -
                                                      self.fracture.mesh.CenterCoor[self.fracture.source[1]][1]) / \
                                                  self.fracture.mesh.hy)
                                new_inter = int(np.ceil(elems_inter / self.sim_prop.remeshFactor))
                                compression_factor = elems_inter / new_inter

                            elif self.fracture.mesh.CenterCoor[self.fracture.source[0]][1] == \
                                    self.fracture.mesh.CenterCoor[self.fracture.source[1]][1]:
                                elems_inter = int(abs(self.fracture.mesh.CenterCoor[self.fracture.source[0]][0] -
                                                      self.fracture.mesh.CenterCoor[self.fracture.source[1]][0]) / \
                                                  self.fracture.mesh.hx)
                                new_inter = int(np.ceil(elems_inter / self.sim_prop.remeshFactor))
                                compression_factor = elems_inter / new_inter

                            else:
                                compression_factor = self.sim_prop.remeshFactor

                            log.info("The real reduction factor used is " + repr(compression_factor))

                        else:
                            index = self.fracture.mesh.locate_element(0., 0.)
                            cent_point = np.asarray([0., 0.])

                            compression_factor = self.sim_prop.remeshFactor

                        row = int(index / self.fracture.mesh.nx)
                        column = index - self.fracture.mesh.nx * row

                        row_frac = (self.fracture.mesh.ny - (row + 1)) / row
                        col_frac = column / (self.fracture.mesh.nx - (column + 1))

                        # We calculate the new dimension of the meshed area
                        new_limits = [[cent_point[0] - round((self.fracture.mesh.nx - 1) / (1 / col_frac + 1)) *
                                       self.fracture.mesh.hx * compression_factor,
                                       cent_point[0] + (self.fracture.mesh.nx - round((self.fracture.mesh.nx - 1) /
                                                                                      (1 / col_frac + 1)) - 1) *
                                       self.fracture.mesh.hx * compression_factor],
                                      [cent_point[1] - round((self.fracture.mesh.ny - 1) / (row_frac + 1)) *
                                       self.fracture.mesh.hy * compression_factor,
                                       cent_point[1] + (self.fracture.mesh.ny - round((self.fracture.mesh.ny - 1) /
                                                                                      (row_frac + 1)) - 1) *
                                       self.fracture.mesh.hy * compression_factor]]

                        # # We calculate the new dimension of the meshed area
                        # new_dimensions = 2 * self.sim_prop.remeshFactor * np.asarray([self.fracture.mesh.Lx,
                        #                                                           self.fracture.mesh.Ly])
                        # new_limits = [[(self.fracture.mesh.domainLimits[2]+self.fracture.mesh.domainLimits[3]) / 2
                        #                - new_dimensions[0]/2, (self.fracture.mesh.domainLimits[2] +
                        #                                        self.fracture.mesh.domainLimits[3]) / 2
                        #                + new_dimensions[0]/2],
                        #               [(self.fracture.mesh.domainLimits[0]+self.fracture.mesh.domainLimits[1]) / 2
                        #                - new_dimensions[1]/2, (self.fracture.mesh.domainLimits[0] +
                        #                                        self.fracture.mesh.domainLimits[1]) / 2
                        #                + new_dimensions[1]/2]]

                        elems = [self.fracture.mesh.nx, self.fracture.mesh.ny]

                        if len(np.intersect1d(self.fracture.mesh.CenterElts, cent_point)) == 0:
                            compression_factor = 10

                        self.remesh(new_limits, elems, rem_factor=compression_factor)

                    log_only_to_logfile.info("\nRemeshed at " + repr(self.fracture.time))

                else:
                    log.info("Reached end of the domain. Exiting...")
                    break

            elif status == 14:
                # fracture fully closed
                self.output(Fr_n_pls1)
                if self.PstvInjJmp is None:
                    inp = input("Fracture is fully closed.\n\nDo you want to jump to"
                            " the time of next positive injection? [y/n]")
                    t0 = time.time()
                    while inp not in ['y', 'Y', 'n', 'N'] and time.time() - t0 < 600:
                        inp = input("Press y or n")

                    if inp == 'y' or inp == 'Y' or time.time() - t0 >= 600:
                        self.PstvInjJmp = True
                    else:
                        self.PstvInjJmp = False

                if self.PstvInjJmp:
                    self.sim_prop.solveDeltaP = False
                    # index of current time in the time series (first row) of the injection rate array
                    time_larger = np.where(Fr_n_pls1.time <= self.injection_prop.injectionRate[0, :])[0]
                    pos_inj = np.where(self.injection_prop.injectionRate[1, :] > 0)[0]
                    Qact = self.injection_prop.get_injection_rate(self.fracture.time, self.fracture)
                    after_time = np.intersect1d(time_larger, pos_inj)
                    if len(after_time) == 0 and max(Qact) == 0.:
                        log.warning("Positive injection not found!")
                        break
                    elif len(after_time) == 0:
                        jump_to = self.fracture.time + self.fracture.time * 0.1
                    else:
                        jump_to = min(self.injection_prop.injectionRate[0, np.intersect1d(time_larger, pos_inj)])
                    Fr_n_pls1.time = jump_to
                elif inp == 'n' or inp == 'N':
                    self.sim_prop.solveDeltaP = True
                self.fullyClosed = True
                self.fracture = copy.deepcopy(Fr_n_pls1)
            elif status == 17:
                # time step too big: you advanced more than one cell
                log.info("The fracture is advancing more than two cells in a row at time "+ repr(self.fracture.time))

                if self.TmStpReductions == self.sim_prop.maxReattemptsFracAdvMore2Cells:
                    log.warning("We can not reduce the time step more than that")
                    if self.sim_prop.collectPerfData:
                        if self.sim_prop.saveToDisk:
                            file_address = self.sim_prop.get_outputFolder() + "perf_data.dat"
                        else:
                            file_address = "./perf_data.dat"
                        with open(file_address, 'wb') as perf_output:
                            dill.dump(self.perfData, perf_output, -1)

                    log.info("\n\n---Simulation failed---")

                    raise SystemExit("Simulation failed.")
                else:
                    log.info("- limiting the time step - ")
                    # decrease time step pre-factor before taking the next fracture in the queue having last
                    # five time steps
                    if isinstance(self.sim_prop.tmStpPrefactor, np.ndarray):
                        indxCurTime = max(np.where(self.fracture.time >= self.sim_prop.tmStpPrefactor[0, :])[0])
                        self.sim_prop.tmStpPrefactor[1, indxCurTime] *= 0.5**self.TmStpReductions
                    else:
                        self.sim_prop.tmStpPrefactor *= 0.5**self.TmStpReductions
                    self.TmStpReductions += 1
            else:
                # time step failed
                log.warning("\n" + self.errorMessages[status])
                log.warning("\nTime step failed at = " + repr(self.fracture.time))
                # check if the queue with last 5 time steps is not empty, or max check points jumps done
                if self.fr_queue[self.successfulTimeSteps % 5] is None or \
                   self.chkPntReattmpts == 4:
                    if self.sim_prop.collectPerfData:
                        if self.sim_prop.saveToDisk:
                            file_address = self.sim_prop.get_outputFolder() + "perf_data.dat"
                        else:
                            file_address = "./perf_data.dat"
                        with open(file_address, 'wb') as perf_output:
                            dill.dump(self.perfData, perf_output, -1)

                    log.info("\n\n---Simulation failed---")

                    raise SystemExit("Simulation failed.")
                else:
                    # decrease time step pre-factor before taking the next fracture in the queue having last
                    # five time steps
                    if isinstance(self.sim_prop.tmStpPrefactor, np.ndarray):
                        indxCurTime = max(np.where(self.fracture.time >= self.sim_prop.tmStpPrefactor[0, :])[0])
                        self.sim_prop.tmStpPrefactor[1, indxCurTime] *= 0.8
                        current_PreFctr = self.sim_prop.tmStpPrefactor[1, indxCurTime]
                    else:
                        self.sim_prop.tmStpPrefactor *= 0.8
                        current_PreFctr = self.sim_prop.tmStpPrefactor

                    self.chkPntReattmpts += 1
                    self.fracture = copy.deepcopy(self.fr_queue[(self.successfulTimeSteps + self.chkPntReattmpts) % 5])
                    log.warning("Time step have failed despite of reattempts with slightly smaller/bigger time steps...\n"
                                  "Going " + repr(5 - self.chkPntReattmpts) + " time steps back and re-attempting with the"
                                    " time step pre-factor of " + repr(current_PreFctr))
                    self.failedTimeSteps += 1

            self.TmStpCount += 1

        print("\n")
        log.info("Final time = " + repr(self.fracture.time))
        log.info("-----Simulation finished------")
        log.info("number of time steps = " + repr(self.successfulTimeSteps))
        log.info("failed time steps = " + repr(self.failedTimeSteps))
        log.info("number of remeshings = " + repr(self.remeshings))

        plt.show(block=False)
        plt.close('all')

        if self.sim_prop.collectPerfData:
            file_address = self.sim_prop.get_outputFolder() + "perf_data.dat"
            os.makedirs(os.path.dirname(file_address), exist_ok=True)
            with open(file_address, 'wb') as output:
                dill.dump(self.perfData, output, -1)
        return True


#-----------------------------------------------------------------------------------------------------------------------

    def advance_time_step(self, Frac, C, timeStep, perfNode=None):
        """
        This function advances the fracture by the given time step. In case of failure, reattempts are made with smaller
        time steps.

        Arguments:
            Frac (Fracture object):         -- fracture object from the last time step
            C (ndarray-float):              -- the elasticity matrix
            timeStep (float):               -- time step to be attempted
            perfNode (IterationProperties)  -- An IterationProperties instance to store performance data

        Return:
            - exitstatus (int)        -- see documentation for possible values.
            - Fr (Fracture)           -- fracture after advancing time step.
        """
        log = logging.getLogger('PyFrac.controller.advance_time_step')
        # loop for reattempting time stepping in case of failure.
        for i in range(0, self.sim_prop.maxReattempts):
            # smaller time step to reattempt time stepping; equal to the given time step on first iteration
            tmStp_to_attempt = timeStep * self.sim_prop.reAttemptFactor ** i

            # try larger prefactor
            if i > self.sim_prop.maxReattempts/2-1:
                tmStp_to_attempt = timeStep * (1/self.sim_prop.reAttemptFactor)**(i+1 - self.sim_prop.maxReattempts/2)

            # check for final time
            if Frac.time + tmStp_to_attempt > 1.01 * self.sim_prop.finalTime:
                log.info(repr(Frac.time + tmStp_to_attempt))
                return status, Fr
            print('\n')
            log.info('Evaluating solution at time = ' + repr(Frac.time+tmStp_to_attempt) + " ...")
            log.debug("Attempting time step of " + repr(tmStp_to_attempt) + " sec...")

            perfNode_TmStpAtmpt = instrument_start('time step attempt', perfNode)

            self.attmptedTimeStep = tmStp_to_attempt
            status, Fr = attempt_time_step(Frac,
                                            C,
                                            self.solid_prop,
                                            self.fluid_prop,
                                            self.sim_prop,
                                            self.injection_prop,
                                            tmStp_to_attempt,
                                            perfNode_TmStpAtmpt)

            if perfNode_TmStpAtmpt is not None:
                instrument_close(perfNode, perfNode_TmStpAtmpt,
                                 None, len(Frac.EltCrack), status == 1,
                                 self.errorMessages[status], Frac.time)
                perfNode.attempts_data.append(perfNode_TmStpAtmpt)

            if status in [1, 12, 14, 16, 17]:
                break
            else:
                log.warning(self.errorMessages[status])
                log.warning("Time step failed...")


        return status, Fr

#-----------------------------------------------------------------------------------------------------------------------

    def output(self, Fr_advanced):
        """
        This function plot the fracture footprint and/or save file to disk according to the parameters set in the
        simulation properties. See documentation of SimulationProperties class to get the details of parameters which
        determines when and how the output is made.

        Arguments:
            Fr_advanced (Fracture object):       -- fracture after time step is advanced.

        """
        log = logging.getLogger('Pyfrac.output')
        in_req_TSrs = False
        # current time in the time series given at which the solution is to be evaluated
        if self.sim_prop.get_solTimeSeries() is not None and  self.sim_prop.plotATsolTimeSeries :
            if Fr_advanced.time in self.sim_prop.get_solTimeSeries():
                in_req_TSrs = True

        # if the time is final time
        if Fr_advanced.time >= self.sim_prop.finalTime:
            in_req_TSrs = True

        if self.sim_prop.saveToDisk:

            save_TP_exceeded = False
            save_TS_exceeded = False

            # check if save time period is exceeded since last save
            if self.sim_prop.saveTimePeriod is not None:
                if Fr_advanced.time >= self.lastSavedTime + self.sim_prop.saveTimePeriod:
                    save_TP_exceeded = True

            # check if the number of time steps since last save exceeded
            if self.sim_prop.saveTSJump is not None:
                if self.successfulTimeSteps % self.sim_prop.saveTSJump == 0:
                    save_TS_exceeded = True

            if save_TP_exceeded or in_req_TSrs or save_TS_exceeded:

                # save fracture to disk
                log.info("Saving solution at " + repr(Fr_advanced.time) + "...")
                Fr_advanced.SaveFracture(self.sim_prop.get_outputFolder() +
                                         self.sim_prop.get_simulation_name() +
                                         '_file_' + repr(self.lastSavedFile))
                self.lastSavedFile += 1
                log.info("Done! ")

                self.lastSavedTime = Fr_advanced.time

        # plot fracture variables
        if self.sim_prop.plotFigure:

            plot_TP_exceeded = False
            plot_TS_exceeded = False

            # check if plot time period is exceeded since last plot
            if self.sim_prop.plotTimePeriod is not None:
                if Fr_advanced.time >= self.lastPlotTime + self.sim_prop.plotTimePeriod:
                    plot_TP_exceeded = True

            # check if the number of time steps since last plot exceeded
            if self.sim_prop.plotTSJump is not None:
                if self.successfulTimeSteps % self.sim_prop.plotTSJump == 0:
                    plot_TS_exceeded = True

            if plot_TP_exceeded or in_req_TSrs or plot_TS_exceeded:

                for index, plt_var in enumerate(self.sim_prop.plotVar):
                    log.info("Plotting solution at " + repr(Fr_advanced.time) + "...")
                    plot_prop = PlotProperties()

                    if self.Figures[index]:
                        axes = self.Figures[index].get_axes()   # save axes from last figure
                        plt.figure(self.Figures[index].number)
                        plt.clf()                              # clear figure
                        self.Figures[index].add_axes(axes[0])   # add axis to the figure

                    if plt_var == 'footprint':
                        # footprint is plotted if variable to plot is not given
                        plot_prop.lineColor = 'b'
                        if self.sim_prop.plotAnalytical:
                            self.Figures[index] = plot_footprint_analytical(self.sim_prop.analyticalSol,
                                                                       self.solid_prop,
                                                                       self.injection_prop,
                                                                       self.fluid_prop,
                                                                       [Fr_advanced.time],
                                                                       fig=self.Figures[index],
                                                                       h=self.sim_prop.height,
                                                                       samp_cell=None,
                                                                       plot_prop=plot_prop,
                                                                       gamma=self.sim_prop.aspectRatio,
                                                                       inj_point=self.injection_prop.sourceCoordinates)

                        self.Figures[index] = Fr_advanced.plot_fracture(variable='mesh',
                                                                       mat_properties=self.solid_prop,
                                                                       projection='2D',
                                                                       backGround_param=self.sim_prop.bckColor,
                                                                       fig=self.Figures[index],
                                                                       plot_prop=plot_prop)

                        plot_prop.lineColor = 'k'
                        self.Figures[index] = Fr_advanced.plot_fracture(variable='footprint',
                                                                       projection='2D',
                                                                       fig=self.Figures[index],
                                                                       plot_prop=plot_prop)
                        # plotting source elements
                        self.Figures[index] = plot_injection_source(self.fracture,
                                              fig=self.Figures[index])
                    elif plt_var in ('fluid velocity as vector field','fvvf','fluid flux as vector field','ffvf'):
                        if self.fluid_prop.viscosity == 0. :
                            raise SystemExit('ERROR: if the fluid viscosity is equal to 0 does not make sense to ask a plot of the fluid velocity or fluid flux')
                        elif self.sim_prop._SimulationProperties__tipAsymptote == 'K':
                            raise SystemExit('ERROR: if tipAsymptote == K, does not make sense to ask a plot of the fluid velocity or fluid flux')
                        self.Figures[index] = Fr_advanced.plot_fracture(variable='mesh',
                                                                       mat_properties=self.solid_prop,
                                                                       projection='2D',
                                                                       backGround_param=self.sim_prop.bckColor,
                                                                       fig=self.Figures[index],
                                                                       plot_prop=plot_prop)

                        plot_prop.lineColor = 'k'
                        self.Figures[index] = Fr_advanced.plot_fracture(variable='footprint',
                                                                       projection='2D',
                                                                       fig=self.Figures[index],
                                                                       plot_prop=plot_prop)

                        self.Figures[index] = Fr_advanced.plot_fracture(variable=plt_var,
                                                                       projection='2D_vectorfield',
                                                                       mat_properties=self.solid_prop,
                                                                       fig=self.Figures[index])
                    else:
                        if self.sim_prop.plotAnalytical:
                            proj = supported_projections[plt_var][0]
                            self.Figures[index] = plot_analytical_solution(regime=self.sim_prop.analyticalSol,
                                                                      variable=plt_var,
                                                                      mat_prop=self.solid_prop,
                                                                      inj_prop=self.injection_prop,
                                                                      fluid_prop=self.fluid_prop,
                                                                      projection=proj,
                                                                      time_srs=[Fr_advanced.time],
                                                                      h=self.sim_prop.height,
                                                                      gamma=self.sim_prop.aspectRatio)

                        fig_labels = LabelProperties(plt_var, 'whole mesh', '2D')
                        fig_labels.figLabel = ''
                        self.Figures[index] = Fr_advanced.plot_fracture(variable='footprint',
                                                                       projection='2D',
                                                                       fig=self.Figures[index],
                                                                       labels=fig_labels)

                        self.Figures[index] = Fr_advanced.plot_fracture(variable=plt_var,
                                                                       projection='2D_clrmap',
                                                                       mat_properties=self.solid_prop,
                                                                       fig=self.Figures[index],
                                                                       elements=get_elements(suitable_elements[plt_var], Fr_advanced))
                        # plotting source elements
                        self.Figures[index] = plot_injection_source(self.fracture,
                                              fig=self.Figures[index])

                    # plotting closed cells
                    if len(Fr_advanced.closed) > 0:
                        plot_prop.lineColor = 'orangered'
                        self.Figures[index] = Fr_advanced.mesh.identify_elements(Fr_advanced.closed,
                                                                                fig=self.Figures[index],
                                                                                plot_prop=plot_prop,
                                                                                plot_mesh=False,
                                                                                print_number=False)
                    plt.ion()
                    plt.pause(0.4)
                    
                # set figure position
                if self.setFigPos:
                    for i in range(len(self.sim_prop.plotVar)):
                        plt.figure(i + 1)
                        mngr = plt.get_current_fig_manager()
                        x_offset = 650 * i
                        y_ofset = 50
                        if i >= 3:
                            x_offset = (i - 3) * 650
                            y_ofset = 500
                        try:
                            mngr.window.setGeometry(x_offset, y_ofset, 640, 545)
                        except AttributeError:
                            pass
                    self.setFigPos = False

                # plot the figure
                log.info("Done! ")
                if self.sim_prop.blockFigure:
                    input("Press any key to continue.")

                self.lastPlotTime = Fr_advanced.time


#------------------------------------------------------------------------------------------------------------------

    def get_time_step(self):
        """
        This function calculates the appropriate time step. It takes minimum of the time steps evaluated according to
        the following:

            - time step evaluated with the current front velocity to limit the increase in length compared to a cell \
                length
            - time step evaluated with the current front velocity to limit the increase in length compared to the \
                current fracture length
            - time step evaluated with the injection rate in the coming time step
            - time step evaluated to limit the change in total volume of the fracture
        In addition, the limit on the time step and the times at which the solution is required are also taken in
        account to get the appropriate time step.

        Returns:
            - time_step (float)   -- the appropriate time step.

        """
        log = logging.getLogger('PyFrac.get_time_step')
        time_step_given = False
        if self.sim_prop.fixedTmStp is not None:
            # fixed time step
            if isinstance(self.sim_prop.fixedTmStp, float) or isinstance(self.sim_prop.fixedTmStp, int):
                time_step = self.sim_prop.fixedTmStp
                time_step_given = True
            elif isinstance(self.sim_prop.fixedTmStp, np.ndarray) and self.sim_prop.fixedTmStp.shape[0] == 2:
                # fixed time step list is given
                times_past = np.where(self.fracture.time >= self.sim_prop.fixedTmStp[0, :])[0]
                if len(times_past) > 0:
                    indxCurTime = max(times_past)
                    if self.sim_prop.fixedTmStp[1, indxCurTime] is not None:
                        # time step is not given as None.
                        time_step = self.sim_prop.fixedTmStp[1, indxCurTime]  # current injection rate
                        time_step_given = True
                    else:
                        time_step_given = False
                else:
                    # time step is given as None. In this case time step will be evaluated with current state
                    time_step_given = False
            else:
                raise ValueError("Fixed time step can be a float or an ndarray with two rows giving the time and"
                                 " corresponding time steps.")

        if not time_step_given:
            delta_x = min(self.fracture.mesh.hx, self.fracture.mesh.hy)
            if np.any(self.fracture.v == np.nan):
                log.warning("you should not get nan velocities")
            non_zero_v = np.where(self.fracture.v > 0)[0]
            # time step is calculated with the current propagation velocity
            if len(non_zero_v) > 0:
                if len(self.injection_prop.sourceElem) < 4:
                    # if point source
                    tipVrtxCoord = self.fracture.mesh.VertexCoor[self.fracture.mesh.Connectivity[self.fracture.EltTip,
                                                                                             self.fracture.ZeroVertex]]
                    # the distance of tip from the injection point in each of the tip cell
                    dist_Inj_pnt = ((tipVrtxCoord[:, 0] - self.injection_prop.sourceCoordinates[0]) ** 2 +
                                    (tipVrtxCoord[:, 1] - self.injection_prop.sourceCoordinates[1]) ** 2) ** 0.5 \
                                   + self.fracture.l

                    # the time step evaluated by restricting the fracture to propagate not more than 20 percent of the
                    # current maximum length
                    TS_fracture_length = min(abs(0.2 * dist_Inj_pnt[non_zero_v] / self.fracture.v[non_zero_v]))
                else:
                    TS_fracture_length = np.inf

                # the time step evaluated by restricting the fraction of the cell that would be traversed in the time
                # step. e.g., if the pre-factor is 0.5, the tip in the cell with the largest velocity will progress half
                # of the cell width in either x or y direction depending on which is smaller.
                TS_cell_length = delta_x / np.max(self.fracture.v)

            else:
                TS_cell_length = np.inf
                TS_fracture_length = np.inf

            # index of current time in the time series (first row) of the injection rate array
            indx_cur_time = max(np.where(self.fracture.time >= self.injection_prop.injectionRate[0, :])[0])
            current_rate = self.injection_prop.injectionRate[1, indx_cur_time]  # current injection rate
            if current_rate < 0:
                vel_injection = current_rate / (2 * (self.fracture.mesh.hx + self.fracture.mesh.hy) *
                                    self.fracture.w[self.fracture.mesh.CenterElts])
                TS_inj_cell = 10 * delta_x / abs(vel_injection[0])
            elif current_rate > 0:
                # for positive injection, use the increase in total fracture volume criteria
                TS_inj_cell = 0.1 * sum(self.fracture.w) * self.fracture.mesh.EltArea / current_rate
            else:
                TS_inj_cell = np.inf

            TS_delta_vol = np.inf
            if self.delta_w is not None:
                delta_vol = sum(self.delta_w) / sum(self.fracture.w)
                if delta_vol < 0:
                    TS_delta_vol = self.lstTmStp / abs(delta_vol) * 0.05
                else:
                    TS_delta_vol = self.lstTmStp / abs(delta_vol) * 0.12

            # getting pre-factor for current time
            current_prefactor = self.sim_prop.get_time_step_prefactor(self.fracture.time)
            time_step = current_prefactor * min(TS_cell_length,
                                              TS_fracture_length,
                                              TS_inj_cell,
                                              TS_delta_vol)

            # limit time step to be max 2 * last time step
            if (self.lstTmStp != None and not np.isinf(time_step)) and time_step > 2 * self.lstTmStp:
                time_step = 2 * self.lstTmStp

            # limit the time step to be at max 15% of the actual time
            if time_step > 0.15 * self.fracture.time:
                time_step = 0.15 * self.fracture.time

        # in case of fracture not propagating
        if time_step <= 0 or np.isinf(time_step):
            if self.stagnant_TS is not None:
                time_step = self.stagnant_TS
                self.stagnant_TS = time_step * 1.2
            else:
                TS_obtained = False
                log.warning("The fracture front is stagnant and there is no injection. In these conditions, "
                            "there is no criterion to calculate time step size.")
                while not TS_obtained:
                    try:
                        inp = input("Enter the time step size(seconds) you would like to try:")
                        time_step = float(inp)
                        TS_obtained = True
                    except ValueError:
                        pass

        # to get the solution at the times given in time series, any change in parameters or final time
        next_in_TS = self.sim_prop.finalTime

        if self.timeToHit is not None:
            larger_in_TS = np.where(self.timeToHit > self.fracture.time)[0]
            if len(larger_in_TS) > 0:
                next_in_TS = np.min(self.timeToHit[larger_in_TS])

        if next_in_TS < self.fracture.time:
            raise SystemExit('The minimum time required in the given time series or the end time'
                             ' is less than initial time.')

        # check if time step would step over the next time in required time series
        if self.fracture.time + time_step > next_in_TS:
            time_step = next_in_TS - self.fracture.time
        # check if the current time is very close the next time to hit. If yes, set it to the next time to avoid
        # very small time step in the next time step advance.
        elif next_in_TS - self.fracture.time < 1.05 * time_step:
            time_step = next_in_TS - self.fracture.time

        # checking if the time step is above the limit
        if self.sim_prop.timeStepLimit is not None and time_step > self.sim_prop.timeStepLimit:
            log.warning("Evaluated/given time step is more than the time step limit! Limiting time step...")
            time_step = self.sim_prop.timeStepLimit

        return time_step

# ------------------------------------------------------------------------------------------------------------------

    def remesh(self, new_limits, elems, direction=None, rem_factor=10):
        log = logging.getLogger('PyFrac.remesh')
        # Generating the new mesh (with new limits but same number of elements)
        coarse_mesh = CartesianMesh(new_limits[0],
                                    new_limits[1],
                                    elems[0],
                                    elems[1],
                                    symmetric=self.sim_prop.symmetric)

        # Finalizing the transfer of information from the fine to the coarse mesh
        self.solid_prop.remesh(coarse_mesh)
        self.injection_prop.remesh(coarse_mesh, self.fracture.mesh)

        # We adapt the elasticity matrix
        if not self.sim_prop.useBlockToeplizCompression:
            if direction is None:
                #rem_factor = self.sim_prop.remeshFactor
                self.C *= 1 / self.sim_prop.remeshFactor
            elif direction == 'reduce':
                #rem_factor = 10
                if not self.sim_prop.symmetric:
                    self.C = load_isotropic_elasticity_matrix(coarse_mesh, self.solid_prop.Eprime)
                else:
                    self.C = load_isotropic_elasticity_matrix_symmetric(coarse_mesh, self.solid_prop.Eprime)
            else:
                #rem_factor = 10
                log.info("Extending the elasticity matrix...")
                self.extend_isotropic_elasticity_matrix(coarse_mesh, direction=direction)
        else:
            # if direction is None:
            #     rem_factor = self.sim_prop.remeshFactor
            # else:
            #     rem_factor = 10
            self.C.reload(coarse_mesh)

        self.fracture = self.fracture.remesh(rem_factor,
                                             self.C,
                                             coarse_mesh,
                                             self.solid_prop,
                                             self.fluid_prop,
                                             self.injection_prop,
                                             self.sim_prop,
                                             direction)

        self.fracture.mesh = coarse_mesh

        # update the saved properties
        if self.sim_prop.saveToDisk:
            if os.path.exists(self.sim_prop.get_outputFolder() + "properties"):
                os.remove(self.sim_prop.get_outputFolder() + "properties")
            prop = (self.solid_prop, self.fluid_prop, self.injection_prop, self.sim_prop)
            with open(self.sim_prop.get_outputFolder() + "properties", 'wb') as output:
                dill.dump(prop, output, -1)
        self.remeshings += 1

        log.info("Done!")

# -----------------------------------------------------------------------------------------------------------------------

    def extend_isotropic_elasticity_matrix(self, new_mesh, direction=None):
        """
        In the case of extension of the mesh we don't need to recalculate the entire elasticity matrix. All we need to do is
        to map all the elements to their new index and calculate what lasts

        Arguments:
            new_mesh (object CartesianMesh):    -- a mesh object describing the domain.
        """

        a = new_mesh.hx / 2.
        b = new_mesh.hy / 2.
        Ne = new_mesh.NumberOfElts
        Ne_old = self.fracture.mesh.NumberOfElts

        new_indexes = np.array(mapping_old_indexes(new_mesh, self.fracture.mesh, direction))

        if len(self.C) != Ne and not self.sim_prop.symmetric:
            self.C = np.vstack((np.hstack((self.C, np.full((Ne_old, Ne - Ne_old), 0.))),
               np.full((Ne - Ne_old, Ne), 0.)))

            self.C[np.ix_(new_indexes, new_indexes)] = self.C[np.ix_(np.arange(Ne_old), np.arange(Ne_old))]

            add_el = np.setdiff1d(np.arange(Ne), new_indexes)

            for i in add_el:
                x = new_mesh.CenterCoor[i, 0] - new_mesh.CenterCoor[:, 0]
                y = new_mesh.CenterCoor[i, 1] - new_mesh.CenterCoor[:, 1]

                self.C[i] = (self.solid_prop.Eprime / (8. * np.pi)) * (
                        np.sqrt(np.square(a - x) + np.square(b - y)) / ((a - x) * (b - y)) + np.sqrt(
                    np.square(a + x) + np.square(b - y)
                ) / ((a + x) * (b - y)) + np.sqrt(np.square(a - x) + np.square(b + y)) / ((a - x) * (b + y)) + np.sqrt(
                    np.square(a + x) + np.square(b + y)) / ((a + x) * (b + y)))

            self.C[np.ix_(new_indexes, add_el)] = np.transpose(self.C[np.ix_(add_el, new_indexes)])

        elif not self.sim_prop.symmetric:
            self.C = load_isotropic_elasticity_matrix(new_mesh, self.solid_prop.Eprime)
        else:
            self.C = load_isotropic_elasticity_matrix_symmetric(new_mesh, self.solid_prop.Eprime)