# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on 11.05.17.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2019.
All rights reserved. See the LICENSE.TXT file for more details.
"""

import copy
import matplotlib.pyplot as plt
import dill
import os
import numpy as np
import time
from time import gmtime, strftime

# local imports
from properties import LabelProperties, IterationProperties, PlotProperties
from properties import instrument_start, instrument_close
from elasticity import load_isotropic_elasticity_matrix, load_TI_elasticity_matrix
from mesh import CartesianMesh
from time_stepping import attempt_time_step
from visualization import plot_footprint_analytical, plot_analytical_solution, plot_injection_source
from symmetry import load_isotropic_elasticity_matrix_symmetric, symmetric_elasticity_matrix_from_full
from labels import TS_errorMessages, supported_projections


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
        self.stagnant_TS = None # ??
        self.perfData = []
        self.lastSavedFile = 0
        self.lastSavedTime = np.NINF
        self.lastPlotTime = np.NINF
        self.TmStpCount = 0
        self.chkPntReattmpts = 0 # should be in simulation properties
        self.delta_w = None # ??
        self.lstTmStp = None
        self.solveDetlaP_cp = self.sim_prop.solveDeltaP # ??
        self.PstvInjJmp = None # ??
        self.fullyClosed = False # should be related to the fracture state (thus in fracture class)
        self.setFigPos = True
        self.lastSuccessfulTS = Fracture.time
        self.maxTmStp = 0 # should be in simulation properties


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
           print("Fluid viscosity is zero. Setting solver to volume control...")
           self.sim_prop.set_volumeControl(True)

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

        Sim_prop.frontAdvancing = 'implicit'

        # todo: Remove following block (never executed)
        if Sim_prop.frontAdvancing in ['explicit', 'predictor-corrector']:
            if np.max(Fracture.v) <= 0 or np.isnan(Fracture.v).any():
                Sim_prop.frontAdvancing = 'implicit'
            else:
                Sim_prop.frAdvCurrent = copy.copy(Sim_prop.frontAdvancing)

        if self.sim_prop.saveToDisk:
            self.logAddress = copy.copy(Sim_prop.get_outputFolder())
        else:
            self.logAddress = './'



#-----------------------------------------------------------------------------------------------------------------------

    def run(self):
        """
        This function runs the simulation according to the parameters given in the properties classes. See especially
        the documentation of the :py:class:`properties.SimulationProperties` class to get details of the parameters
        controlling the simulation run.
        """

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

        if self.sim_prop.saveToDisk:
            if os.path.exists(self.logAddress + "log.txt"):
                os.remove(self.logAddress + "log.txt")
            with open(self.logAddress + 'log.txt', 'w+') as file:
                file.writelines('log file, simulation run at: ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '\n\n')

        # load elasticity matrix
        if self.C is None:
            print("Making elasticity matrix...")
            if self.sim_prop.symmetric:
                if not self.sim_prop.get_volumeControl():
                    raise ValueError("Symmetric fracture is only supported for inviscid fluid yet!")

            if not self.solid_prop.TI_elasticity:
                if self.sim_prop.symmetric:
                    self.C = load_isotropic_elasticity_matrix_symmetric(self.fracture.mesh,
                                                                        self.solid_prop.Eprime)
                else:
                    self.C = load_isotropic_elasticity_matrix(self.fracture.mesh,
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

            print('Done!')

        # # perform first time step with implicit front advancing due to non-availability of velocity
        # if not self.sim_prop.symmetric:
        #     if self.sim_prop.frontAdvancing == "predictor-corrector":
        #         self.sim_prop.frontAdvancing = "implicit"

        print("Starting time = " + repr(self.fracture.time))
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
                print("Time step successful!")
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
                    print("Max time steps reached!")

            elif status == 12:
                # re-meshing required
                if self.sim_prop.enableRemeshing:
                    self.C *= 1 / self.sim_prop.remeshFactor
                    print("Remeshing...")
                    coarse_mesh = CartesianMesh(self.sim_prop.remeshFactor * self.fracture.mesh.Lx,
                                                self.sim_prop.remeshFactor * self.fracture.mesh.Ly,
                                                self.fracture.mesh.nx,
                                                self.fracture.mesh.ny,
                                                symmetric=self.sim_prop.symmetric)
                    self.solid_prop.remesh(coarse_mesh)
                    self.injection_prop.remesh(coarse_mesh, self.fracture.mesh)
                    self.fracture = self.fracture.remesh(self.sim_prop.remeshFactor,
                                    self.C,
                                    coarse_mesh,
                                    self.solid_prop,
                                    self.fluid_prop,
                                    self.injection_prop,
                                    self.sim_prop)
                    # update the saved properties
                    if self.sim_prop.saveToDisk:
                        prop = (self.solid_prop, self.fluid_prop, self.injection_prop, self.sim_prop)
                        with open(self.sim_prop.get_outputFolder() + "properties", 'wb') as output:
                            dill.dump(prop, output, -1)
                    self.remeshings += 1
                    print("Done!")

                    self.write_to_log("\nRemeshed at " + repr(self.fracture.time))

                else:
                    print("Reached end of the domain. Exiting...")
                    break

            elif status == 14:
                # fracture fully closed
                self.output(Fr_n_pls1)
                if self.PstvInjJmp is None:
                    inp = input("Fracture is fully closed.\n\nDo you want to jump to"
                            " the time of next positive injection? [y/n]")
                    while inp not in ['y', 'Y', 'n', 'N']:
                        inp = input("Press y or n")

                    if inp is 'y' or inp is 'Y':
                        self.PstvInjJmp = True
                    else:
                        self.PstvInjJmp = False

                if self.PstvInjJmp:
                    self.sim_prop.solveDeltaP = False
                    # index of current time in the time series (first row) of the injection rate array
                    time_larger = np.where(Fr_n_pls1.time <= self.injection_prop.injectionRate[0, :])[0]
                    pos_inj = np.where(self.injection_prop.injectionRate[1, :] > 0)[0]
                    after_time = np.intersect1d(time_larger, pos_inj)
                    if len(after_time) == 0:
                        print("Positive injection not found!")
                        break
                    jump_to = min(self.injection_prop.injectionRate[0, np.intersect1d(time_larger, pos_inj)])
                    Fr_n_pls1.time = jump_to
                elif inp is 'n' or inp is 'N':
                    self.sim_prop.solveDeltaP = True
                self.fullyClosed = True
                self.fracture = copy.deepcopy(Fr_n_pls1)

            else:
                # time step failed
                self.write_to_log("\n" + self.errorMessages[status])
                self.write_to_log("\nTime step failed at = " + repr(self.fracture.time))
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

                    self.write_to_log("\n\n---Simulation failed---")

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
                    print("Time step have failed despite of reattempts with slightly smaller/bigger time steps...\n"
                          "Going " + repr(5 - self.chkPntReattmpts) + " time steps back and re-attempting with the"
                            " time step pre-factor of " + repr(current_PreFctr))
                    self.write_to_log("\nTime step have failed. Going " + repr(6 - self.chkPntReattmpts) + " time steps"
                                                                                                      " back...\n")
                    self.failedTimeSteps += 1

            self.TmStpCount += 1

        self.write_to_log("\n\n-----Simulation finished------")
        self.write_to_log("\n\nnumber of time steps = " + repr(self.successfulTimeSteps))
        self.write_to_log("\nfailed time steps = " + repr(self.failedTimeSteps))
        self.write_to_log("\nnumber of remeshings = " + repr(self.remeshings))

        plt.show(block=False)
        plt.close('all')

        if self.sim_prop.collectPerfData:
            file_address = self.sim_prop.get_outputFolder() + "perf_data.dat"
            os.makedirs(os.path.dirname(file_address), exist_ok=True)
            with open(file_address, 'wb') as output:
                dill.dump(self.perfData, output, -1)

        print("\nFinal time = " + repr(self.fracture.time))
        print("\n\n-----Simulation finished------")
        print("See log file for details\n\n")


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

        # loop for reattempting time stepping in case of failure.
        for i in range(0, self.sim_prop.maxReattempts):
            # smaller time step to reattempt time stepping; equal to the given time step on first iteration
            tmStp_to_attempt = timeStep * self.sim_prop.reAttemptFactor ** i

            # try larger prefactor
            if i > self.sim_prop.maxReattempts/2-1:
                tmStp_to_attempt = timeStep * (1/self.sim_prop.reAttemptFactor)**(i+1 - self.sim_prop.maxReattempts/2)

            # check for final time
            if Frac.time + tmStp_to_attempt > 1.01 * self.sim_prop.finalTime:
                print(repr(Frac.time + tmStp_to_attempt))
                return status, Fr

            print('\nEvaluating solution at time = ' + repr(Frac.time+tmStp_to_attempt) + " ...")
            if self.sim_prop.verbosity > 1:
                print("Attempting time step of " + repr(tmStp_to_attempt) + " sec...")

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

            if status in [1, 12, 14]:
                break
            else:
                if self.sim_prop.verbosity > 1:
                    print(self.errorMessages[status])
                print("Time step failed...")


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

        in_req_TSrs = False
        # current time in the time series given at which the solution is to be evaluated
        if self.sim_prop.get_solTimeSeries() is not None:
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
                print("Saving solution at " + repr(Fr_advanced.time) + "...")
                Fr_advanced.SaveFracture(self.sim_prop.get_outputFolder() +
                                         self.sim_prop.get_simulation_name() +
                                         '_file_' + repr(self.lastSavedFile))
                self.lastSavedFile += 1
                print("Done! ")

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
                    print("Plotting solution at " + repr(Fr_advanced.time) + "...")
                    plot_prop = PlotProperties()

                    if self.Figures[index]:
                        axes = self.Figures[index].get_axes()   # save axes from last figure
                        plt.figure(self.Figures[index].number)
                        plt.clf()                              # clear figure
                        self.Figures[index].add_axes(axes[0])   # add axis to the figure

                    if plt_var is 'footprint':
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
                                                                       fig=self.Figures[index])
                    # plotting source elements
                    plot_injection_source(self.injection_prop,
                                          self.fracture.mesh,
                                          fig=self.Figures[index])

                    # plotting closed cells
                    if len(Fr_advanced.closed) > 0:
                        plot_prop.lineColor = 'orangered'
                        self.Figures[index] = Fr_advanced.mesh.identify_elements(Fr_advanced.closed,
                                                                                fig=self.Figures[index],
                                                                                plot_prop=plot_prop,
                                                                                plot_mesh=False,
                                                                                print_number=False)
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
                plt.ion()
                plt.pause(0.01)
                print("Done! ")
                if self.sim_prop.blockFigure:
                    input("Press any key to continue.")

                self.lastPlotTime = Fr_advanced.time


    #-------------------------------------------------------------------------------------------------------------------

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

        # in case of fracture not propagating
        if time_step <= 0 or np.isinf(time_step):
            if self.stagnant_TS is not None:
                time_step = self.stagnant_TS
                self.stagnant_TS = time_step * 1.2
            else:
                TS_obtained = False
                print("The fracture front is stagnant and there is no injection. In these conditions, "
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
            print("Evaluated/given time step is more than the time step limit! Limiting time step...")
            time_step = self.sim_prop.timeStepLimit

        return time_step

    def write_to_log(self, line):
        """ This function writes the given line to the log file."""
        with open(self.logAddress + 'log.txt', 'a+') as file:
            file.writelines(line)
