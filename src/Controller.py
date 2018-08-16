#
# This file is part of PyFrac.
#
# Created by Haseeb Zia on 11.05.17.
# Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2017.  All rights
# reserved. See the LICENSE.TXT file for more details.
#

# local imports
from src.Properties import *
from src.Elasticity import *
from src.HFAnalyticalSolutions import *
from src.TimeStepping import attempt_time_step
from src.Visualization import plot_footprint_analytical
from src.Symmetry import load_isotropic_elasticity_matrix_symmetric

import copy
import matplotlib.pyplot as plt
import dill
import os


class Controller:
    """

    """
    errorMessages = ("Propagation not attempted!",
                     "Time step successful!",
                     "Evaluated level set is not valid!",
                     "Front is not tracked correctly!",
                     "Evaluated tip volume is not valid!",
                     "Solution obtained from the elastohydrodynamic solver is not valid!",
                     "Did not converge after max iterations!",
                     "Tip inversion is not correct!",
                     "Ribbon element not found in the enclosure of the tip cell!",
                     "Filling fraction not correct!",
                     "Toughness iteration did not converge!",
                     "projection could not be found!",
                     "Reached end of grid!",
                     "Leak off can't be evaluated!"
                     )


    def __init__(self, Fracture, Solid_prop, Fluid_prop, Injection_prop, Sim_prop, Load_prop=None, C=None):

       self.fracture = Fracture
       self.solid_prop = Solid_prop
       self.fluid_prop = Fluid_prop
       self.injection_prop = Injection_prop
       self.sim_prop = Sim_prop
       self.load_prop = Load_prop
       self.C = C
       self.fr_queue = [None, None, None, None, None] # queue of fractures from the last five time steps
       self.stepsFromChckPnt = 0
       self.tmStpPrefactor_max = Sim_prop.tmStpPrefactor
       self.perfData = []
       self.lastSavedFile = 0
       self.Figure = None

       self.tmSrsIndex = 0
       # the next time where the solution is required to be evaluated
       if (self.sim_prop).get_solTimeSeries() is not None:
           self.nextInTmSrs = (self.sim_prop).get_solTimeSeries()[self.tmSrsIndex]
       else:
           # set final time as the time where the solution is required
           self.nextInTmSrs = self.sim_prop.FinalTime

       # basic performance data
       self.remeshings = 0
       self.TotalTimeSteps = 0
       self.failedTimeSteps = 0
       self.frontAdvancing = Sim_prop.frontAdvancing

    def run(self):
        """
        This function runs the simulation according to the given parameters.
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
            self.output(self.fracture,
                        self.sim_prop,
                        self.solid_prop,
                        self.injection_prop,
                        self.fluid_prop)

        if self.sim_prop.saveToDisk:
            f = open(self.sim_prop.get_outputFolder() + 'log', 'w+')
        else:
            f = open('log', 'w+')
        from time import gmtime, strftime
        f.write('log file, program run at: ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '\n\n\n')
        f.close()

        # load elasticity matrix
        if self.C is None:
            print("Making elasticity matrix...")
            if self.sim_prop.symmetric:
                if self.solid_prop.TI_elasticity or self.solid_prop.freeSurf:
                    raise ValueError("Symmetric fracture for TI material is not yet supported")
                else:
                    self.C = load_isotropic_elasticity_matrix_symmetric(self.fracture.mesh,
                                                                        self.solid_prop.Eprime)
            else:
                if self.solid_prop.TI_elasticity or self.solid_prop.freeSurf:
                    self.C = load_TI_elasticity_matrix(self.fracture.mesh,
                                                       self.solid_prop,
                                                       self.sim_prop)
                else:
                    self.C = load_isotropic_elasticity_matrix(self.fracture.mesh,
                                                              self.solid_prop.Eprime)
            print('Done!')

        # perform first time step with implicit front advancing
        # if self.sim_prop.frontAdvancing == "semi-implicit":
        #     self.sim_prop.frontAdvancing = "implicit"

        i = 0
        print("Starting time = " + repr(self.fracture.time))
        # starting time stepping loop
        while self.fracture.time < 0.999 * self.sim_prop.FinalTime and i < self.sim_prop.maxTimeSteps:

            TimeStep = self.get_time_step()


            if self.sim_prop.collectPerfData:
                tmStp_perf = IterationProperties(itr_type="time step")
            else:
                tmStp_perf = None

            # advancing time step
            status, Fr_n_pls1 = self.advance_time_step(self.fracture,
                                                 self.C,
                                                 TimeStep,
                                                 tmStp_perf)

            if self.sim_prop.collectPerfData:
                tmStp_perf.CpuTime_end = time.time()
                if status == 1 :
                    tmStp_perf.status = 'successful'
                else:
                    tmStp_perf.status = 'failed'
                self.perfData.append(tmStp_perf)

            self.TotalTimeSteps += 1

            # saving the last five steps to restart if required
            if status == 1:
                print("Time step successful!")
                self.fracture = copy.deepcopy(Fr_n_pls1)
                self.fr_queue[i%5] = copy.deepcopy(Fr_n_pls1)
                self.stepsFromChckPnt += 1
                if self.stepsFromChckPnt%4 == 0:
                    # set the prefactor to the original value after four time steps (after the 5 time steps back jump)
                    self.sim_prop.tmStpPrefactor = self.tmStpPrefactor_max

            # re-meshing required
            elif status == 12:
                if self.sim_prop.enableRemeshing:
                    self.C *= 1 / self.sim_prop.remeshFactor
                    print("Remeshing...")
                    self.fracture = self.fracture.remesh(self.sim_prop.remeshFactor,
                                    self.C,
                                    self.solid_prop,
                                    self.fluid_prop,
                                    self.injection_prop,
                                    self.sim_prop)
                    self.remeshings += 1
                    print("Done!")

                    # saving to log file
                    f = open('log', 'a')
                    f.writelines("\nRemeshed at " + repr(self.fracture.time))
                    f.close()
                else:
                    print("Reached end of the domain. Exiting...")
                    break

            else:
                self.sim_prop.tmStpPrefactor *= 0.8
                self.stepsFromChckPnt = 0
                if self.fr_queue[(i+1) % 5 ] == None or self.sim_prop.tmStpPrefactor < 0.1:
                    raise SystemExit("Simulation failed.")
                else:
                    self.fracture = copy.deepcopy(self.fr_queue[(i+1) % 5])
                    print("Restarting with the last check point...")

                    self.failedTimeSteps += 1
                    self.TotalTimeSteps += 1

                    f = open('log', 'a')
                    f.writelines("\n" + self.errorMessages[status])
                    f.writelines("\nTime step failed at = " + repr(self.fr_queue[i % 5].time))
                    f.close()

            i = i + 1
            # set front advancing beck as set in simulation properties originally
            self.sim_prop.frontAdvancing = self.frontAdvancing

        f = open('log', 'a')
        f.writelines("\n\nnumber of time steps = " + repr(self.TotalTimeSteps))
        f.writelines("\nfailed time steps = " + repr(self.failedTimeSteps))
        f.writelines("\nnumber of remeshings = " + repr(self.remeshings))
        f.close()

        if self.sim_prop.collectPerfData:
            with open(self.sim_prop.get_outFileAddress() + "perf_data.dat", 'wb') as output:
                dill.dump(self.perfData, output, -1)

        print("\nFinal time = " + repr(self.fracture.time))
        print("\n\n-----Simulation finished------")
        print("See log file for details\n\n")

#-----------------------------------------------------------------------------------------------------------------------


    def advance_time_step(self, Frac, C, TimeStep, PerfNode=None):
        """
        This function advances the fracture by the given time step. In case of failure, reattempts are made with smaller
        time steps. A system exit is raised after maximum allowed reattempts.

        Arguments:
            Frac (Fracture object):         -- fracture object from the last time step
            C (ndarray-float):              -- the elasticity matrix
            TimeStep (float):               -- time step to be attempted
            PerfNode (IterationProperties)  -- An IterationProperties instance to store performance data

        Return:
                    exitstatus (int)        -- see documentation for possible values.
                    Fr (Fracture)           -- fracture after advancing time step.
        """

        # loop for reattempting time stepping in case of failure.
        for i in range(0, self.sim_prop.maxReattempts):
            # smaller time step to reattempt time stepping; equal to the given time step on first iteration
            tmStp_to_attempt = TimeStep * self.sim_prop.reAttemptFactor ** i

            # try larger prefactor
            if i > self.sim_prop.maxReattempts/2-1:
                tmStp_to_attempt = TimeStep * (1/self.sim_prop.reAttemptFactor)**(i+1 - self.sim_prop.maxReattempts/2)

            # check for final time
            if Frac.time + tmStp_to_attempt > 1.01 * self.sim_prop.FinalTime:
                return status, Fr

            print('\nEvaluating solution at time = ' + repr(Frac.time+tmStp_to_attempt) + " ...")
            if self.sim_prop.verbosity > 1:
                print("Attempting time step of " + repr(tmStp_to_attempt) + " sec...")

            if PerfNode is not None:
                PerfNode_TmStpAtmpt = IterationProperties(itr_type="time step attempt")
                PerfNode_TmStpAtmpt.subIterations = [[], [], []]
            else:
                PerfNode_TmStpAtmpt = None


            status, Fr = attempt_time_step(Frac,
                                            C,
                                            self.solid_prop,
                                            self.fluid_prop,
                                            self.sim_prop,
                                            self.injection_prop,
                                            tmStp_to_attempt,
                                            PerfNode_TmStpAtmpt)



            if PerfNode_TmStpAtmpt is not None:
                PerfNode_TmStpAtmpt.CpuTime_end = time.time()
                PerfNode.iterations += 1
                PerfNode.subIterations.append(PerfNode_TmStpAtmpt)
                if status == 1:
                    PerfNode_TmStpAtmpt.status = 'successful'
                else:
                    PerfNode_TmStpAtmpt.status = 'failed'

            if status == 1:
                if self.sim_prop.verbosity > 1:
                    print(self.errorMessages[status])

                # output
                if self.sim_prop.plotFigure or self.sim_prop.saveToDisk:
                    self.output(Fr,
                                self.sim_prop,
                                self.solid_prop,
                                self.injection_prop,
                                self.fluid_prop,
                                Frac)

                return status, Fr
            else:

                if status == 12:
                    return status, Fr
            if self.sim_prop.verbosity > 1:
                print(self.errorMessages[status])
                print("Time step failed...")

        return status, Fr

#-----------------------------------------------------------------------------------------------------------------------


    def output(self, Fr_advanced, simulation_parameters, material_properties, injection_parameters,
               fluid_properties, Fr_lstTmStp=None):
        """
        This function plot the fracture footprint and/or save file to disk according to the given time period.

        Arguments:
            Fr_lstTmStp (Fracture object):                      fracture from last time step
            Fr_advanced (Fracture object):                      fracture after time step advancing
            simulation_parameters (SimulationParameters object): simulation parameters
            material_properties (MaterialProperties object):    Material properties

        Returns:
        """

        if Fr_lstTmStp is not None:
            # output time period exceeded after this time step
            out_TP_exceeded = not simulation_parameters.outputTimePeriod is None and (Fr_lstTmStp.time //
                simulation_parameters.outputTimePeriod != Fr_advanced.time // simulation_parameters.outputTimePeriod)
        else:
            out_TP_exceeded = True

        # current time in the time series given at which the solution is to be evaluated
        in_req_TS = (simulation_parameters.get_solTimeSeries() is not None) and \
                    Fr_advanced.time in simulation_parameters.get_solTimeSeries()

        if out_TP_exceeded or in_req_TS:
            # plot fracture footprint
            if simulation_parameters.plotFigure:
                if not simulation_parameters.blockFigure:
                    plt.close()

                print("Plotting solution at " + repr(Fr_advanced.time) + "...")
                plot_prop = PlotProperties()

                plot_prop.lineColor = 'r'
                if simulation_parameters.plotAnalytical:
                    self.Figure = plot_footprint_analytical(simulation_parameters.analyticalSol,
                                                      self.solid_prop,
                                                      self.injection_prop,
                                                      self.fluid_prop,
                                                      [Fr_advanced.time],
                                                      h=simulation_parameters.height,
                                                      samp_cell=None,
                                                      plot_prop=plot_prop,
                                                      gamma=simulation_parameters.aspectRatio)
                else:
                    self.Figure = None

                self.Figure = Fr_advanced.plot_fracture(variable='mesh',
                                                        mat_properties=self.solid_prop,
                                                        projection='2D',
                                                        backGround_param=self.sim_prop.bckColor,
                                                        fig=self.Figure,
                                                        plot_prop=plot_prop)

                plot_prop.lineColor = 'k'
                self.Figure = Fr_advanced.plot_fracture(variable='footprint',
                                                        projection='2D',
                                                        fig=self.Figure,
                                                        plot_prop=plot_prop)
                if simulation_parameters.blockFigure:
                    plt.figure(1)
                    plt.show()
                else:
                    plt.show(block=False)
                    plt.pause(0.5)
                print("Done! ")

            # save fracture to disk
            if simulation_parameters.saveToDisk:
                print("Saving solution at " + repr(Fr_advanced.time) + "...")
                Fr_advanced.SaveFracture(simulation_parameters.get_outputFolder() +
                                         simulation_parameters.get_simulation_name() +
                                         '_file_' + repr(self.lastSavedFile))
                self.lastSavedFile += 1
                print("Done! ")


#-----------------------------------------------------------------------------------------------------------------------

    def get_time_step(self):
        """
        This function calculate the appropriate time step.

        Arguments:
            Frac (Fracture)     -- fracture from the last time step
            pre_factor (float)  -- the pre-factor to be multiplied to the time step evaluated with the maximum propagation
                                   velocity from the last time step.

        Returns:
            time_step (float)   -- the appropriate time step
        """

        if self.nextInTmSrs < self.fracture.time:
            raise SystemExit('The minimum time required in the given time series or the end time'
                             ' is less than initial time.')

        if self.sim_prop.fixedTmStp is not None:
            # fixed time step
            TimeStep = self.sim_prop.fixedTmStp

        else:
            # time step is calculated with the current propagation velocity
            tipVrtxCoord = self.fracture.mesh.VertexCoor[self.fracture.mesh.Connectivity[self.fracture.EltTip,
                                                                                         self.fracture.ZeroVertex]]
            # the distance of tip from the injection point in each of the tip cell
            dist_Inj_pnt = (tipVrtxCoord[:, 0] ** 2 + tipVrtxCoord[:, 1] ** 2) ** 0.5 + self.fracture.l
            # the time step evaluated by restricting the fracture to propagate not more than 8 percent of the current
            # maximum length
            TS_cell_length = min(abs(0.08 * self.sim_prop.tmStpPrefactor * dist_Inj_pnt / self.fracture.v))

            # the time step evaluated by restricting the fraction of the cell that would be traversed in the time step.
            # e.g., if the prefactor is 0.5, the tip in the cell with the largest velocity will progress half of the
            # cell width in either x or y direction depending on which is smaller.
            TS_fracture_length = self.sim_prop.tmStpPrefactor * min(self.fracture.mesh.hx, self.fracture.mesh.hy
                                                            ) / np.max(self.fracture.v)

            TimeStep = min(TS_cell_length, TS_fracture_length)


        # to get the solution at the times given in time series or final time
        if self.fracture.time + TimeStep > self.nextInTmSrs:
            TimeStep = self.nextInTmSrs - self.fracture.time
            # set next in time series
            if self.sim_prop.get_solTimeSeries() is not None:
                if self.tmSrsIndex < len(self.sim_prop.get_solTimeSeries()) - 1:
                    self.tmSrsIndex += 1
                self.nextInTmSrs = self.sim_prop.get_solTimeSeries()[self.tmSrsIndex]

        # checking if the time step is above the limit
        if self.sim_prop.timeStepLimit is None:
            self.sim_prop.timeStepLimit = TimeStep
        else:
            if TimeStep > self.sim_prop.timeStepLimit:
                TimeStep = self.sim_prop.timeStepLimit

        # set time step limit according to largest time step (likely to be the most recent one)
        self.sim_prop.timeStepLimit = max(TimeStep * self.sim_prop.tmStpFactLimit, self.sim_prop.timeStepLimit)

        # in case of fracture not propagating
        if TimeStep <= 0:
            TimeStep = self.sim_prop.timeStepLimit * 2
            self.sim_prop.timeStepLimit *= 1.5

        return TimeStep
