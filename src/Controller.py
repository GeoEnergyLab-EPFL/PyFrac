#
# This file is part of PyFrac.
#
# Created by Haseeb Zia on 11.05.17.
# Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2019.  All rights
# reserved. See the LICENSE.TXT file for more details.
#

# local imports
from src.Properties import *
from src.Elasticity import *
from src.HFAnalyticalSolutions import *
from src.TimeStepping import attempt_time_step
from src.Visualization import plot_footprint_analytical, plot_analytical_solution
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
       self.stagnant_TS = None
       self.perfData = []
       self.lastSavedFile = 0
       self.lastSavedTime = -1e99
       self.Figure = None
       self.TmStpCount = 0
       self.chkPntReattmpts = 0

       # Find the times where any parameter changes. These times will be added to the time series where the solution is
       # required to ensure the exact time is hit during time stepping and the change is applied at the correct time.
       param_change_at = np.array([], dtype=np.float64)
       if Injection_prop.injectionRate.shape[1] > 1:
           param_change_at = np.hstack((param_change_at, Injection_prop.injectionRate[0]))
       if isinstance(Sim_prop.fixedTmStp, np.ndarray):
           param_change_at = np.hstack((param_change_at, Sim_prop.fixedTmStp[0]))

       if len(param_change_at) > 0:
            if self.sim_prop.get_solTimeSeries() is not None:
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
                raise ValueError("The final time to stop the simulation is not provided!")
           else:
               self.sim_prop.finalTime = np.max(self.sim_prop.get_solTimeSeries())

       # Setting to volume control solver if viscosity is zero
       if self.fluid_prop.viscosity < 1e-15:
           print("Fluid viscosity is zero. Setting solver to volume control...")
           self.sim_prop.set_volumeControl(True)

       # basic performance data
       self.remeshings = 0
       self.successfulTimeSteps = 0
       self.failedTimeSteps = 0

       self.frontAdvancing = Sim_prop.frontAdvancing

    #-------------------------------------------------------------------------------------------------------------------


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
            self.output(self.fracture)
            self.lastSavedTime = self.fracture.time

        if self.sim_prop.saveToDisk:
            f = open(self.sim_prop.get_outputFolder() + 'log.txt', 'w+')
        else:
            os.remove("log.txt")
            f = open('log.txt', 'w+')
        from time import gmtime, strftime
        f.write('log file, simulation run at: ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '\n\n')
        # f.close()

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
        if not self.sim_prop.symmetric:
            if self.sim_prop.frontAdvancing == "semi-implicit":
                self.sim_prop.frontAdvancing = "implicit"

        print("Starting time = " + repr(self.fracture.time))
        # starting time stepping loop
        while self.fracture.time < 0.999 * self.sim_prop.finalTime and self.TmStpCount < self.sim_prop.maxTimeSteps:

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



            # saving the last five steps to restart if required
            if status == 1:
                print("Time step successful!")

                # output
                if self.sim_prop.plotFigure or self.sim_prop.saveToDisk:
                    if Fr_n_pls1.time > self.lastSavedTime:
                        self.output(Fr_n_pls1)

                # add the advanced fracture to the last five fractures list
                self.fracture = copy.deepcopy(Fr_n_pls1)
                self.fr_queue[self.successfulTimeSteps % 5] = copy.deepcopy(Fr_n_pls1)
                self.chkPntReattmpts = 0
                self.stepsFromChckPnt += 1
                if self.stepsFromChckPnt % 4 == 0:
                    # set the prefactor to the original value after four time steps (after the 5 time steps back jump)
                    self.sim_prop.tmStpPrefactor = self.tmStpPrefactor_max
                self.successfulTimeSteps += 1

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

                    f.writelines("\nRemeshed at " + repr(self.fracture.time))

                else:
                    print("Reached end of the domain. Exiting...")
                    break

            else:
                f.writelines("\n" + self.errorMessages[status])
                f.writelines("\nTime step failed at = " + repr(self.fracture.time))

                self.sim_prop.tmStpPrefactor *= 0.8
                if self.fr_queue[(self.successfulTimeSteps + 1) % 5 ] == None or self.sim_prop.tmStpPrefactor < 0.1:
                    if self.sim_prop.collectPerfData:
                        with open(self.sim_prop.get_outputFolder() + "perf_data.dat", 'wb') as output:
                            dill.dump(self.perfData, output, -1)

                    f.writelines("\n\n---Simulation failed---")
                    f.close()

                    raise SystemExit("Simulation failed.")
                else:
                    self.chkPntReattmpts += 1
                    self.fracture = copy.deepcopy(self.fr_queue[(self.successfulTimeSteps + self.chkPntReattmpts) % 5])
                    print("Restarting from the last check point...")
                    f.writelines("\nRestarting from the last check point...")

                    self.failedTimeSteps += 1

            self.TmStpCount += 1
            # set front advancing beck as set in simulation properties originally
            if self.TmStpCount == 1:
                self.sim_prop.frontAdvancing = self.frontAdvancing

        f.writelines("\n\n-----Simulation finished------")
        f.writelines("\n\nnumber of time steps = " + repr(self.successfulTimeSteps))
        f.writelines("\nfailed time steps = " + repr(self.failedTimeSteps))
        f.writelines("\nnumber of remeshings = " + repr(self.remeshings))
        f.close()

        if self.sim_prop.collectPerfData:
            with open(self.sim_prop.get_outputFolder() + "perf_data.dat", 'wb') as output:
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
            if Frac.time + tmStp_to_attempt > 1.01 * self.sim_prop.finalTime:
                print(repr(Frac.time + tmStp_to_attempt))
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

            if status == 1 or status == 12:
                break
            else:
                if self.sim_prop.verbosity > 1:
                    print(self.errorMessages[status])
                print("Time step failed...")


        return status, Fr

#-----------------------------------------------------------------------------------------------------------------------


    def output(self, Fr_advanced):
        """
        This function plot the fracture footprint and/or save file to disk according to the given time period.

        Arguments:
            Fr_lstTmStp (Fracture object):                      fracture from last time step
            Fr_advanced (Fracture object):                      fracture after time step advancing
            simulation_parameters (SimulationParameters object): simulation parameters
            material_properties (MaterialProperties object):    Material properties

        Returns:
        """

        out_TP_exceeded = False
        in_req_TS = False
        in_plot_every_TS = False

        # check if output time period is exceeded since last output
        if self.sim_prop.outputTimePeriod is not None:
            if Fr_advanced.time >= self.lastSavedTime + self.sim_prop.outputTimePeriod:
                out_TP_exceeded = True

        # current time in the time series given at which the solution is to be evaluated
        if self.sim_prop.get_solTimeSeries() is not None:
            if Fr_advanced.time in self.sim_prop.get_solTimeSeries():
                in_req_TS = True

        # check if the number of time steps since last output exceeded
        if self.sim_prop.outputEveryTS is not None:
            if self.successfulTimeSteps % self.sim_prop.outputEveryTS == 0:
                in_plot_every_TS = True

        if self.sim_prop.outputTimePeriod is None and \
           self.sim_prop.get_solTimeSeries() is None and \
           self.sim_prop.outputEveryTS is None:
            in_plot_every_TS = True

        if out_TP_exceeded or in_req_TS or in_plot_every_TS:
            # output fracture
            self.lastSavedTime = Fr_advanced.time

            # plot fracture footprint
            if self.sim_prop.plotFigure:
                if not self.sim_prop.blockFigure:
                    plt.close()

                print("Plotting solution at " + repr(Fr_advanced.time) + "...")
                plot_prop = PlotProperties()

                if self.sim_prop.plotVar is None or self.sim_prop.plotVar is 'footprint':
                    # footprint is plotted if variable to plot is not given
                    plot_prop.lineColor = 'b'
                    if self.sim_prop.plotAnalytical:
                        self.Figure = plot_footprint_analytical(self.sim_prop.analyticalSol,
                                                          self.solid_prop,
                                                          self.injection_prop,
                                                          self.fluid_prop,
                                                          [Fr_advanced.time],
                                                          h=self.sim_prop.height,
                                                          samp_cell=None,
                                                          plot_prop=plot_prop,
                                                          gamma=self.sim_prop.aspectRatio)
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

                    if len(Fr_advanced.closed) > 0:
                        plot_prop.lineColor = 'r'
                        self.Figure = Fr_advanced.mesh.identify_elements(Fr_advanced.closed,
                                                                         fig=self.Figure,
                                                                         plot_prop=plot_prop,
                                                                         plot_mesh=False)
                else:
                    if self.sim_prop.plotAnalytical:
                        self.Figure = plot_analytical_solution(regime=self.sim_prop.analyticalSol,
                                                              variable=self.sim_prop.plotVar,
                                                              mat_prop=self.solid_prop,
                                                              inj_prop=self.injection_prop,
                                                              fluid_prop=self.fluid_prop,
                                                              time_srs=[Fr_advanced.time],
                                                              h=self.sim_prop.height,
                                                              gamma=self.sim_prop.aspectRatio)
                    else:
                        self.Figure = None

                    fig_labels = LabelProperties(self.sim_prop.plotVar, 'whole mesh', '2D')
                    fig_labels.figLabel = ''
                    self.Figure = Fr_advanced.plot_fracture(variable='footprint',
                                                            projection='2D',
                                                            fig=self.Figure,
                                                            labels=fig_labels)

                    self.Figure = Fr_advanced.plot_fracture(variable=self.sim_prop.plotVar,
                                                            projection=self.sim_prop.plotProj,
                                                            mat_properties=self.solid_prop,
                                                            fig=self.Figure)


                if self.sim_prop.blockFigure:
                    plt.show(block=True)
                else:
                    plt.show(block=False)
                    plt.pause(0.5)
                print("Done! ")

            # save fracture to disk
            if self.sim_prop.saveToDisk:
                print("Saving solution at " + repr(Fr_advanced.time) + "...")
                Fr_advanced.SaveFracture(self.sim_prop.get_outputFolder() +
                                         self.sim_prop.get_simulation_name() +
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

        time_step_given = False
        if self.sim_prop.fixedTmStp is not None:
            # fixed time step
            if isinstance(self.sim_prop.fixedTmStp, float) or isinstance(self.sim_prop.fixedTmStp, int):
                TimeStep = self.sim_prop.fixedTmStp
                time_step_given = True
            elif isinstance(self.sim_prop.fixedTmStp, np.ndarray) and self.sim_prop.fixedTmStp.shape[0]==2:
                # fixed time step list is given
                times_past = np.where(self.fracture.time >= self.sim_prop.fixedTmStp[0, :])[0]
                if len(times_past) > 0:
                    indxCurTime = max(times_past)
                    if self.sim_prop.fixedTmStp[1, indxCurTime] is not None:
                        # time step is not given as None.
                        TimeStep = self.sim_prop.fixedTmStp[1, indxCurTime]  # current injection rate
                        time_step_given = True
                    else:
                        time_step_given = False
                else:
                    # time step is given as None. In this case time step will be evaluated with front velocity
                    time_step_given = False
            else:
                raise ValueError("Fixed time step can be a float or an ndarray with two rows giving the time and"
                                 " corresponding time steps.")

        if not time_step_given:
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

        # in case of fracture not propagating
        if TimeStep <= 0 or np.isinf(TimeStep):
            if self.stagnant_TS is not None:
                TimeStep = self.stagnant_TS
            else:
                raise ValueError("The fracture front seems to be stagnant and no time step is available. Provide a "
                                 "fixed time step at this time.")

        # to get the solution at the times given in time series or final time
        next_in_TS = self.sim_prop.finalTime

        if self.timeToHit is not None:
            larger_in_TS = np.where(self.timeToHit > self.fracture.time)[0]
            if len(larger_in_TS) > 0:
                next_in_TS = np.min(self.timeToHit[larger_in_TS])

        if next_in_TS < self.fracture.time:
            raise SystemExit('The minimum time required in the given time series or the end time'
                             ' is less than initial time.')

        # check if time step would step over the next time in required time series
        if self.fracture.time + TimeStep > next_in_TS:
            TimeStep = next_in_TS - self.fracture.time

        self.stagnant_TS = TimeStep * 1.2

        # checking if the time step is above the limit
        if self.sim_prop.timeStepLimit is not None and TimeStep > self.sim_prop.timeStepLimit:
            TimeStep = self.sim_prop.timeStepLimit

        return TimeStep
