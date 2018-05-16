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
from src.TimeSteppingVolumeControl import attempt_time_step_volumeControl
from src.TimeSteppingViscousFluid import attempt_time_step_viscousFluid
from src.TimeSteppingMechLoading import attempt_time_step_mechLoading

import copy
import matplotlib.pyplot as plt
import dill

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


    def __init__(self, Fracture=None, Solid_prop=None, Fluid_prop=None, Injection_prop=None, Sim_prop=None,
                 Load_prop=None, C=None):

       self.fracture = Fracture
       self.solid_prop = Solid_prop
       self.fluid_prop = Fluid_prop
       self.injection_prop = Injection_prop
       self.sim_prop = Sim_prop
       self.load_prop = Load_prop
       self.C = C
       self.fr_queue = [None, None, None, None, None] # queue of fractures from the last five time steps
       self.smallStep_cnt = 0
       self.tmStpPrefactor_max = Sim_prop.tmStpPrefactor
       self.perfData = []
       self.Figure = plt.figure()

       # basic performance data
       self.remeshings = 0
       self.TotalTimeSteps = 0
       self.failedTimeSteps = 0


    def run(self):
        """
        This function runs the simulation according to the given parameters.
        """
        # todo change it to data file?
        f = open('log', 'w+')
        from time import gmtime, strftime
        f.write('log file, program run at: ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '\n\n\n')
        f.close()

        # load elasticity matrix
        if self.C is None:
            self.C = load_elasticity_matrix(self.fracture.mesh, self.solid_prop.Eprime)

        i = 0
        Fr = self.fracture
        tmSrs_indx = 0
        # the next time where the solution is required to be evaluated
        if not (self.sim_prop).get_solTimeSeries() is None:
            next_in_tmSrs = (self.sim_prop).get_solTimeSeries()[tmSrs_indx]
        else:
            next_in_tmSrs = self.sim_prop.FinalTime

        if next_in_tmSrs < Fr.time:
            raise SystemExit('The minimum time required in the given time series or the end time'
                             ' is less than initial time.')

        print("Starting time = " + repr(Fr.time))
        # starting time stepping loop
        while Fr.time < 0.999 * self.sim_prop.FinalTime and i < self.sim_prop.maxTimeSteps:


            if self.sim_prop.fixedTmStp is not None:
                TimeStep = self.sim_prop.fixedTmStp
            else:
                # time step is calculated with the current propagation velocity
                TimeStep = get_time_step(Fr, self.sim_prop.tmStpPrefactor)

            # to get the solution at the times given in time series
            if self.sim_prop.get_solTimeSeries() is not None and Fr.time + TimeStep > next_in_tmSrs:
                TimeStep = next_in_tmSrs - Fr.time
                if tmSrs_indx < len(self.sim_prop.get_solTimeSeries())-1:
                    tmSrs_indx += 1
                next_in_tmSrs = self.sim_prop.get_solTimeSeries()[tmSrs_indx]
            elif Fr.time + TimeStep > next_in_tmSrs:
                TimeStep = next_in_tmSrs - Fr.time

            if self.sim_prop.collectPerfData:
                tmStp_perf = IterationProperties(itr_type="time step")
            else:
                tmStp_perf = None

            # advancing time step
            status, Fr_n_pls1 = self.advance_time_step(Fr,
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
                Fr = copy.deepcopy(Fr_n_pls1)
                self.fr_queue[i%5] = copy.deepcopy(Fr_n_pls1)
                self.smallStep_cnt += 1
                if self.smallStep_cnt%4 == 0:
                    # set the prefactor to the original value after four time steps (after the 5 time steps back jump)
                    self.sim_prop.tmStpPrefactor = self.tmStpPrefactor_max

            # remeshing required
            elif status == 12:
                self.C *= 1/self.sim_prop.remeshFactor
                print("Remeshing...")
                Fr = Fr.remesh(self.sim_prop.remeshFactor,
                               self.C,
                               self.solid_prop,
                               self.fluid_prop,
                               self.injection_prop,
                               self.sim_prop)
                self.remeshings += 1
                print("Done!")

                # saving to log file
                f = open('log', 'a')
                f.writelines("\nRemeshed at " + repr(Fr.time))
                f.close()

            else:
                self.sim_prop.tmStpPrefactor *= 0.8
                self.smallStep_cnt = 0
                if self.fr_queue[(i+1) % 5 ] == None or self.sim_prop.tmStpPrefactor < 0.1:
                    raise SystemExit("Simulation failed.")
                else:
                    Fr = copy.deepcopy(self.fr_queue[(i+1) % 5])
                    print("Restarting with the last check point...")

                    self.failedTimeSteps += 1
                    self.TotalTimeSteps += 1

                    f = open('log', 'a')
                    f.writelines("\n" + self.errorMessages[status])
                    f.writelines("\nTime step failed at = " + repr(self.fr_queue[i % 5].time))
                    f.close()

            i = i + 1

        f = open('log', 'a')
        f.writelines("\n\nnumber of time steps = " + repr(self.TotalTimeSteps))
        f.writelines("\nfailed time steps = " + repr(self.failedTimeSteps))
        f.writelines("\nnumber of remeshings = " + repr(self.remeshings))
        f.close()

        if self.sim_prop.collectPerfData:
            with open(self.sim_prop.get_outFileAddress() + "perf_data.dat", 'wb') as output:
                dill.dump(self.perfData, output, -1)

        print("\nFinal time = " + repr(Fr.time))
        print("\n\n-----Simulation successfully finished------")
        print("See log file for details\n\n")

#-----------------------------------------------------------------------------------------------------------------------


    def advance_time_step(self, Frac, C, TimeStep, PerfNode=None):
        """
        This function advances the fracture by the given time step. In case of failure, reattempts are made with smaller
        time steps. A system exit is raised after maximum allowed reattempts.

        Arguments:
            Frac (Fracture object):                                 fracture object from the last time step
            C (ndarray-float):                                      the elasticity matrix


            TimeStep (float):                                       time step to be attempted

            Loading_Properties (LoadingProperties object)


        Return:
            int:   possible values:
                                        0       -- not propagated
                                        1       -- iteration successful
                                        2       -- evaluated level set is not valid
                                        3       -- front is not tracked correctly
                                        4       -- evaluated tip volume is not valid
                                        5       -- solution of elastohydrodynamic solver is not valid
                                        6       -- did not converge after max iterations
                                        7       -- tip inversion not successful
                                        8       -- Ribbon element not found in the enclosure of a tip cell
                                        9       -- Filling fraction not correct
                                        10      -- Toughness iteration did not converge
                                        11      -- projection could not be found
                                        12      -- Reached end of grid

            Fracture object:            fracture after advancing time step.
        """

        # checking if the time step is above the limit
        if TimeStep > self.sim_prop.timeStepLimit:
            TimeStep = self.sim_prop.timeStepLimit

        self.sim_prop.timeStepLimit = max(TimeStep * self.sim_prop.tmStpFactLimit, self.sim_prop.timeStepLimit)

        # in case of fracture not propagating
        if TimeStep <= 0:
            TimeStep = self.sim_prop.timeStepLimit*2
            self.sim_prop.timeStepLimit *= 1.5

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

            if self.sim_prop.get_viscousInjection():
                status, Fr = attempt_time_step_viscousFluid(Frac,
                                                            C,
                                                            self.solid_prop,
                                                            self.fluid_prop,
                                                            self.sim_prop,
                                                            self.injection_prop,
                                                            tmStp_to_attempt,
                                                            PerfNode_TmStpAtmpt)

            elif self.sim_prop.get_dryCrack_mechLoading():
                status, Fr = attempt_time_step_mechLoading(Frac,
                                                            C,
                                                            self.solid_prop,
                                                            self.sim_prop,
                                                            self.load_prop,
                                                            tmStp_to_attempt,
                                                            Frac.mesh,
                                                            PerfNode_TmStpAtmpt)

            elif self.sim_prop.get_volumeControl():
                status, Fr = attempt_time_step_volumeControl(Frac,
                                                             C,
                                                             self.solid_prop,
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
                    self.output(Frac,
                           Fr,
                           self.sim_prop,
                           self.solid_prop,
                           self.injection_prop,
                           self.fluid_prop)

                return status, Fr
            else:

                if status == 12:
                    return status, Fr
            if self.sim_prop.verbosity > 1:
                print(self.errorMessages[status])
                print("Time step failed...")

        return status, Fr

#-----------------------------------------------------------------------------------------------------------------------


    def output(self, Fr_lstTmStp, Fr_advanced, simulation_parameters, material_properties, injection_parameters,
               fluid_properties):
        """
        This function plot the fracture footprint and/or save file to disk according to the given time period.

        Arguments:
            Fr_lstTmStp (Fracture object):                      fracture from last time step
            Fr_advanced (Fracture object):                      fracture after time step advancing
            simulation_parameters (SimulationParameters object): simulation parameters
            material_properties (MaterialProperties object):    Material properties

        Returns:
        """
        # output time period exceeded after this time step
        out_TP_exceeded = not simulation_parameters.outputTimePeriod is None and (Fr_lstTmStp.time //
            simulation_parameters.outputTimePeriod != Fr_advanced.time // simulation_parameters.outputTimePeriod)

        # current time in the time series given at which the solution is to be evaluated
        in_req_TS = (not simulation_parameters.get_solTimeSeries() is None) and \
                    Fr_advanced.time in simulation_parameters.get_solTimeSeries()

        if out_TP_exceeded or in_req_TS:
            # plot fracture footprint
            if simulation_parameters.plotFigure:
                print("Plotting solution at " + repr(Fr_advanced.time) + "...")
                # if ploting analytical solution enabled
                if simulation_parameters.plotAnalytical:
                    Q0 = injection_parameters.injectionRate[1, 0]  # injection rate at the start of injection
                    if simulation_parameters.analyticalSol in ('M', 'Mt', 'K', 'Kt', 'E'): #radial fracture
                        t, R, p, w, v, actvElts = HF_analytical_sol(simulation_parameters.analyticalSol,
                                                                    Fr_lstTmStp.mesh,
                                                                    material_properties.Eprime,
                                                                    Q0,
                                                                    muPrime=fluid_properties.muPrime,
                                                                    Kprime=material_properties.Kprime[
                                                                        Fr_lstTmStp.mesh.CenterElts],
                                                                    Cprime=material_properties.Cprime[
                                                                        Fr_lstTmStp.mesh.CenterElts],
                                                                    t=Fr_advanced.time,
                                                                    KIc_min=material_properties.K1c_perp)
                    elif simulation_parameters.analyticalSol == 'PKN':
                        print("PKN is to be implemented.")

                    plt.close(self.Figure)
                    self.Figure = Fr_advanced.plot_fracture(analytical=R,
                                                mat_properties=material_properties,
                                                sim_properties=simulation_parameters)
                else:
                    plt.close(self.Figure)
                    self.Figure = Fr_advanced.plot_fracture(mat_properties=material_properties,
                                              # fig=self.Figure,
                                              sim_properties=simulation_parameters)
                plt.show(block=False)
                plt.pause(0.5)
                print("Done! ")

            # save fracture to disk
            if simulation_parameters.saveToDisk:
                print("Saving solution at " + repr(Fr_advanced.time) + "...")
                simulation_parameters.lastSavedFile += 1
                Fr_advanced.SaveFracture(simulation_parameters.get_outFileAddress() + "fracture_"
                                         + repr(simulation_parameters.lastSavedFile))
                print("Done! ")

def get_time_step(Frac, pre_factor):
    """
    This function calculate the appropriate time step.

    Arguments:
        Frac (Fracture)     -- fracture from the last time step
        pre_factor (float)  -- the pre-factor to be multiplied to the time step evaluated with the maximum propagation
                               velocity from the last time step.

    Returns:
        time_step (float)   -- the appropriate time step
    """

    tipVrtxCoord = Frac.mesh.VertexCoor[Frac.mesh.Connectivity[Frac.EltTip, Frac.ZeroVertex]]

    # the distance of tip from the injection point in each of the tip cell
    dist_Inj_pnt = (tipVrtxCoord[:, 0] ** 2 + tipVrtxCoord[:, 1] ** 2) ** 0.5 + Frac.l

    # the time step evaluated by restricting the fracture to propagate not more than 8 percent of the current maximum
    # length
    TimeStep_1 = min(abs(0.08 * pre_factor * dist_Inj_pnt / Frac.v))

    # the time step evaluated by restricting the fraction of the cell that would be traversed in the time step. e.g., if
    # the prefactor is 0.5, the tip in the cell with the largest velocity will progress half of the cell width in either
    # x or y direction depending on which is smaller.
    TimeStep_2 = pre_factor * min(Frac.mesh.hx, Frac.mesh.hy) / np.max(Frac.v)

    if TimeStep_1 < TimeStep_2:
        print("Limiting time step according to the factor by which the fracture length extends...")

    time_step = min(TimeStep_1, TimeStep_2)

    return time_step
