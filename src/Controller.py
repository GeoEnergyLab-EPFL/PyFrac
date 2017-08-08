
from src.Fracture import Fracture
from src.Properties import *
from src.Elasticity import *
from src.HFAnalyticalSolutions import *
from src.TimeSteppingVolumeControl import attempt_time_step_volumeControl
from src.TimeSteppingViscousFluid import attempt_time_step_viscousFluid
from src.TimeSteppingMechLoading import attempt_time_step_mechLoading

import copy
import matplotlib.pyplot as plt

class Controller:

    errorMessages = ("Propagation not attempted",
                     "Time step successful",
                     "Evaluated level set is not valid",
                     "Front is not tracked correctly",
                     "Evaluated tip volume is not valid",
                     "Solution obtained from the elastohydrodynamic solver is not valid",
                     "Did not converge after max iterations",
                     "Tip inversion is not correct",
                     "Ribbon element not found in the enclosure of the tip cell",
                     "Filling fraction not correct",
                     "Toughness iteration did not converge"
                     )

    #todo add mesh as an argument
    def __init__(self, Fracture=None, Solid_prop=None, Fluid_prop=None, Injection_prop=None, Sim_prop=None, Load_prop=None):

       self.fracture = Fracture
       self.solid_prop = Solid_prop
       self.fluid_prop = Fluid_prop
       self.injection_prop = Injection_prop
       self.sim_prop = Sim_prop
       self.load_prop = Load_prop

    def run(self):
        # load elasticity matrix
        C = load_elasticity_matrix(self.fracture.mesh, self.solid_prop.Eprime)


        # starting time stepping loop
        i = 0
        Fr_k = self.fracture
        tmSrs_indx = 0
        next_in_tmSrs = self.sim_prop.solTimeSeries[tmSrs_indx]
        if next_in_tmSrs < Fr_k.time:
            raise SystemExit('The minimum time required in the given time series is less than initial time.')

        while (Fr_k.time < self.sim_prop.FinalTime) and (i < self.sim_prop.maxTimeSteps):
            i = i + 1

            # time step is calculated with the current propagation velocity
            TimeStep = self.sim_prop.tmStpPrefactor * min(Fr_k.mesh.hx, Fr_k.mesh.hy) / np.max(Fr_k.v)

            # to get the solution at the times given in time series
            if Fr_k.time + TimeStep > next_in_tmSrs:
                TimeStep = next_in_tmSrs - Fr_k.time
                if tmSrs_indx < len(self.sim_prop.solTimeSeries)-1:
                    tmSrs_indx += 1
                next_in_tmSrs = self.sim_prop.solTimeSeries[tmSrs_indx]

            status, Fr_k = self.advance_time_step(Fr_k,
                                                 C,
                                                 TimeStep)

            Fr = copy.deepcopy(Fr_k)

        print("\n\n-----Simulation successfully finished------")
        print("Final time = " + repr(Fr.time))


#-----------------------------------------------------------------------------------------------------------------------


    def advance_time_step(self, Frac, C, TimeStep):
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

            Fracture object:            fracture after advancing time step.
        """
        print('\n--------------------------------\ntime = ' + repr(Frac.time))
        print("Attempting time step of " + repr(TimeStep) + " sec...")
        # loop for reattempting time stepping in case of failure.
        for i in range(0, self.sim_prop.maxReattempts):
            # smaller time step to reattempt time stepping; equal to the given time step on first iteration
            smallerTimeStep = TimeStep * self.sim_prop.reAttemptFactor ** i

            if self.sim_prop.viscousInjection:
                status, Fr = attempt_time_step_viscousFluid(Frac,
                                                            C,
                                                            self.solid_prop,
                                                            self.fluid_prop,
                                                            self.sim_prop,
                                                            self.injection_prop,
                                                            smallerTimeStep)

            elif self.sim_prop.dryCrack_mechLoading:
                status, Fr = attempt_time_step_mechLoading(Frac,
                                                           C,
                                                           self.solid_prop,
                                                           self.sim_prop,
                                                           self.load_prop,
                                                           smallerTimeStep,
                                                           Frac.mesh)

            elif self.sim_prop.volumeControl:
                status, Fr = attempt_time_step_volumeControl(Frac,
                                                             C,
                                                             self.solid_prop,
                                                             self.sim_prop,
                                                             self.injection_prop,
                                                             smallerTimeStep,
                                                             Frac.mesh)
            if status == 1:
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
                print(self.errorMessages[status])

            print("Time step failed...")
            print("Reattempting with time step of " + repr(TimeStep * self.sim_prop.reAttemptFactor**(i + 1)) + " sec")
        Frac.plot_fracture("complete", "footPrint")
        plt.show()
        raise SystemExit("Propagation not successful. Exiting...")


    #-------------------------------------------------------------------------------------------------------------------

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
        if (Fr_lstTmStp.time // simulation_parameters.outputTimePeriod !=
                    Fr_advanced.time // simulation_parameters.outputTimePeriod):
            # plot fracture footprint
            if simulation_parameters.plotFigure:
                # if ploting analytical solution enabled
                if simulation_parameters.plotAnalytical:
                    Q0 = injection_parameters.injectionRate[1, 0]  # injection rate at the time of injection
                    if simulation_parameters.analyticalSol == "M":
                        (R, p, w, v) = M_vertex_solution_t_given(material_properties.Eprime,
                                                                 Q0,
                                                                 fluid_properties.muPrime,
                                                                 Fr_lstTmStp.mesh,
                                                                 Fr_advanced.time)

                    elif simulation_parameters.analyticalSol == "K":
                        (R, p, w, v) = K_vertex_solution_t_given(material_properties.Kprime,
                                                                 material_properties.Eprime,
                                                                 Q0,
                                                                 Fr_lstTmStp.mesh,
                                                                 Fr_advanced.time)

                    fig = Fr_advanced.plot_fracture('complete',
                                                    'footPrint',
                                                    analytical=R,
                                                    mat_Properties=material_properties)
                else:
                    fig = Fr_advanced.plot_fracture('complete',
                                                    'footPrint',
                                                    mat_Properties=material_properties)
                plt.show()

            # save fracture to disk
            if simulation_parameters.saveToDisk:
                simulation_parameters.lastSavedFile += 1
                Fr_advanced.SaveFracture(simulation_parameters.outFileAddress + "file_"
                                         + repr(simulation_parameters.lastSavedFile))