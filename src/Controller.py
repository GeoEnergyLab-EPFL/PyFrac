
from src.Fracture import Fracture
from src.Properties import *
from src.Elasticity import *
from src.FractureFrontLoop import advance_time_step
import copy

class Controller:

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

            if self.sim_prop.dryCrack_mechLoading:
                status, Fr_k = advance_time_step(Fr_k,
                                                 C,
                                                 self.solid_prop,
                                                 self.sim_prop,
                                                 TimeStep,
                                                 Loading_Parameters=self.load_prop
                                                 )
            elif self.sim_prop.viscousInjection:
                status, Fr_k = advance_time_step(Fr_k,
                                                 C,
                                                 self.solid_prop,
                                                 self.sim_prop,
                                                 TimeStep,
                                                 Fluid_properties=self.fluid_prop,
                                                 Injection_Parameters=self.injection_prop,
                                                 )
            elif self.sim_prop.volumeControl:
                status, Fr_k = advance_time_step(Fr_k,
                                                 C,
                                                 self.solid_prop,
                                                 self.sim_prop,
                                                 TimeStep,
                                                 Injection_Parameters=self.injection_prop,
                                                 )

            Fr = copy.deepcopy(Fr_k)

        print("\n\n-----Simulation successfully finished------")
        print("Final time = " + repr(Fr.time))