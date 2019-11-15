# -*- coding: utf-8 -*-
"""
This file is an edited version of the PyFrac example files.

Tim Davis GFZ Potsdam, 16-Nov-2019
"""

import numpy as np
import math
import time

# local imports
from mesh import CartesianMesh
from properties import MaterialProperties, FluidProperties, InjectionProperties, SimulationProperties
from fracture import Fracture
from controller import Controller
from fracture_initialization import Geometry, InitializationParameters

# solid properties
nu = 0.25                           # Poisson's ratio       [dmlss] 
G= 1e10                             # Shear modulus         [pa] 
youngs_mod = (2*G)*(1+nu)           # Young's modulus       [pa] 
Eprime = youngs_mod / (1 - nu ** 2) # plain strain modulus  [pa]
K_Ic = 1*1e6                        # fracture toughness    [pa.sqrt(m)]
fluiddensity=1000                   # [kg/m3]
rockdensity=2700                    # [kg/m3]
deltagamma=(rockdensity-fluiddensity)*9.8 # gradient in weight [pa.m^-1]
Cl=0.;


#Some approximate fracture length scales derived analytically based on the parameters above
maxaperture=0.0003101439481468251;
critradius=17.100314412817873;
volume=0.04953872959754216;

#Length and rate of injection (based on volume and parameters above)
lengthofinjection_s=6000/(K_Ic/1e6)              #[s]
rate=volume/lengthofinjection_s                  #[m/s] 
fluidviscosity=1.1e-3;                            #[pa.s] - water=~1.1e-3


#Some scales for the limits of the grid (mesh)
ylength=2
xlength=1
# creating mesh
steprad=round(critradius)
Mesh = CartesianMesh(steprad*xlength, steprad*ylength, int(20),  int(50)) #(50, 75, 41, 61)


#vertical offset
offset=2000;
def sigmaO_func(x, y):
    """ This function provides the confining stress over the domain"""

    # only dependant on the depth
    density = rockdensity
    return (offset-y) * density * 9.8


# material properties
Solid = MaterialProperties(Mesh,
                           Eprime,
                           toughness=K_Ic,
                           confining_stress_func=sigmaO_func,
                           minimum_width=maxaperture/100, 
                           free_surf=0,
                           Carters_coef=Cl) #minimum_width=1e-6  
                           #Carters_coef=1e-6)

# injection parameters
#The simulation consists of a 100 minutes injection (6000s) of water into a rock
Q0 = np.asarray([[0, lengthofinjection_s], [rate, 0]])
Injection = InjectionProperties(Q0,
                                Mesh,
                                source_coordinates=[0, -critradius*(ylength*0.7)])

# fluid properties
Fluid = FluidProperties(viscosity=fluidviscosity,density=fluiddensity) #, 

# simulation properties
simulProp = SimulationProperties()
totaltime= 3.39e3*50; #Arbitary large value
simulProp.finalTime =  totaltime                      # the time at which the simulation stops.
simulProp.gravity = True                              # take the effect of gravity into account.
#simulProp.front_advancing = 'explicit'                # possible options include 'implicit', 'explicit' and default... 'predictor-corrector'.
simulProp.enable_remeshing = False  #kill once at end of domain   


# Formatting strings and saving file
critradiusf="%3g" % critradius 
volumef="%5.3f" % volume       
deltagammaf="%3g" % deltagamma 
Gf="%3g" % G 
K_Icf="%3g" % K_Ic      
Filename='_'.join(['./Data/PyFrac',
  'vol',str(volumef),
  'visc',str(fluidviscosity),
  'c_crit',str(critradiusf),
  'G',str(Gf),
  'nu',str(nu),
  'K_Ic',str(K_Icf),
  'deltagamma',str(deltagammaf)])
simulProp.set_outputFolder(Filename)      # the disk address where the files are saved.


simulProp.tolFractFront = 3.0e-4          # increase the tolerance for fracture front iteration
simulProp.toleranceEHL  = 0.003           # tolerance for the elastohydrodynamic system solver.
simulProp.toleranceProjection = 0.025     # tolerance for the toughness iteration.
simulProp.maxFrontItrs = 25               # maximum iterations for the fracture front.
simulProp.maxSolverItrs = 250             # maximum iterations for the elastohydrodynamic solver.
simulProp.maxProjItrs = 10                # maximum projection iterations.
simulProp.maxReattempts = 16              # maximum reattempts in case of time step failure.
simulProp.reAttemptFactor = 0.9           # the factor by which time step is reduced on reattempts.
simulProp.outputEveryTS = 5
simulProp.elastohydrSolver = 'implicit_Anderson'# the time after the output is generated (saving or plotting).

simulProp.plotVar = ['w']   #, 'v']          # plot fracture width and fracture front velocity

''' - default tolerances

toleranceFractureFront = 1.0e-3         # tolerance for the fracture front position solver.
toleranceEHL = 1.0e-4                   # tolerance for the elastohydrodynamic system solver.
tol_projection = 2.5e-3                 # tolerance for the toughness iteration.

# max iterations
max_front_itrs = 25                     # maximum iterations for the fracture front.
max_solver_itrs = 80                    # maximum iterations for the elastohydrodynamic solver.
max_proj_Itrs = 10                      # maximum projection iterations.

# time step re-attempt
max_reattemps = 8                       # maximum reattempts in case of time step failure.
reattempt_factor = 0.8                  # the factor by which time step is reduced on reattempts.
'''

# initialization parameters
Fr_geometry = Geometry('radial', radius=critradius/2)
init_param = InitializationParameters(Fr_geometry, regime='K')

# creating fracture object
Fr = Fracture(Mesh,
              init_param,
              Solid,
              Fluid,
              Injection,
              simulProp)

# create a Controller
controller = Controller(Fr,
                        Solid,
                        Fluid,
                        Injection,
                        simulProp)

# run the simulation
#controller.run()

###################
#plotting results #
###################

from visualization import *

Fr_list, properties = load_fractures(address=Filename,
                                     #time_srs=time_srs)
                                     time_period=1e4)
time_srs = get_fracture_variable(Fr_list,
                                 variable='time')

# plot footprint
Fig_FP = None
Fig_FP = plot_fracture_list(Fr_list,
                            variable='mesh',
                            projection='2D',
                            mat_properties=Solid,
                            backGround_param='confining stress')
plt_prop = PlotProperties(plot_FP_time=False)
Fig_FP = plot_fracture_list(Fr_list,
                            variable='footprint',
                            projection='2D',
                            fig=Fig_FP,
                            plot_prop=plt_prop)

intercepts = get_front_intercepts(Fr_list,[0,10])
b_stable = intercepts[-1][3] - intercepts[-1][2]

print(b_stable/2)


plt.show(block=True)