## * --- Importing the important libraries --- * ##
import csv
import numpy as np


# local imports
from mesh import CartesianMesh
from properties import MaterialProperties, FluidProperties, InjectionProperties, SimulationProperties
from fracture import Fracture
from controller import Controller
from fracture_initialization import Geometry, InitializationParameters
from utility import setup_logging_to_console

# setting up the verbosity level of the log at console
setup_logging_to_console(verbosity_level='Info')

## * --- Load the information on the data --- * ##
with open('DikeSimulations.csv', 'r') as dest_f:
    data_iter = csv.reader(dest_f,
                           delimiter=';')
    data = [data for data in data_iter]

## * --- Give here the simulation you want to perform --- * ##
simulation_Name = 'DI_114'

for sim in data:
    if sim[0] == simulation_Name:
        break

## * --- We set up everything for the simulation --- * ##

# creating mesh
Mesh = CartesianMesh(float(sim[9]), float(sim[9]), int(sim[8]), int(sim[8]))

# injection parameters
if sim[7] == '-':
    Injpar = float(sim[6])  # injection rate
else:
    Injpar = np.asarray([[0., float(sim[7])],
                        [float(sim[6]), 0.]])  # injection rate

Injection = InjectionProperties(Injpar,
                                Mesh)

# solid properties
Eprime = float(sim[2]) / (1 - float(sim[3]) ** 2)              # plain strain modulus
density_rock = 2700                                            # density of the rock
density_fluid = density_rock - float(sim[5])                   # fluid density
K_Ic = float(sim[1])                                           # fracture toughness

def sigmaO_func(x, y):                                         # background stress function
    """ This function provides the confining stress over the domain"""
    return (7.5e7 - y) * density_rock * 9.81

# material properties assignment
Solid = MaterialProperties(Mesh,
                           Eprime,
                           toughness=K_Ic,
                           confining_stress_func=sigmaO_func,
                           minimum_width=1e-8)

# fluid properties
Fluid = FluidProperties(viscosity=float(sim[4]), density=density_fluid)

# simulation properties
simulProp = SimulationProperties()
simulProp.finalTime = float(sim[11])            # set very high. simulation will stop either naturally or user needs to
                                                # break it
simulProp.set_outputFolder("./Data/Dikes")      # the disk address where the files are saved
simulProp.gravity = True                        # set up the gravity flag
simulProp.tolFractFront = 3e-3                  # increase the tolerance for fracture front iteration
simulProp.plotTSJump = 10                       # plot every tenth time step
simulProp.saveTSJump = 5                        # save every fifth time step
simulProp.maxSolverItrs = 200                   # maximum iterations of the fixed point iteration
simulProp.toleranceEHL = 1e-3                   # rise the tolerance of the lubrication system

simulProp.set_simulation_name(simulation_Name)  # save it with the name of the simulation
simulProp.plotVar = ['w']                       # plotting the opening of the fracture
simulProp.useBlockToeplizCompression = True     # use a accelerated calculation of the elasticity matrix

simulProp.remeshFactor = float(sim[12])         # remeshing factor when compressing the domain
simulProp.set_mesh_extension_factor([float(sim[13]), float(sim[14]), float(sim[15]), float(sim[16])])
                                                # factor for the mesh extension
simulProp.set_mesh_extension_direction(['top']) # extend only towards the top of the domain
simulProp.meshReductionPossible = True          # allow for mesh reduction
simulProp.meshReductionFactor = float(sim[17])  # factor reducing the number of elements for mesh reduction
simulProp.maxElementIn = float(sim[18])
simulProp.maxCellSize = float(Sim[19])

simulProp.maxTimeSteps = 10000                  # the simulation will stop after 10'000 time steps

# set up the geometry of the initial fracture
Fr_geometry = Geometry('radial')
init_param = InitializationParameters(Fr_geometry,
                                      regime='M',
                                      time=float(sim[10]))

# generate the first fracture
Fr = Fracture(Mesh,
              init_param,
              Solid,
              Fluid,
              Injection,
              simulProp)

# create a controller
controller = Controller(Fr,
                        Solid,
                        Fluid,
                        Injection,
                        simulProp)

# run the simulation
controller.run()