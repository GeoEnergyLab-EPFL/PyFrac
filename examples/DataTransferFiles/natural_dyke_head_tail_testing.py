# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri June 16 17:49:21 2017.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2019. All rights
reserved. See the LICENSE.TXT file for more details.
"""

import numpy as np

# local imports
from mesh import CartesianMesh
from properties import MaterialProperties, FluidProperties, InjectionProperties, SimulationProperties
from fracture import Fracture
from controller import Controller
from fracture_initialization import Geometry, InitializationParameters
from elasticity import load_isotropic_elasticity_matrix
import time
# creating mesh
Mesh = CartesianMesh(1100, 5000, 61, 201)

# solid properties
nu = 0.25                           # Poisson's ratio
youngs_mod = 20e9               # Young's modulus
Eprime = youngs_mod / (1 - nu ** 2) # plain strain modulus


def sigmaO_func(x, y):
    """ This function provides the confining stress over the domain"""
    density_high = 2700
    density_low = 2675
    layer = 500
    Ly = 7500
    #if y > layer:
        #return (Ly - y) * density_low * 9.8
    # only dependant on the depth
    #return (Ly - y) * density_high * 9.8 - (Ly - layer) * (density_high - density_low) * 9.8
    return (Ly - y) * density_high * 9.8

# material properties
Solid = MaterialProperties(Mesh,
                           Eprime,
                           toughness=50e6,
                           confining_stress_func=sigmaO_func,
                           minimum_width=1e-5)

# injection parameters
Q0 = np.asarray([[0.0,  15000],
                [25,    0]])  # injection rate
Injection = InjectionProperties(Q0,
                                Mesh,
                                source_coordinates=[0, -3900])

# fluid properties
Fluid = FluidProperties(viscosity=50, density=2650)

# simulation properties
simulProp = SimulationProperties()
simulProp.finalTime = 1e15
simulProp.frontAdvancing = 'implicit'  # the time at which the simulation stops
simulProp.set_outputFolder("./Data/neutral_buoyancy") # the disk address where the files are saved
simulProp.gravity = True                    # set up the gravity flag
simulProp.tolFractFront = 3e-3              # increase the tolerance for fracture front iteration
simulProp.plotTSJump = 1                    # plot every fourth time step
simulProp.saveTSJump = 5                   # save every second time step
simulProp.maxSolverItrs = 200               # increase the Picard iteration limit for the elastohydrodynamic solver
#simulProp.tmStpPrefactor = np.asarray([[0, 100], [0.3, 0.1]]) # set up the time step prefactor
simulProp.timeStepLimit = 1e10       # time step limit
simulProp.plotVar = ['w']              # plot fracture width and fracture front velocity
#simulProp.blockFigure = True
sim_name = "027_pc_NR"
simulProp.set_simulation_name(sim_name)
simulProp.solveStagnantTip = True
simulProp.saveRegime = True
simulProp.solveTipCorrRib = True
#simulProp.elastohydrSolver = 'implicit_Anderson'
simulProp.projMethod = 'LS_continousfront'
simulProp.maxReattempts = 10
simulProp.saveToDisk = False

# initializing a static fracture
C = load_isotropic_elasticity_matrix(Mesh, Solid.Eprime)
Fr_geometry = Geometry('radial', radius=300)
init_param = InitializationParameters(Fr_geometry,
                                      regime='M',
                                      net_pressure=1e6,
                                      elasticity_matrix=C)

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

start_time = time.time()
# run the simulation
controller.run()
print("--- %s seconds ---" % (time.time() - start_time))

####################
# plotting results #
####################

from visualization import *

# loading simulation results
time_srs = np.asarray([1, 1265.4078530361207,5000,1e6,5e6,7.5e6,1e8,1e12,1e13,1e15])
Fr_list, properties = load_fractures(address="./Data/neutral_buoyancy",
                                     sim_name=sim_name,
                                     time_period=1e12)
                                     #time_srs=time_srs)
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

# plot width in 3D
plot_prop_magma=PlotProperties(color_map='jet', alpha=0.2)
Fig_Fr = plot_fracture_list(Fr_list[2:],
                            variable='width',
                            projection='3D',
                            plot_prop=plot_prop_magma
                            )
Fig_Fr = plot_fracture_list(Fr_list[1:],
                            variable='footprint',
                            projection='3D',
                            fig=Fig_Fr)

# plot slice
ext_pnts = np.empty((2, 2), dtype=np.float64)
Fig_WS = plot_fracture_list_slice(Fr_list,
                                  variable='w',
                                  point1=[0,1000],
                                  point2=[0,-2000],
                                  projection='2D',
                                  plot_cell_center=True,
                                  orientation='vertical',
                                  extreme_points=ext_pnts)

evaluation_point = [0.0,-2100]

intercepts = get_front_intercepts(Fr_list,evaluation_point)
b_stable = intercepts[-2][3] - intercepts[-2][2]

print(b_stable/2)

plt.show(block=True)
print(b_stable/2)
#  set block=True and comment last 2 lines if you want to keep the window open
#plt.show(block=False)
#plt.pause(5)
plt.close()