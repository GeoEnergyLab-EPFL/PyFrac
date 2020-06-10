# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri June 16 17:49:21 2017.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2019.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# local imports
from mesh import CartesianMesh
from properties import MaterialProperties, FluidProperties, InjectionProperties, SimulationProperties
from fracture import Fracture
from controller import Controller
from fracture_initialization import Geometry, InitializationParameters
from visualization import *
from scipy import interpolate

# creating mesh
Mesh = CartesianMesh(.4, .4, 41, 41)

# solid properties
nu = 0.15                               # Poisson's ratio
youngs_mod = 3e10                       # Young's modulus
Eprime = youngs_mod / (1 - nu ** 2)     # plain strain modulus
K_Ic = 2e6                              # fracture toughness
sigma0 = 15e6                           # confining stress

# material properties
Solid = MaterialProperties(Mesh,
                           Eprime,
                           K_Ic, 
                           Carters_coef=1e-6,
                           confining_stress=sigma0,
                           minimum_width=1e-5)

Q0 = 0.001/60
Q0 = np.asarray([[0, 240], [0.001/60, 0]])
Injection = InjectionProperties(Q0, Mesh)

# fluid properties
Fluid = FluidProperties(viscosity=0.22, rheology='HBF', compressibility=1e-11, n=0.617, k=0.22, T0=2.3)

# simulation properties
simulProp = SimulationProperties()
simulProp.finalTime = 2555                              # the time at which the simulation stops
simulProp.set_outputFolder("./Data/HB")                 # the disk address where the files are saved
simulProp.set_simulation_name('Herschel-Bulkley_closure')
simulProp.saveG = True                                  # enable saving the coefficient G
simulProp.plotVar = ['w']                               # plot width of fracture 
simulProp.saveEffVisc = True                            # enable saving of the effective viscosity
simulProp.relaxation_factor = 0.3                       # relax Anderson iteration
simulProp.maxSolverItrs = 200                           # set maximum number of Anderson iterations to 200
simulProp.Anderson_parameter = 10                       # save last 10 iterations in Anderson iteration
simulProp.collectPerfData = True                        # enable collect performace data
simulProp.fixedTmStp = np.asarray([[0, 350],[None, 250]])# set fixed time step after injection stops


# starting simulation with a static radial fracture with radius 20cm and pressure of 1MPa
Fr_geometry = Geometry('radial', radius=0.2)
from elasticity import load_isotropic_elasticity_matrix
C = load_isotropic_elasticity_matrix(Mesh, Eprime)
init_param = InitializationParameters(Fr_geometry, regime='static', net_pressure=1e6, elasticity_matrix=C)

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
controller.run()

####################
# plotting results #
####################

from visualization import *

# loading simulation results
tm_srs = np.linspace(0.07, 0.35, 5)
Fr_list, properties = load_fractures(address="./Data/HB",
                                      sim_name='Herschel-Bulkley_closure',
                                        time_srs=tm_srs,  
                                      )
Fig_G = Fig_w = Fig_ev = None
plt_prop = PlotProperties(line_style='.')
Fig_G = plot_fracture_list_slice(Fr_list,
                          variable='w',
                          plot_prop=plt_prop,
                          fig = Fig_G)


plt_prop = PlotProperties(line_style='.')
Fig_ev = plot_fracture_list_slice(Fr_list,
                          variable='ev',
                          plot_prop=plt_prop,
                          fig = Fig_ev)


tm_srs = np.geomspace(0.35, 240, 5)
Fr_list, properties = load_fractures(address="./Data/HB",
                                      sim_name='Herschel-Bulkley_closure',
                                        time_srs=tm_srs,  
                                      )
Fig_G = Fig_w = Fig_ev = None
plt_prop = PlotProperties(line_style='.')
Fig_G = plot_fracture_list_slice(Fr_list,
                          variable='w',
                          plot_prop=plt_prop,
                          fig = Fig_G)


plt_prop = PlotProperties(line_style='.')
Fig_ev = plot_fracture_list_slice(Fr_list,
                          variable='ev',
                          plot_prop=plt_prop,
                          fig = Fig_ev)


tm_srs = np.geomspace(241, 2550, 5)
Fr_list, properties = load_fractures(address="./Data/HB",
                                      sim_name='Herschel-Bulkley_closure',
                                        time_srs=tm_srs,  
                                      )
Fig_G = Fig_w = Fig_ev = None
plt_prop = PlotProperties(line_style='.')
Fig_G = plot_fracture_list_slice(Fr_list,
                          variable='w',
                          plot_prop=plt_prop,
                          fig = Fig_G)


plt_prop = PlotProperties(line_style='.')
Fig_ev = plot_fracture_list_slice(Fr_list,
                          variable='ev',
                          plot_prop=plt_prop,
                          fig = Fig_ev)

# plt.show(block=True)

