# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri Dec 16 17:49:21 2017.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2017. All rights
reserved. See the LICENSE.TXT file for more details.
"""

# imports
from src.Fracture import *
from src.Controller import *

# creating mesh
Mesh = CartesianMesh(22, 22, 41, 41)

# solid properties
nu = 0.4                            # Poisson's ratio
youngs_mod = 3.3e10                 # Young's modulus
Eprime = youngs_mod / (1 - nu**2)   # plain strain modulus
K1c = 1e4                           # fracture toughness

Solid = MaterialProperties(Mesh,
                           Eprime,
                           K1c,
                           Carters_coef=1.0e-6,
                           confining_stress=1e6,
                           minimum_width=5e-5,
                           pore_pressure=1.e5)

# injection parameters
Q0 = np.asarray([[0, 60., 1600, 1700], [0.01, 0, 0.01, -0.005]])
Injection = InjectionProperties(Q0, Mesh)

# fluid properties
viscosity = 1e-4
Fluid = FluidProperties(viscosity=viscosity, compressibility=1e-10)

# simulation properties
simulProp = SimulationProperties()
simulProp.finalTime = 2500               # the time at which the simulation stops
simulProp.set_outputFolder(".\\Data\\closure") # the disk address where the files are saved
simulProp.plotVar = ['w', 'lk']
simulProp.tolFractFront = 4e-3

# initializing fracture
Fr_geometry = Geometry('radial')
init_param = InitializationParameters(Fr_geometry, regime='M', time=18)

# #creating fracture object
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

##################
# plotting results
##################

# loading results
Fr_list, properties = load_fractures(".\\Data\\closure")
time_srs = get_fracture_variable(Fr_list,
                                 'time')

# show an animation for fracture width and pressure
animate_simulation_results(Fr_list,
                           variable=['w', 'pf'])

# plotting pressure at injection point
p_prop = PlotProperties(line_style='.', graph_scaling='loglog')
Fig_p = plot_fracture_list_at_point(Fr_list,
                                    variable='pf')

# plotting pressure during initial propagation
time_srs = np.linspace(21, 200, 6)
Fr_list, properties = load_fractures(".\\Data\\closure",
                                     time_srs=time_srs)
plot_prop = PlotProperties(line_style='.-')
Fig_Ps = plot_fracture_list_slice(Fr_list,
                                  variable='pf',
                                  plot_cell_center=True,
                                  plot_prop=plot_prop)

# plotting pressure during closure due to leak off
time_srs = np.linspace(500, 800, 10)
Fr_list, properties = load_fractures(".\\Data\\closure",
                                     time_srs=time_srs)
plot_prop = PlotProperties(line_style='.-')
Fig_Ps = plot_fracture_list_slice(Fr_list,
                                  variable='pf',
                                  plot_cell_center=True,
                                  plot_prop=plot_prop)

# plotting pressure during propagation after re-injection
time_srs = np.linspace(1600, 1700, 5)
Fr_list, properties = load_fractures(".\\Data\\closure",
                                     time_srs=time_srs)
plot_prop = PlotProperties(line_style='.-')
Fig_Ps = plot_fracture_list_slice(Fr_list,
                                  variable='pf',
                                  plot_cell_center=True,
                                  plot_prop=plot_prop)

# plotting pressure during closure due to flow back
time_srs = np.linspace(1700, 1840, 5)
Fr_list, properties = load_fractures(".\\Data\\closure",
                                     time_srs=time_srs)
plot_prop = PlotProperties(line_style='.-')
Fig_Ps = plot_fracture_list_slice(Fr_list,
                                  variable='pf',
                                  plot_cell_center=True,
                                  plot_prop=plot_prop)

plt.show(block=True)
