# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri June 16 17:49:21 2017.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2019. All rights
reserved. See the LICENSE.TXT file for more details.
"""

# imports
from src.Fracture import *
from src.Controller import *


# creating mesh
Mesh = CartesianMesh(2., 2., 41, 41)

# solid properties
nu = 0.4                            # Poisson's ratio
youngs_mod = 3.3e10                 # Young's modulus
Eprime = youngs_mod / (1 - nu ** 2) # plain strain modulus
K_Ic = 1.5e6                        # fracture toughness

# material properties
Solid = MaterialProperties(Mesh,
                           Eprime,
                           K_Ic)

# injection parameters
Q0 = 0.01  # injection rate
Injection = InjectionProperties(Q0, Mesh)

# fluid properties
Fluid = FluidProperties(viscosity=1.1e-3)

# simulation properties
simulProp = SimulationProperties()
simulProp.finalTime = 1e7               # the time at which the simulation stops
simulProp.saveRegime = True             # enable saving the regime
simulProp.set_outputFolder("./Data/MtoK") #the folder where the results are saved
simulProp.frontAdvancing = 'explicit'

# initializing fracture
Fr_geometry = Geometry('radial', radius=1.5)
init_param = InitializationParameters(Fr_geometry, regime='M')

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

# loading 8 fractures between 0.134s and 1e7s spaced with equal time periods
Fr_list, properties = load_fractures("./Data/MtoK",
                                     time_srs=np.linspace(0.134, 1e7, 8))
# getting exact time of the loaded fractures to be used to compare analytical solution
time_srs = get_fracture_variable(Fr_list,
                                 'time')

# plotting footprint
Fig_FP = plot_fracture_list(Fr_list,
                           variable='mesh')
Fig_FP = plot_fracture_list(Fr_list,
                           variable='footprint',
                           fig=Fig_FP)
Fig_FP = plot_analytical_solution(regime="K",
                                  variable='footprint',
                                  mat_prop=Solid,
                                  inj_prop=Injection,
                                  fluid_prop=Fluid,
                                  fig=Fig_FP,
                                  time_srs=time_srs)
# plotting radius
Fr_list, properties = load_fractures("./Data/MtoK")
time_srs = get_fracture_variable(Fr_list,
                                 'time')

plot_prop = PlotProperties(graph_scaling='loglog',
                           line_style='.')
label = LabelProperties('d_mean', 'wm')
label.legend = 'radius'
Fig_r = plot_fracture_list(Fr_list,
                           variable='d_mean',
                           plot_prop=plot_prop,
                           labels=label)

label.legend = 'radius analytical (viscosity dominated)'
Fig_r = plot_analytical_solution(regime="M",
                                  variable='d_mean',
                                  mat_prop=Solid,
                                  inj_prop=Injection,
                                  fluid_prop=Fluid,
                                  fig=Fig_r,
                                  time_srs=time_srs,
                                  labels=label)
plot_prop.lineColorAnal = 'b'
label.legend = 'radius analytical (toughness dominated)'
Fig_r = plot_analytical_solution(regime="K",
                                  variable='d_mean',
                                  mat_prop=Solid,
                                  inj_prop=Injection,
                                  fluid_prop=Fluid,
                                  fig=Fig_r,
                                  time_srs=time_srs,
                                  plot_prop=plot_prop,
                                  labels=label)

# plotting width at injection point
plot_prop = PlotProperties(graph_scaling='loglog',
                           line_style='.')
Fig_w = plot_fracture_list_at_point(Fr_list,
                                    variable='width',
                                    plot_prop=plot_prop)

label = LabelProperties('width', data_subset='point')
label.legend = 'width analytical (viscosity dominated)'
Fig_w = plot_analytical_solution_at_point(regime="M",
                                          variable='width',
                                          mat_prop=Solid,
                                          inj_prop=Injection,
                                          fluid_prop=Fluid,
                                          fig=Fig_w,
                                          time_srs=time_srs,
                                          labels=label)
plot_prop.lineColorAnal = 'b'
label.legend = 'width analytical (toughness dominated)'
Fig_w = plot_analytical_solution_at_point(regime="K",
                                      variable='width',
                                      mat_prop=Solid,
                                      inj_prop=Injection,
                                      fluid_prop=Fluid,
                                      fig=Fig_w,
                                      time_srs=time_srs,
                                      plot_prop=plot_prop,
                                      labels=label)
plt.show(block=True)
