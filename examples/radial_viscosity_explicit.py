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
Mesh = CartesianMesh(0.3, 0.3, 41, 41)

# solid properties
nu = 0.4                            # Poisson's ratio
youngs_mod = 3.3e10                 # Young's modulus
Eprime = youngs_mod / (1 - nu ** 2) # plain strain modulus
K_Ic = 0.5                          # fracture toughness

# material properties
Solid = MaterialProperties(Mesh,
                           Eprime,
                           K_Ic)

# injection parameters
Q0 = 0.001  # injection rate
Injection = InjectionProperties(Q0, Mesh)

# fluid properties
Fluid = FluidProperties(viscosity=1.1e-3)

# simulation properties
simulProp = SimulationProperties()
simulProp.finalTime = 1e5               # the time at which the simulation stops
simulProp.set_tipAsymptote('M')         # the tip asymptote is evaluated with the viscosity dominated assumption
simulProp.frontAdvancing = 'explicit'   # to set explicit front tracking
simulProp.outputTimePeriod = 1e-4       # to save after every time step
simulProp.tmStpPrefactor = 0.5          # decrease the pre-factor due to explicit front tracking
simulProp.set_outputFolder(".\\Data\\M_radial_explicit") # the disk address where the files are saved

# initialization parameters
initRad = 0.1
init_param = ("M", "length", initRad)

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
# controller.run()


####################
# plotting results #
####################

# loading simulation results
Fr_list, properties = load_fractures(address=".\\Data\\M_radial_explicit")       # load all fractures
time_srs = get_fracture_variable(Fr_list,                                        # list of times
                                 variable='time')

# plot fracture radius
plot_prop = PlotProperties()
plot_prop.lineStyle = '.'               # setting the linestyle to point
plot_prop.graphScaling = 'loglog'       # setting to log log plot
Fig_R = plot_fracture_list(Fr_list,
                           variable='d_mean',
                           plot_prop=plot_prop)
# plot analytical radius
Fig_R = plot_analytical_solution(regime='M',
                                 variable='d_mean',
                                 mat_prop=Solid,
                                 inj_prop=Injection,
                                 fluid_prop=Fluid,
                                 time_srs=time_srs,
                                 fig=Fig_R)

# plot width at center
Fig_w = plot_fracture_list_at_point(Fr_list,
                                    variable='w',
                                    plot_prop=plot_prop)
# plot analytical width at center
Fig_w = plot_analytical_solution_at_point('M',
                                          'w',
                                          Solid,
                                          Injection,
                                          fluid_prop=Fluid,
                                          time_srs=time_srs,
                                          fig=Fig_w)


time_srs = np.linspace(1, 1e5, 5)
Fr_list, properties = load_fractures(address=".\\Data\\M_radial_explicit",
                                     time_srs=time_srs)
time_srs = get_fracture_variable(Fr_list,
                                 variable='time')

# plot footprint
Fig_FP = plot_fracture_list(Fr_list,
                            variable='mesh',
                            projection='2D')
Fig_FP = plot_fracture_list(Fr_list,
                            variable='footprint',
                            projection='2D',
                            fig=Fig_FP)
# plot analytical footprint
Fig_FP = plot_analytical_solution('M',
                                  'footprint',
                                  Solid,
                                  Injection,
                                  fluid_prop=Fluid,
                                  time_srs=time_srs,
                                  projection='2D',
                                  fig=Fig_FP)


# plot slice
ext_pnts = np.empty((2, 2), dtype=np.float64)
Fig_WS = plot_fracture_list_slice(Fr_list,
                                  variable='w',
                                  projection='2D',
                                  plot_cell_center=True,
                                  extreme_points=ext_pnts)
#plot slice analytical
Fig_WS = plot_analytical_solution_slice('M',
                                        'w',
                                        Solid,
                                        Injection,
                                        fluid_prop=Fluid,
                                        fig=Fig_WS,
                                        time_srs=time_srs,
                                        point1=ext_pnts[0],
                                        point2=ext_pnts[1],
                                        plt_top_view=True)

#plotting in 3D
Fig_Fr = plot_fracture_list(Fr_list,
                            variable='mesh',
                            projection='3D')
Fig_Fr = plot_fracture_list(Fr_list,
                            variable='width',
                            projection='3D',
                            fig=Fig_Fr)
Fig_Fr = plot_fracture_list(Fr_list,
                            variable='footprint',
                            projection='3D',
                            fig=Fig_Fr)

plt.show()

