# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri June 16 17:49:21 2017.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2017. All rights reserved.
See the LICENSE.TXT file for more details.
"""

# imports
from src.Fracture import *
from src.Controller import *
from src.PostProcessFracture import *

# creating mesh
Mesh = CartesianMesh(0.2, 0.2, 45, 45)

# solid properties
Eprime = 2.0e10             # plain strain modulus
K_Ic = 0.5e6                 # fracture toughness

Solid = MaterialProperties(Mesh,
                           Eprime,
                           K_Ic)

# injection parameters
Q0 = 0.079  # massive injection rate - 80 BPM
Injection = InjectionProperties(Q0, Mesh)


# fluid properties
Fluid = FluidProperties(viscosity=1.e-3,
                        turbulence=True)

# simulation properties
simulProp = SimulationParameters()
simulProp.FinalTime = 2000.             # the time at which the simulation stops
simulProp.outputEveryTS = 2             # write a file after every 3 time steps
simulProp.set_outputFolder(".\\Data\\radial_TtoM") # the disk address where the files are saved
simulProp.saveReynNumb = True           # enable saving Reynolds number at each edge of grid
simulProp.plotFigure = True
simulProp.plotAnalytical = True
simulProp.analyticalSol = 'MDR'
simulProp.verbosity = 2

# initializing fracture
initTime = 2e-4
init_param = ("MDR", "time", initTime)

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


# plot results
Fr_list, properties = load_fractures(".\\Data\\radial_TtoM",)
time_srs = get_fracture_variable(Fr_list,
                                 'time')

plot_prop = PlotProperties(graph_scaling='loglog',
                           line_style='.')

# plotting mean distance from the injection point (radius in this case)
label = get_labels('d_mean')
label.legend = 'radius'
Fig_r = plot_fracture_list(Fr_list,
                           variable='d_mean',
                           plot_prop=plot_prop,
                           labels=label)

# plotting analytical radius from MDR radial solution
label.legend = 'radius analytical (MDR asymptote)'
Fig_r = plot_analytical_solution(regime="MDR",
                                  variable='d_mean',
                                  mat_prop=Solid,
                                  inj_prop=Injection,
                                  fluid_prop=Fluid,
                                  fig=Fig_r,
                                  time_srs=time_srs,
                                  labels=label)
# plotting analytical radius from toughness dominated radial solution
plot_prop.lineColorAnal = 'b'
label.legend = 'radius analytical (viscosity dominated)'
Fig_r = plot_analytical_solution(regime="M",
                                  variable='d_mean',
                                  mat_prop=Solid,
                                  inj_prop=Injection,
                                  fluid_prop=Fluid,
                                  fig=Fig_r,
                                  time_srs=time_srs,
                                  plot_prop=plot_prop,
                                  labels=label)

label = get_labels('w')
label.legend = 'width at injection'
Fig_wc = plot_fracture_list_at_point(Fr_list,
                           variable='w',
                           plot_prop=plot_prop,
                           labels=label)
# plotting analytical radius from MDR radial solution
label.legend = 'radius analytical (MDR asymptote)'
Fig_r = plot_analytical_solution_at_point(regime="MDR",
                                  variable='w',
                                  mat_prop=Solid,
                                  inj_prop=Injection,
                                  fluid_prop=Fluid,
                                  fig=Fig_wc,
                                  time_srs=time_srs,
                                  labels=label)
# plotting analytical radius from toughness dominated radial solution
plot_prop.lineColorAnal = 'b'
label.legend = 'radius analytical (viscosity dominated)'
Fig_r = plot_analytical_solution_at_point(regime="M",
                                  variable='w',
                                  mat_prop=Solid,
                                  inj_prop=Injection,
                                  fluid_prop=Fluid,
                                  fig=Fig_wc,
                                  time_srs=time_srs,
                                  plot_prop=plot_prop,
                                  labels=label)

# plotting fracture slice along x-axis
Fr_list, properties = load_fractures(".\\Data\\radial_TtoM",
                                     time_srs=np.e**np.linspace(np.log(0.02), np.log(2000), 8))
time_srs = get_fracture_variable(Fr_list,
                                 'time')

plot_prop = PlotProperties(plot_legend=False) # instantiate plot properties object with legends disabled

Fig_w, pnt1, pnt2 = plot_fracture_list_slice(Fr_list,
                                  variable='w',
                                  # point1=[-0.5, .5],
                                  # plt_2D_image=False,
                                  plot_cell_center=True,
                                 orientation='horizontal',
                                             )

Fig_w = plot_analytical_solution_slice('M',
                                       'w',
                                       Solid,
                                       Injection,
                                       fluid_prop=Fluid,
                                       point1=pnt1,
                                       point2=pnt2,
                                       time_srs=time_srs,
                                       fig=Fig_w,
                                       # plt_2D_image=False,      # only plot slice without top view
                                       plot_prop=plot_prop,     # give the plot properties with legends disabled
                                       )




# Fig_wT = plot_analytical_solution_slice('MDR',
#                                        'w',
#                                        Solid,
#                                        Injection,
#                                        fluid_prop=Fluid,
#                                        point1=[-Fr_list[-1].mesh.Lx, 0],
#                                        point2=[Fr_list[-1].mesh.Lx, 0],
#                                        time_srs=time_srs,
#                                        # plt_2D_image=False,
#                                        # plot_prop=plot_prop,
#                                        )
#
# Fig_wT = plot_fracture_list_slice(Fr_list,
#                                   variable='w',
#                                   point1=[-Fr_list[-1].mesh.Lx, 0],
#                                   point2=[Fr_list[-1].mesh.Lx, 0],
#                                     fig=Fig_wT,
#                                   # plt_2D_image=False,
#                                   # plot_prop=plot_prop,
#                                   )

plt.show()
