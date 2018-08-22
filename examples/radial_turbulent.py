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
Mesh = CartesianMesh(0.3, 0.3, 45, 45)

# solid properties
nu = 0.                             # Poisson's ratio
youngs_mod = 2.0e10                 # Young's modulus
Eprime = youngs_mod / (1 - nu ** 2) # plain strain modulus
K_Ic = 1.e5                         # fracture toughness

Solid = MaterialProperties(Mesh,
                           Eprime,
                           K_Ic)

# injection parameters
Q0 = 0.211983  # massive injection rate - 80 BPM
Injection = InjectionProperties(Q0, Mesh)


# fluid properties
Fluid = FluidProperties(viscosity=1.e-3,
                        turbulence=True)

# simulation properties
simulProp = SimulationParameters()
simulProp.FinalTime = 2000.              # the time at which the simulation stops
simulProp.set_tipAsymptote('U')         # the tip asymptote is evaluated with the toughness dominated assumption
simulProp.outputEveryTS = 3
simulProp.set_outputFolder(".\\Data\\MDR_r1") # the disk address where the files are saved
simulProp.saveReynNumb = True
simulProp.verbosity = 2

# initializing fracture
initRad = 0.2
init_param = ("MDR", "length", initRad)

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
Fr_list, properties = load_fractures(".\\Data\\MDR_r1",
                                     time_srs=np.e**np.linspace(np.log(0.00028), np.log(2000), 8))
time_srs = get_fracture_variable(Fr_list,
                                 'time')

Fig_FP = plot_fracture_list(Fr_list,
                           variable='mesh')
Fig_FP = plot_fracture_list(Fr_list,
                           variable='footprint',
                           fig=Fig_FP)
Fig_FP = plot_analytical_solution(regime="M",
                                  variable='footprint',
                                  mat_prop=Solid,
                                  inj_prop=Injection,
                                  fluid_prop=Fluid,
                                  fig=Fig_FP,
                                  time_srs=time_srs)


Fr_list, properties = load_fractures(".\\Data\\MDR_r1",)
time_srs = get_fracture_variable(Fr_list,
                                 'time')

plot_prop = PlotProperties(graph_scaling='loglog',
                           line_style='.')

label = get_labels('d_mean')
label.legend = 'radius'
Fig_r = plot_fracture_list(Fr_list,
                           variable='d_mean',
                           plot_prop=plot_prop,
                           labels=label)

label.legend = 'radius analytical (MDR asymptote)'
Fig_r = plot_analytical_solution(regime="MDR",
                                  variable='d_mean',
                                  mat_prop=Solid,
                                  inj_prop=Injection,
                                  fluid_prop=Fluid,
                                  fig=Fig_r,
                                  time_srs=time_srs,
                                  labels=label)
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

Fr_list, properties = load_fractures(".\\Data\\MDR_r1",
                                     time_srs=np.e**np.linspace(np.log(0.0002), np.log(2000), 10))
Fig_RN = plot_fracture_list_slice(Fr_list,
                                  variable='Rn',
                                  point1=[-Fr_list[-1].mesh.Lx, 0],
                                  point2=[Fr_list[-1].mesh.Lx, 0],
                                  edge=1)
plt.show()


