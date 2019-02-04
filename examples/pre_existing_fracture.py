# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri June 16 17:49:21 2017.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2019. All rights reserved.
See the LICENSE.TXT file for more details.
"""

# imports
from src.Fracture import *
from src.Controller import *
from src.FractureInitilization import get_eliptical_survey_cells

# creating mesh
Mesh = CartesianMesh(6, 6, 41, 41)

# solid properties
nu = 0.4                            # Poisson's ratio
youngs_mod = 3.3e10                 # Young's modulus
Eprime = youngs_mod / (1 - nu ** 2) # plain strain modulus
K_Ic = 5e6                          # fracture toughness

Solid = MaterialProperties(Mesh,
                           Eprime,
                           K_Ic)

# injection parameters
Q0 = 0.001  # injection rate
Injection = InjectionProperties(Q0, Mesh)

# fluid properties
viscosity = 1.1e-3
Fluid = FluidProperties(viscosity=viscosity)

# simulation properties
simulProp = SimulationProperties()
simulProp.finalTime = 50                        # the time at which the simulation stops
simulProp.outputEveryTS = 2                     # the fracture file will be saved after every 2 time steps
simulProp.set_outputFolder("./Data/star")     # the address of the output folder
simulProp.fixedTmStp = np.asarray([[2e-4, 1.], [1., None]]) # list giving the time steps to adopt at the given times
simulProp.plotFigure = True                     # the fracture footprint will be plotted during the simulation

# initializing fracture
initRad = np.pi
surv_cells, inner_cells = get_eliptical_survey_cells(Mesh, initRad, initRad)
surv_cells_dist = np.cos(Mesh.CenterCoor[surv_cells,0])+2.5 - abs(Mesh.CenterCoor[surv_cells,1])
C = load_elasticity_matrix(Mesh, Eprime)
init_param = ('G',              # type of initialization
              surv_cells,       # the given survey cells
              inner_cells,      # the cell enclosed by the survey cells
              surv_cells_dist,  # the distance of the survey cells from the front
              None,             # the given width
              1e4,              # the pressure (uniform in this case)
              C,                # the elasticity matrix
              None,             # the volume of the fracture
              0)                # the velocity of the propagating front (stationary in this case)


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

# loading simulation results
Fr_list, properties = load_fractures(address="./Data/star")
time_srs = get_fracture_variable(Fr_list,
                                 'time')

# plotting maximum distance of the front from the injection point
plot_prop = PlotProperties(line_style='.')
Fig_d = plot_fracture_list(Fr_list,
                           variable='d_max',
                           plot_prop=plot_prop)
Fig_FP = plot_analytical_solution(regime='K',
                                 variable='d_max',
                                 mat_prop=Solid,
                                 inj_prop=Injection,
                                 fig=Fig_d,
                                 time_srs=time_srs)

# loading five fractures from the simulation separated with equal time period
Fr_list, properties = load_fractures(address="./Data/star",
                                     time_srs=np.linspace(0, 50, 5))
# getting exact time of the loaded fractures
time_srs = get_fracture_variable(Fr_list,
                                 'time')

# plotting footprint
Fig_FP = plot_fracture_list(Fr_list,
                                variable='mesh')
Fig_FP = plot_fracture_list(Fr_list,
                                variable='footprint',
                                fig=Fig_FP)
Fig_FP = plot_analytical_solution(regime='K',
                                 variable='footprint',
                                 mat_prop=Solid,
                                 inj_prop=Injection,
                                 fig=Fig_FP,
                                 time_srs=time_srs)

# plotting fracture in 3D
Fig_3D = plot_fracture_list(Fr_list,
                            variable='mesh',
                            projection='3D')
Fig_3D = plot_fracture_list(Fr_list,
                            variable='footprint',
                            projection='3D',
                            fig=Fig_3D)
plot_prop = PlotProperties(alpha=0.3)
Fig_3D = plot_fracture_list(Fr_list,
                            variable='width',
                            projection='3D',
                            fig=Fig_3D,
                            plot_prop=plot_prop)


plt.show()

