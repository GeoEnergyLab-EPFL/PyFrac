# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri June 16 17:49:21 2017.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2019. All rights reserved.
See the LICENSE.TXT file for more details.
"""

# imports
import os

# local imports
from mesh import CartesianMesh
from properties import MaterialProperties, FluidProperties, InjectionProperties, SimulationProperties
from fracture import Fracture
from controller import Controller
from fracture_initialization import Geometry, InitializationParameters
from utility import setup_logging_to_console

# setting up the verbosity level of the log at console
setup_logging_to_console(verbosity_level='info')

# creating mesh
Mesh = CartesianMesh(2.75, 2.4, 17, 35)

# solid properties
nu = 0.4                            # Poisson's ratio
youngs_mod = 3.3e10                 # Young's modulus
Eprime = youngs_mod / (1 - nu ** 2) # plain strain modulus
K_Ic = 0                          # fracture toughness of the material

def sigmaO_func(x, y):
    """ The function providing the confining stress"""
    if abs(y) > 3:
        return 7.5e6
    else:
        return 1e6

Solid = MaterialProperties(Mesh,
                           Eprime,
                           K_Ic,
                           confining_stress_func=sigmaO_func)

# injection parameters
Q0 = 0.001  # injection rate
Injection = InjectionProperties(Q0, Mesh)

# fluid properties
Fluid = FluidProperties(viscosity=1.1e-3)

# simulation properties
simulProp = SimulationProperties()
simulProp.finalTime = 145.                              # the time at which the simulation stops
simulProp.bckColor = 'sigma0'                           # setting the parameter according to which the mesh is colored
simulProp.set_outputFolder("./Data/height_contained")   # set the directory to save the simulation
simulProp.tmStpPrefactor = 1.0                          # decreasing the size of time step
simulProp.plotVar = ['footprint']                       # plotting footprint
simulProp.set_mesh_extension_direction(['horizontal'])  # allow the mesh to extend horizontally
simulProp.set_mesh_extension_factor(1.35)               # setting the mesh extension factor
simulProp.useBlockToeplizCompression = True             # use the Toepliz elasticity to save memory

# initializing fracture
Fr_geometry = Geometry(shape='radial', radius=1.)
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

if not os.path.isfile('./batch_run.txt'): # We only visualize for runs of specific examples

    from visualization import *

    # loading simulation results
    Fr_list, properties = load_fractures(address="./Data/height_contained")
    time_srs = get_fracture_variable(Fr_list,
                                     variable='time')

    label = LabelProperties('d_max')
    label.legend = 'fracture length'

    plot_prop = PlotProperties(line_style='.',
                               graph_scaling='loglog')

    Fig_r = plot_fracture_list(Fr_list,  # plotting footprint
                               variable='d_max',
                               plot_prop=plot_prop,
                               labels=label)

    label.legend = 'fracture length analytical (PKN)'
    Fig_r = plot_analytical_solution('PKN',
                                     variable='d_max',
                                     mat_prop=Solid,
                                     inj_prop=Injection,
                                     fluid_prop=Fluid,
                                     fig=Fig_r,
                                     time_srs=time_srs,
                                     h=7.0,
                                     labels=label)
    label.legend = 'radius analytical (viscosity dominated)'
    plot_prop.lineColorAnal = 'b'
    Fig_r = plot_analytical_solution('M',
                                     variable='d_max',
                                     mat_prop=Solid,
                                     inj_prop=Injection,
                                     fig=Fig_r,
                                     fluid_prop=Fluid,
                                     time_srs=time_srs,
                                     plot_prop=plot_prop,
                                     labels=label)

    label = LabelProperties(variable='w', data_subset='point')
    plot_prop = PlotProperties(line_style='.',
                               graph_scaling='loglog')
    Fig_w = plot_fracture_list_at_point(Fr_list,  # plotting footprint
                                        variable='w',
                                        plot_prop=plot_prop,
                                        labels=label)

    label.legend = 'width at injection point (analytical PKN)'
    Fig_w = plot_analytical_solution_at_point('PKN',
                                              variable='w',
                                              mat_prop=Solid,
                                              inj_prop=Injection,
                                              fluid_prop=Fluid,
                                              fig=Fig_w,
                                              time_srs=time_srs,
                                              h=7.,
                                              labels=label)

    label.legend = 'width at injection point (viscosity dominated)'
    plot_prop.lineColorAnal = 'b'
    Fig_w = plot_analytical_solution_at_point('M',
                                              variable='w',
                                              mat_prop=Solid,
                                              inj_prop=Injection,
                                              fig=Fig_w,
                                              fluid_prop=Fluid,
                                              time_srs=time_srs,
                                              plot_prop=plot_prop,
                                              labels=label)

    # plotting in 3D
    Fr_list, properties = load_fractures(address="./Data/height_contained",
                                         time_srs=np.asarray([1, 5, 20, 50, 80, 110, 140]))
    time_srs = get_fracture_variable(Fr_list,
                                     variable='time')
    plot_prop_mesh = PlotProperties(text_size=1.7)
    Fig_Fr = plot_fracture_list(Fr_list,  # plotting mesh
                                variable='mesh',
                                projection='3D',
                                backGround_param='sigma0',
                                mat_properties=properties[0],
                                plot_prop=plot_prop_mesh)
    Fig_Fr = plot_analytical_solution('PKN',
                                      variable='footprint',
                                      mat_prop=Solid,
                                      inj_prop=Injection,
                                      fluid_prop=Fluid,
                                      fig=Fig_Fr,
                                      projection='3D',
                                      time_srs=time_srs[2:],
                                      h=7.0)
    plt_prop = PlotProperties(line_color_anal='b')
    Fig_Fr = plot_analytical_solution('M',
                                      variable='footprint',
                                      mat_prop=Solid,
                                      inj_prop=Injection,
                                      fluid_prop=Fluid,
                                      fig=Fig_Fr,
                                      projection='3D',
                                      time_srs=time_srs[:2],
                                      h=7.0,
                                      plot_prop=plt_prop)

    # Fig_Fr = None
    Fig_Fr = plot_fracture_list(Fr_list,  # plotting footprint
                                variable='footprint',
                                projection='3D',
                                fig=Fig_Fr)

    plot_prop = PlotProperties(alpha=0.2, text_size=5)  # plotting width
    Fig_Fr = plot_fracture_list(Fr_list,
                                variable='w',
                                projection='3D',
                                fig=Fig_Fr,
                                plot_prop=plot_prop)
    ax = Fig_Fr.get_axes()[0]
    ax.view_init(60, -114)

    plt.show(block=True)