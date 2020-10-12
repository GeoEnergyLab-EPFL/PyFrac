# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri June 16 17:49:21 2017.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2019.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# imports
import os

# local imports
from mesh import CartesianMesh
from properties import MaterialProperties, FluidProperties, InjectionProperties, SimulationProperties
from fracture import Fracture
from controller import Controller
from fracture_initialization import Geometry, InitializationParameters
from visualization import *
from utility import setup_logging_to_console

# setting up the verbosity level of the log at console
setup_logging_to_console(verbosity_level='info')

# creating mesh
Mesh = CartesianMesh(8, 4, 81, 41, symmetric=True)

# solid properties
nu = 0.4                            # Poisson's ratio
youngs_mod = 3.3e10                 # Young's modulus
Eprime = youngs_mod / (1 - nu ** 2) # plain strain modulus

# the function below will make the fracture propagate in the form of an ellipse (see Zia and Lecampion 2018)
def K1c_func(alpha):
    K1c_1 = 1.e6                    # fracture toughness along x-axis
    K1c_2 = 1.414e6                 # fracture toughness along y-axis

    beta = np.arctan((K1c_1 / K1c_2)**2 * np.tan(alpha))
    return 4 * (2/np.pi)**0.5 * K1c_2 * (np.sin(beta)**2 + (K1c_1 / K1c_2)**4 * np.cos(beta)**2)**0.25

Solid = MaterialProperties(Mesh,
                           Eprime,
                           anisotropic_K1c=True,
                           K1c_func=K1c_func)

# injection parameters
Q0 = 0.001  # injection rate
Injection = InjectionProperties(Q0, Mesh)

# fluid properties
Fluid = FluidProperties(viscosity=1.1e-5)

# simulation properties
simulProp = SimulationProperties()
simulProp.finalTime = 500               # the time at which the simulation stops
simulProp.set_volumeControl(True)       # to set up the solver in volume control mode (inviscid fluid)
simulProp.tolFractFront = 4e-3          # increase tolerance for the anisotropic case
simulProp.remeshFactor = 1.5            # the factor by which the mesh will be compressed.
simulProp.set_outputFolder("./Data/ellipse") # the disk address where the files are saved
simulProp.set_simulation_name('anisotropic_toughness_benchmark')
simulProp.symmetric = True              # solving with faster solver that assumes fracture is symmetric
simulProp.projMethod = 'ILSA_orig'
simulProp.set_tipAsymptote('U')

# initializing fracture
gamma = (K1c_func(np.pi/2) / K1c_func(0))**2    # gamma = (Kc1/Kc3)**2
Fr_geometry = Geometry('elliptical',
                       minor_axis=2.,
                       gamma=gamma)
init_param = InitializationParameters(Fr_geometry, regime='E_K')

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
    time_srs = np.geomspace(1, 500, 8)
    Fr_list, properties = load_fractures(address="./Data/ellipse",
                                        sim_name='anisotropic_toughness_benchmark',
                                        time_srs=time_srs)
    time_srs = get_fracture_variable(Fr_list,
                                     variable='time')

    # plotting footprint
    Fig_FP = plot_fracture_list(Fr_list,
                                variable='mesh',
                                projection='2D')
    Fig_FP = plot_fracture_list(Fr_list,
                                variable='footprint',
                                projection='2D',
                                fig=Fig_FP)
    Fig_FP = plot_analytical_solution('E_K',
                                      'footprint',
                                      Solid,
                                      Injection,
                                      fluid_prop=Fluid,
                                      fig=Fig_FP,
                                      projection='2D',
                                      time_srs=time_srs)

    #plotting width
    Fig_w_slice = plot_fracture_list_slice(Fr_list,
                                           variable='width',
                                           point1=[-Fr_list[-1].mesh.Lx, 0],
                                           point2=[Fr_list[-1].mesh.Lx, 0],
                                           plot_prop=PlotProperties(line_style='.'))
    Fig_w_slice = plot_analytical_solution_slice('E_K',
                                                 variable='width',
                                                 mat_prop=Solid,
                                                 inj_prop=Injection,
                                                 fluid_prop=Fluid,
                                                 fig=Fig_w_slice,
                                                 point1=[-Fr_list[-1].mesh.Lx, 0],
                                                 point2=[Fr_list[-1].mesh.Lx, 0],
                                                 time_srs=time_srs,
                                                 plt_top_view=True)

    # loading all fractures
    Fr_list, properties = load_fractures(address="./Data/ellipse",
                                         sim_name ='anisotropic_toughness_benchmark')
    time_srs = get_fracture_variable(Fr_list,
                                     variable='time')

    # making a plot properties object with desired line style and scaling
    plot_prop = PlotProperties(line_style='.',
                               graph_scaling='loglog')
    # plotting minor axis length
    labels = LabelProperties('d_min')
    labels.figLabel = 'Minor axis length'
    Fig_len_a = plot_fracture_list(Fr_list,
                                 variable='d_min',
                                 plot_prop=plot_prop,
                                 labels=labels)
    Fig_len_a = plot_analytical_solution('E_K',
                                       'd_min',
                                       Solid,
                                       Injection,
                                       fluid_prop=Fluid,
                                       fig=Fig_len_a,
                                       time_srs=time_srs,
                                       labels=labels)

    # plotting major axis length
    labels.figLabel = 'Major axis length'
    Fig_len_b = plot_fracture_list(Fr_list,
                                 variable='d_max',
                                 plot_prop=plot_prop,
                                 labels=labels)
    Fig_len_b = plot_analytical_solution('E_K',
                                       'd_max',
                                       Solid,
                                       Injection,
                                       fluid_prop=Fluid,
                                       fig=Fig_len_b,
                                       time_srs=time_srs,
                                       labels=labels)

    # plotting major axis length
    labels.figLabel = 'aspect ratio'
    plot_prop = PlotProperties(line_style = '.', graph_scaling='semilogx')
    Fig_ar = plot_fracture_list(Fr_list,
                                 variable='ar',
                                 plot_prop=plot_prop,
                                 labels=labels)
    ax = Fig_ar.get_axes()[0]
    ax.set_ylim(1.5, 2.5)

    plt.show(block=True)