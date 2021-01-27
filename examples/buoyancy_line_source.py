# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri June 16 17:49:21 2017.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2019. All rights
reserved. See the LICENSE.TXT file for more details.
"""

# imports
import os

# local imports
from mesh import CartesianMesh
from properties import MaterialProperties, FluidProperties, InjectionProperties, SimulationProperties
from fracture import Fracture
from controller import Controller
from fracture_initialization import Geometry, InitializationParameters, get_eliptical_survey_cells
from elasticity import load_isotropic_elasticity_matrix
from utility import setup_logging_to_console

# setting up the verbosity level of the log at console
setup_logging_to_console(verbosity_level='info')

# creating mesh
Mesh = CartesianMesh(115, 115, 51, 51)

# solid properties
nu = 0.4                            # Poisson's ratio
youngs_mod = 3.3e10                 # Young's modulus
Eprime = youngs_mod / (1 - nu ** 2) # plain strain modulus
K_Ic = 0.5                          # fracture toughness

def sigmaO_func(x, y):
    """ This function provides the confining stress over the domain"""

    # only dependant on the depth
    density = 2000
    return -(y - 400) * density * 9.8


# material properties
Solid = MaterialProperties(Mesh,
                           Eprime,
                           K_Ic,
                           confining_stress_func=sigmaO_func)

def source_location(x, y):
    """ This function is used to evaluate if a point is included in source, i.e. the fluid is injected at the given
        point.
    """
    tolerance = 2.
    # the condition
    return abs(x) < 75 and (y >= - 75. - tolerance and y <= -75. + tolerance)

# injection parameters
Q0 = 0.001  # injection rate
Injection = InjectionProperties(Q0, Mesh, source_loc_func=source_location)

# fluid properties
Fluid = FluidProperties(viscosity=1.1e-3, density=1000)

# simulation properties
simulProp = SimulationProperties()
simulProp.finalTime = 1.1e4                                   # the time at which the simulation stops
simulProp.set_outputFolder("./Data/buoyant_line_source")    # the disk address where the files are saved
simulProp.gravity = True                                    # take the effect of gravity into account
simulProp.set_mesh_extension_direction(['top'])
simulProp.plotVar = ['w', 'regime']
simulProp.toleranceEHL = 1e-3

# initializing fracture
surv_cells, _, inner_cells = get_eliptical_survey_cells(Mesh, 80, 20, center=[0.0, -75.0])
surv_cells_dist= surv_cells * 0 + (Mesh.hx ** 2 + Mesh.hy ** 2) ** (1/2)
Fr_geometry = Geometry(shape='level set',
                       survey_cells=surv_cells,
                       tip_distances=surv_cells_dist,
                       inner_cells=inner_cells)

C = load_isotropic_elasticity_matrix(Mesh, Eprime)
init_param = InitializationParameters(Fr_geometry,
                                      regime='static',
                                      net_pressure=5e4,
                                      elasticity_matrix=C)

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
    time_srs = np.linspace(1, 1e4, 7)
    Fr_list, properties = load_fractures(address="./Data/buoyant_line_source",
                                         time_srs=time_srs)
    time_srs = get_fracture_variable(Fr_list,
                                     variable='time')

    # plot footprint
    Fig_FP = plot_fracture_list(Fr_list,
                                variable='mesh',
                                projection='2D',
                                mat_properties=properties[0],
                                backGround_param='confining stress')
    Fig_FP = plot_fracture_list(Fr_list,
                                variable='footprint',
                                projection='2D',
                                fig=Fig_FP)

    # plot slice
    plot_prop = PlotProperties(line_style='.-')
    Fig_WS = plot_fracture_list_slice(Fr_list,
                                      variable='w',
                                      projection='2D',
                                      plot_prop=plot_prop,
                                      plot_cell_center=True,
                                      orientation='vertical')

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

    plt.show(block=True)