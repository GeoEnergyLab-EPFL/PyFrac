# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri June 16 17:49:21 2017.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2019. All rights
reserved. See the LICENSE.TXT file for more details.
"""

# local imports
from mesh import CartesianMesh
from properties import MaterialProperties, FluidProperties, InjectionProperties, SimulationProperties
from fracture import Fracture
from controller import Controller
from fracture_initialization import Geometry, InitializationParameters


# creating mesh
Mesh = CartesianMesh(100, 150, 41, 61)

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
    # the condition
    return abs(x) < 75. and (y > -80. and y < -74)

# injection parameters
Q0 = 0.001  # injection rate
Injection = InjectionProperties(Q0, Mesh, source_loc_func=source_location)

# fluid properties
Fluid = FluidProperties(viscosity=1.1e-3, density=1000)

# simulation properties
simulProp = SimulationProperties()
simulProp.finalTime = 6000              # the time at which the simulation stops
simulProp.set_outputFolder("./Data/M_radial_explicit") # the disk address where the files are saved
simulProp.gravity = True                # take the effect of gravity into account

# initialization parameters
Fr_geometry = Geometry(shape='height contained',
                       fracture_length=80,
                       fracture_height=35)
init_param = InitializationParameters(Fr_geometry, regime='PKN')

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
time_srs = np.linspace(1, 6000, 5)
Fr_list, properties = load_fractures(address="./Data/M_radial_explicit",
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

#plt.show(block=True)
#  set block=True and comment last 2 lines if you want to keep the window open
plt.show(block=False)
plt.pause(5)
plt.close()

