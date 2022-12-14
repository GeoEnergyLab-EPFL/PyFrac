# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri June 16 17:49:21 2017.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2019. All rights
reserved. See the LICENSE.TXT file for more details.
"""

import numpy as np

# local imports
from src.mesh_obj.mesh import CartesianMesh
from src.solid.solid_prop import MaterialProperties
from src.fluid.fluid_prop import FluidProperties
from src.properties import InjectionProperties, SimulationProperties
from src.fracture_obj.fracture import Fracture
from src.controller import Controller
from src.fracture_obj.fracture_initialization import Geometry, InitializationParameters
from src.utilities.utility import setup_logging_to_console
from src.utilities.postprocess_fracture import load_fractures
from src.solid.elasticity_isotropic import load_isotropic_elasticity_matrix_toepliz
from utilities.visualization import *
from utilities.postprocess_fracture import *


# setting up the verbosity level of the log at console
setup_logging_to_console(verbosity_level='debug')

# creating mesh
Mesh = CartesianMesh(1.e-6, 1.e-6, 61, 61)

# solid properties
nu = 0.25                         # Poisson's ratio
youngs_mod = 5e10                  # Young's modulus
Eprime = youngs_mod / (1 - nu ** 2)             # plain strain modulus
K_Ic = 1e6         # Toughness

# injection parameters
Q0 = 1e-3    # injection rate
# ts = 1

delta_density = 50

fixed_Bks = 10
# fixed_Kms = 5e-3
fixed_Mkh = 1

# mup = K_Ic ** (18/5) * ts ** (2/5) / (Eprime ** (13/5) * Q0 ** (3/5) * fixed_Kms ** (18/5))
# delta_density = fixed_Bks * K_Ic ** (8/5) / (9.81 * Eprime ** (3/5) * Q0 ** (3/5) * ts ** (3/5))

mup = fixed_Mkh * K_Ic ** (14/3) / (Eprime ** 3 * Q0 * (delta_density * 9.81) ** (2/3))
ts = fixed_Bks ** (5/3) * K_Ic ** (8/3) / (Eprime * Q0 * (delta_density * 9.81) ** (5/3))

density_rock = 2700                                                 # density of the rock

def sigmaO_func(x, y):
    """ This function provides the confining stress over the domain"""
    Ly = 7.5e6
    return (Ly - y) * density_rock * 9.81

density_fluid = density_rock - delta_density

Injection = InjectionProperties(np.asarray([[0.0,  ts],
                                            [Q0,    0]]), Mesh)

# fluid properties
Fluid = FluidProperties(viscosity=mup/12, density=density_fluid)

# simulation properties
simulProp = SimulationProperties()
simulProp.finalTime = 1e50
simulProp.set_outputFolder("./Data/Dikes_Pulse") # the disk address where the files are saved
simulProp.gravity = True                    # set up the gravity flag
simulProp.plotTSJump = 1                 # plot every fourth time step
simulProp.saveTSJump = 5                    # save every second time step
simulProp.maxSolverItrs = 200
sim_name = "DI_174"
simulProp.set_simulation_name(sim_name)
simulProp.projMethod = 'LS_continousfront'
# custom plotting
class custom_factory():
    def __init__(self,  xlabel, ylabel):
        self.data = {'xlabel' : xlabel,
                     'ylabel': ylabel,
                     'xdata': [],
                     'ydata': [],
                     'xdata1': [],
                     'ydata1': []} # max value of x that can be reached during the simulation

    def custom_plot(self, sim_prop, fig=None):
        # this method is mandatory
        if fig is None:
            fig = plt.figure()
            ax = fig.gca()
        else:
            ax = fig.get_axes()[0]

        ax.scatter(self.data['xdata'], self.data['ydata'], color='b')
        ax.scatter(self.data['xdata1'], self.data['ydata1'], color='r')
        ax.set_xlabel(self.data['xlabel'])
        ax.set_ylabel(self.data['ylabel'])
        ax.set_yscale('log')
        ax.set_xscale('log')
        return fig

    def postprocess_fracture(self, sim_prop, fr):
        # this method is mandatory
        geom_data = get_fracture_geometric_parameters([fr])
        self.data['xdata'].append(fr.time)
        self.data['ydata'].append(geom_data['l'][0])
        self.data['xdata1'].append(fr.time)
        self.data['ydata1'].append(geom_data['bmax'][0])
        fr.postprocess_info = self.data
        return fr

simulProp.custom = custom_factory('t', 'l')
simulProp.customPlotsOnTheFly = True
simulProp.plotVar = ['custom', 'w']
simulProp.useBlockToeplizCompression = True
simulProp.solve_monolithic = False
simulProp.useHmat = True
simulProp.EHL_iter_lin_solve = True
simulProp.gmres_Restart = 1000
simulProp.gmres_maxiter = 1000
simulProp.Anderson_parameter = 50

simulProp.remeshFactor = 2
simulProp.set_mesh_extension_factor([1.1, 1.1, 1.1, 1.6])
simulProp.set_mesh_extension_direction(['top'])

# initializing a static fracture
t_init = 1e-15
K_init = K_Ic * t_init ** (1/9) / (Eprime ** (13/18) * Q0 ** (1/6) * mup ** (5/18))

Fr_geometry = Geometry('radial')
init_param = InitializationParameters(Fr_geometry,
                                      regime='M',
                                      time=t_init)


from utilities.visualization import *

# Decide on which simulation to load
sim_name = "DI_174_3"

dir = "./Data"

Fr_list, properties = load_fractures(address=dir,
                                     sim_name=sim_name)

Fr = Fr_list[-1]
if isinstance(Fr_list[-1].mesh, int):
    Fr.mesh = Fr_list[Fr_list[-1].mesh].mesh

Solid, Fluid, Injection, simulProp = properties
# simulation properties
simulProp.finalTime = 1e50
simulProp.saveToDisk = False
simulProp.set_outputFolder("./Data/Dikes_Pulse") # the disk address where the files are saved
simulProp.gravity = True                    # set up the gravity flag
simulProp.plotTSJump = 1                   # plot every fourth time step
simulProp.saveTSJump = 1                    # save every second time step
simulProp.maxSolverItrs = 200
simulProp.projMethod = 'LS_continousfront'

simulProp.plotVar = ["w"]
simulProp.useBlockToeplizCompression = True
simulProp.solve_monolithic = False
simulProp.useHmat = True
simulProp.EHL_iter_lin_solve = True
simulProp.gmres_Restart = 100
simulProp.gmres_maxiter = 100
simulProp.Anderson_parameter = 15

# remeshing
simulProp.set_mesh_extension_factor([1.15, 1.15, 1.05, 1.15])
simulProp.set_mesh_extension_direction(['all'])
simulProp.meshExtensionAllDir = True

# material properties
Solid = MaterialProperties(Fr.mesh,
                           properties[0].Eprime,
                           toughness=properties[0].K1c[0],
                           confining_stress_func=sigmaO_func,
                           minimum_width=1e-8)

# injection properties
Injection = InjectionProperties(properties[2].injectionRate,
                                Fr.mesh)

# create a Controller
## --- prepare the HMat parameters --- ##
HMATparam = [1000, 10, 1e-4]

C = load_isotropic_elasticity_matrix_toepliz(Fr.mesh, Solid.Eprime, C_precision=np.float64,
                                             useHMATdot=True, nu=.25,
                                             HMATparam=HMATparam)

controller = Controller(Fr,
                        Solid,
                        properties[1],
                        Injection,
                        simulProp,
                        C=C)

# run the simulation
controller.run()