# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri Dec 16 17:49:21 2017.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2019.
All rights reserved. See the LICENSE.TXT file for more details.
"""

import numpy as np

# local imports
from mesh import CartesianMesh
from properties import MaterialProperties, FluidProperties, InjectionProperties, SimulationProperties
from fracture import Fracture
from controller import Controller
from fracture_initialization import Geometry, InitializationParameters
from scipy import interpolate
from visualization import *

# creating mesh
Mesh = CartesianMesh(0.2, 0.2, 41, 41)

# solid properties
nu = 0.4                            # Poisson's ratio
youngs_mod = 3.3e10                 # Young's modulus
Eprime = youngs_mod / (1 - nu**2)   # plain strain modulus
K1c = 5e5 / (32 / np.pi)**0.5       # K' = 5e5
Cl = 0.5e-6                         # Carter's leak off coefficient

# material properties
Solid = MaterialProperties(Mesh,
                           Eprime,
                           K1c,
                           Carters_coef=Cl)

# injection parameters
Q0 = 0.01  # injection rate
Injection = InjectionProperties(Q0, Mesh)

# fluid properties
Fluid = FluidProperties(rheology='PLF', n=0.6, k=0.001 / 12)

# simulation properties
simulProp = SimulationProperties()
simulProp.finalTime = 1e7                               # the time at which the simulation stops
simulProp.set_outputFolder("./Data/MtoK_leakoff")       # the disk address where the files are saved
simulProp.set_simulation_name('PLF_MtoKtilde')
simulProp.tolFractFront = 0.003                         # increase the tolerance for faster run
simulProp.remeshFactor = 1.5                            # set the size of the new mesh to 1.5 times the previous size
simulProp.projMethod = 'LS_continousfront'              # using the continous front algorithm


# initializing the fracture width with the solution provided by  Madyarova & Detournay 2004 for power-law fluids. 
w = np.zeros(Mesh.NumberOfElts)
xw = np.genfromtxt('width_n_05.csv', delimiter=',')
t = 0.00005
n = Fluid.n
gamma = 0.699
Mprime = 2**(n + 1) * (2 * n + 1)**n / n**n * Fluid.k
Vel = 2 * (n + 1) / (n + 2) / 3 * gamma * (Eprime * Q0 ** (n + 2) / Mprime) ** (1 / (3 * n + 6)) / t ** ((n + 4) / (3 * n + 6))
eps = (Mprime / Eprime / t**n) ** (1 / (n + 2))
L = (Eprime * Q0**(n + 2) * t**(2 * n + 2) / Mprime) ** (1 / (3 * n + 6))
f = interpolate.interp1d(gamma * L * xw[:, 0], L * eps * xw[:, 1], bounds_error=False, fill_value='extrapolate')
w = f(Mesh.distCenter)
w[w < 0] = 0.
    
# initialization parameters
Fr_geometry = Geometry('radial', radius=gamma * L)
from elasticity import load_isotropic_elasticity_matrix
C = load_isotropic_elasticity_matrix(Mesh, Eprime)
init_param = InitializationParameters(Fr_geometry, regime='static', width=w, elasticity_matrix=C, tip_velocity=Vel)

# creating fracture object
Fr = Fracture(Mesh,
              init_param,
              Solid,
              Fluid,
              Injection,
              simulProp)
Fr.time = 5e-5

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
Fr_list, properties = load_fractures("./Data/MtoK_leakoff",
                                     sim_name='PLF_MtoKtilde',
                                      step_size=5
                                     )

plot_prop = PlotProperties(graph_scaling='loglog',
                            line_style='.',
                            line_color='green')
label = LabelProperties('d_mean')
label.legend = 'radius numerical'
Fig_r = plot_fracture_list(Fr_list,
                            variable='d_mean',
                            plot_prop=plot_prop,
                            labels=label)


# plot analytical radius
label = LabelProperties('d_mean')
label.legend = 'Newtonian M solution'
fluid_Newtonian = FluidProperties(viscosity = 0.001/12)
Fig_r = plot_analytical_solution(regime='M',
                                  variable='d_mean',
                                  mat_prop=Solid,
                                  inj_prop=Injection,
                                  fluid_prop=fluid_Newtonian,
                                  time_srs=np.geomspace(0.5, 1e3, 50),
                                  fig=Fig_r,
                                  plot_prop=plot_prop)

t = 2**np.linspace(np.log2(0.5),np.log2(1e7),30)
# solution taken from matlab code provided by Dontsov EV (2016)
r_analytical = np.asarray([0.0035, 0.0046, 0.0059, 0.0076, 0.0099, 0.0128, 0.0165, 0.0212, 0.0274, 0.0352, 0.0453,
                           0.0581, 0.0744, 0.0951, 0.1212, 0.1539, 0.1948, 0.2454, 0.3075, 0.3831, 0.4742, 0.5829,
                           0.7114, 0.8620, 1.0370, 1.2395, 1.4726, 1.7406, 2.0483, 2.4016])*1e3
ax_r = Fig_r.get_axes()[0]
ax_r.loglog(t, r_analytical, 'r-', label='semi-anlytical Newtonian M to K~')


time_srs_np = np.geomspace(1e3, 1e7, 50)
R = 2**(1/2) / np.pi * Q0**(1/2) * time_srs_np**(1/4) / Solid.Cprime[0]**(1/2)
ax_r.plot(time_srs_np, R, 'k', label='K~ solution')

n = 0.75
time_srs_np = np.geomspace(0.5, 1e4, 50)
Mprime = 2**(n + 1) * (2 * n + 1)**n / n**n  * Fluid.k
R = (Eprime * Q0**(n + 2) * time_srs_np**(2*n + 2) / Mprime)**(1 / (3*n + 6)) * 0.7092
ax_r.plot(time_srs_np, R, 'b', label='analytical Power law n=0.75')

n = 0.6
time_srs_np = np.geomspace(5e-5, 1e4, 50)
Mprime = 2**(n + 1) * (2 * n + 1)**n / n**n  * Fluid.k
R = (Eprime * Q0**(n + 2) * time_srs_np**(2*n + 2) / Mprime)**(1 / (3*n + 6)) * 0.7155
ax_r.plot(time_srs_np, R, 'g', label='analytical Power law n=0.6')
ax_r.legend()

# plt.show(block=True)

