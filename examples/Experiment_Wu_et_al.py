# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri March 13 2018.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2019. All rights reserved.
See the LICENSE.TXT file for more details.
"""

# imports
from src.Fracture import *
from src.Controller import *


# creating mesh
Mesh = CartesianMesh(0.065, 0.085, 31, 41)

# solid properties
nu = 0.4                            # Poisson's ratio
youngs_mod = 3.3e9                  # Young's modulus
Eprime = youngs_mod / (1 - nu ** 2) # plain strain modulus

def sigmaO_func(x, y):
    """ The function providing the confining stress"""
    if y > 0.025:
        return 11.2e6
    elif y < -0.025:
        return 5.0e6
    else:
        return 7.0e6

Solid = MaterialProperties(Mesh,
                           Eprime,
                           confining_stress_func=sigmaO_func,
                           minimum_width=1e-8)

# injection parameters
Q0 = np.asarray([[0, 31, 151], [0.0009e-6, 0.0065e-6, 0.0023e-6]])
Injection = InjectionProperties(Q0, Mesh)

# fluid properties
Fluid = FluidProperties(viscosity=30)

# simulation properties
simulProp = SimulationProperties()
simulProp.bckColor = 'confining stress'           # the parameter according to which the background is color coded
simulProp.frontAdvancing = 'explicit'
simulProp.set_outputFolder('./Data/Wu_et_al')
simulProp.set_solTimeSeries(np.asarray([22., 60., 144., 376., 665.]))

# initializing fracture
initRad = 0.019
init_param = ('M', "length", initRad)

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

# loading the experiment data file
import csv
with open('./wu_et_al_data.csv', 'r') as f:
    reader = csv.reader(f)
    data = np.asarray(list(reader), dtype=np.float64)


####################
# plotting results #
####################

# plotting fracture footprint
Fr_list, properties = load_fractures(address='./Data/Wu_et_al',
                                     time_srs=simulProp.get_solTimeSeries())
Fig = plot_fracture_list(Fr_list,
                         variable='mesh',
                         backGround_param='sigma0',
                         mat_properties=Solid)
plot_prop = PlotProperties(line_color='darkmagenta')
Fig = plot_fracture_list(Fr_list,
                         variable='footprint',
                         fig=Fig,
                         plot_prop=plot_prop)

# plotting footprint from experiment
ax = Fig.get_axes()[0]
ax.plot(data[:, 0]*1e-3, -1e-3*data[:, 1], 'k')
ax.plot(data[:, 2]*1e-3, -1e-3*data[:, 3], 'k')
ax.plot(data[:, 4]*1e-3, -1e-3*data[:, 5], 'k')
ax.plot(data[:, 6]*1e-3, -1e-3*data[:, 7], 'k')
ax.plot(data[:, 8]*1e-3, -1e-3*data[:, 9], 'k')

blue_patch = mpatches.mlines.Line2D([], [], color='k', label='experiment (Wu et al. 2008)')
black_patch = mpatches.mlines.Line2D([], [], color='darkmagenta', label='numerical')
plt.legend(handles=[blue_patch, black_patch])
ax.set_ylim(-170e-3, 50e-3,)

plt.show(block=True)