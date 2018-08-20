# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri March 13 2018.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2017. All rights reserved.
See the LICENSE.TXT file for more details.
"""

# imports
from src.Fracture import *
from src.Controller import *


# creating mesh
Mesh = CartesianMesh(0.13, 0.17, 51, 67)

# solid properties
nu = 0.4                            # Poisson's ratio
youngs_mod = 3.3e9                  # Young's modulus
Eprime = youngs_mod / (1 - nu ** 2) # plain strain modulus
K_Ic = 10000                        # set toughness to a very low value

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
                           K_Ic,
                           SigmaO_func=sigmaO_func)

# injection parameters
Q0 = np.asarray([[0, 31, 151], [0.0009e-6, 0.0065e-6, 0.0023e-6]])
Injection = InjectionProperties(Q0, Mesh)

# fluid properties
Fluid = FluidProperties(viscosity=30)

# simulation properties
simulProp = SimulationParameters()
simulProp.outputTimePeriod = 0.1        # Setting it small so the file is saved after every time step
simulProp.bckColor = 'sigma0'           # the parameter according to which the background is color coded
simulProp.tmStpPrefactor = 0.5          # set the time step prefactor
simulProp.set_outputFolder('.\\Data\\Wu_et_al')
simulProp.set_solTimeSeries(np.asarray([22., 60., 144., 376., 665.]))

# initializing fracture
initRad = 0.014
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

# plotting fracture footprint
Fr_list, properties = load_fractures(address='.\\Data\\Wu_et_al',
                                     time_srs=simulProp.get_solTimeSeries())
Fig = plot_fracture_list(Fr_list,
                         variable='mesh',
                         backGround_param='sigma0',
                         mat_properties=Solid)
plot_prop = PlotProperties(line_color='r')
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
black_patch = mpatches.mlines.Line2D([], [], color='r', label='numerical')
plt.legend(handles=[blue_patch, black_patch])
ax.set_ylim(-170e-3, 50e-3,)

plt.show()