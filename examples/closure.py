# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri Dec 16 17:49:21 2017.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2017. All rights
reserved. See the LICENSE.TXT file for more details.
"""

# imports
from src.Fracture import *
from src.Controller import *

# creating mesh
Mesh = CartesianMesh(16, 16, 41, 41)

# solid properties
nu = 0.4                            # Poisson's ratio
youngs_mod = 3.3e10                 # Young's modulus
Eprime = youngs_mod / (1 - nu**2)   # plain strain modulus
K1c = 5e5 / (32 / math.pi)**0.5     # K' = 5e5
Cl = np.full((Mesh.NumberOfElts,), 1.0e-6, dtype=np.float64)

Solid = MaterialProperties(Mesh,
                           Eprime,
                           K1c,
                           Cl=Cl)

# injection parameters
# Q0 = 0.01  # injection rate
Q0 = np.asarray([[0, 15., 1415.], [0.01, 0, 0.01]])
Injection = InjectionProperties(Q0, Mesh)

# fluid properties
viscosity = 0.001 / 12  # mu' =0.001
Fluid = FluidProperties(viscosity=viscosity)

# simulation properties
simulProp = SimulationParameters()
simulProp.FinalTime = 1500               # the time at which the simulation stops
simulProp.outputEveryTS = 1             # the time after the output is generated (saving or plotting)
simulProp.set_outputFolder(".\\Data\\closure") # the disk address where the files are saved
simulProp.verbosity = 2
simulProp.plotFigure = True
simulProp.timeStepLimit = 100.
simulProp.fixedTmStp = np.asarray([[0, 1415, 1440], [None, 3, None]])


# initializing fracture
initTime = 10
init_param = ("M", "time", initTime)

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

# plotting efficiency
Fr_list, properties = load_fractures(".\\Data\\closure")
time_srs = get_fracture_variable(Fr_list,
                                 'time')


p_prop = PlotProperties(line_style='.', graph_scaling='loglog')
Fig_p = plot_fracture_list_at_point(Fr_list,
                   variable='p',
                   plot_prop=p_prop)
ax = Fig_p.get_axes()[0]
ax.set_ylim((-.4, .8))

plt.show()
