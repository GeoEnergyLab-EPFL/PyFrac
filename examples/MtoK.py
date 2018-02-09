# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri June 16 17:49:21 2017.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2017. All rights
reserved. See the LICENSE.TXT file for more details.
"""

# imports
from src.Fracture import *
from src.Controller import *
from src.PostProcess import *

# creating mesh
Mesh = CartesianMesh(2., 2., 41, 41)

# solid properties
nu = 0.4                            # Poisson's ratio
youngs_mod = 3.3e10                 # Young's modulus
Eprime = youngs_mod / (1 - nu ** 2) # plain strain modulus
K_Ic = 1.5e6                          # fracture toughness

Solid = MaterialProperties(Mesh,
                           Eprime,
                           K_Ic)

# injection parameters
Q0 = 0.01  # injection rate
Injection = InjectionProperties(Q0, Mesh)

# fluid properties
Fluid = FluidProperties(viscosity=1.1e-3)

# simulation properties
simulProp = SimulationParameters()
simulProp.FinalTime = 1e7               # the time at which the simulation stops
simulProp.plotFigure = False            # to disable plotting of figures while the simulation runs
simulProp.saveToDisk = True             # to enable saving the results (to hard disk)
simulProp.set_outFileAddress(".\\Data\\MtoK") # the disk address where the files are saved
simulProp.saveRegime = True             # enable saving the regime

# initializing fracture
initRad = 0.5
init_param = ("M", "length", initRad)

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
# controller.run()

# plotting footprint
plot_footprint(simulProp.get_outFileAddress(),
                        sol_t_srs=simulProp.get_solTimeSeries(),
                        analytical_sol='K')

# plotting radius
Fig_r, Fig_err = plot_radius(simulProp.get_outFileAddress(),
                            analytical_sol='M',
                            plt_error=False)
plot_radius(simulProp.get_outFileAddress(),
                            fig_r=Fig_r,
                            analytical_sol='K',
                            anltcl_lnStyle='g',
                            add_labels=False,
                            plt_error=False)

# plotting width at center
Fig_w, Fig_p = plot_at_injection_point(simulProp.get_outFileAddress(),
                            plt_pressure=False,
                            analytical_sol='M',
                            plt_error=False)
plot_at_injection_point(simulProp.get_outFileAddress(),
                            fig_w=Fig_w,
                            plt_pressure=False,
                            analytical_sol='K',
                            anltcl_lnStyle='g',
                            plt_error=False)

# plotting regime
plot_footprint(simulProp.get_outFileAddress(),
                        sol_t_srs=1.1*simulProp.get_solTimeSeries(),
                        plt_mesh=False,
                        plt_regime=True)
plt.show()