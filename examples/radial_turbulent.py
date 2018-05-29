# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri June 16 17:49:21 2017.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2017. All rights reserved.
See the LICENSE.TXT file for more details.
"""

# imports
from src.Fracture import *
from src.Controller import *
from src.PostProcess import *
from src.PostProcessFracture import *

# creating mesh
Mesh = CartesianMesh(3, 3, 41, 41)

# solid properties
nu = 0.                            # Poisson's ratio
youngs_mod = 1.0e10                 # Young's modulus
Eprime = youngs_mod / (1 - nu ** 2) # plain strain modulus
K_Ic = 0.5e6                          # fracture toughness

Solid = MaterialProperties(Mesh,
                           Eprime,
                           K_Ic)

# injection parameters
Q0 = 0.5  # massive injection rate
Injection = InjectionProperties(Q0, Mesh)

# fluid properties
Fluid = FluidProperties(viscosity=1.e-3,
                        turbulence=True)

# simulation properties
simulProp = SimulationParameters()
simulProp.FinalTime = 0.05             # the time at which the simulation stops
simulProp.set_tipAsymptote('U')         # the tip asymptote is evaluated with the toughness dominated assumption
simulProp.frontAdvancing = 'implicit'   # to set explicit front tracking
simulProp.outputTimePeriod = 1e-4       # to save after every time step
simulProp.set_outFileAddress("./Data/MDR") # the disk address where the files are saved
simulProp.plotFigure = True
simulProp.plotAnalytical = True
simulProp.analyticalSol = 'M'
simulProp.saveReynNumb = True
simulProp.verbosity = 2

# initializing fracture
initRad = 1
init_param = ("M", "length", initRad)

# initRad = 1.
# surv_cells, inner_cells = get_circular_survey_cells(Mesh, initRad)
# surv_cells_dist = initRad  - (abs(Mesh.CenterCoor[surv_cells,1])**2+abs(Mesh.CenterCoor[surv_cells,0])**2)**0.5
# C = load_elasticity_matrix(Mesh, Eprime)
#
# init_param = ('G',              # type of initialization
#               surv_cells,       # the given survey cells
#               inner_cells,      # the cell enclosed by the fracture
#               surv_cells_dist,  # the distance of the survey cells from the front
#               None,             # the given width
#               50,               # the pressure (uniform in this case)
#               C,                # the elasticity matrix
#               None,             # the volume of the fracture
#               0)                # the velocity of the propagating front (stationary in this case)




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


# plot results
Fr_list = load_fractures(simulProp.get_outFileAddress(), multpl_sim=True,
               time_srs=np.linspace(0.0028,0.031,10))


fig1,fig2=plot_radius(simulProp.get_outFileAddress(), multpl_sim=True,
                        r_type='mean',
                        analytical_sol='M',add_labels=False,anltcl_lnStyle='r')

plot_radius(simulProp.get_outFileAddress(), multpl_sim=True,
                        r_type='mean',fig_r=fig1,fig_err=fig2,
                        analytical_sol='MDR',add_labels=False)
plt.show()


t, R, p, w, v, actvElts =MDR_M_vertex_solution(Eprime,Q0,density=Fluid.density,visc=Fluid.viscosity,Mesh=Mesh,t=Fr_list[5].time)

for ff in Fr_list:
    #plot_Reynolds_number(ff,1810)
    #plot_pressure_contour(ff)
    plot_width_contour(ff,bck_colMap='jet')

    plt.show()


plot_footprint(simulProp.get_outFileAddress(),         # the address where the results are stored
                        plot_at_times=np.linspace(0.0028,0.031,7),# the time series at which the solution is plotted
                        analytical_sol='M')            # analytical solution for reference
plt.show()


plot_radius(simulProp.get_outFileAddress(), multpl_sim=True,
                        r_type='mean',
                        analytical_sol='MDR')
plt.show()

plot_at_injection_point(simulProp.get_outFileAddress(), multpl_sim=True,
                        plt_pressure=False,
                        analytical_sol='MDR')

plt.show()


plot_profile(simulProp.get_outFileAddress(),multpl_sim=True,
                        plot_at_times=np.linspace(0.0028,0.031,5),
                        analytical_sol='MDR') #,plt_pressure=True
plt.show()






plot_profile(simulProp.get_outFileAddress(),
                        plot_at_times=np.linspace(0.0028,0.031,5), plt_symbol=".-b")
plt.show()


plot_profile(simulProp.get_outFileAddress(), plt_symbol=".-b")
plt.show()


