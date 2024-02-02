# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Andreas Möri on Sat Nov 25 11:41:33 2023.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2019.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# External imports
import os
import numpy as np
import logging

# local imports
from mesh_obj.mesh import CartesianMesh
from solid.solid_prop import MaterialProperties
from fluid.fluid_prop import FluidProperties
from properties import InjectionProperties, SimulationProperties
from fracture_obj.fracture import Fracture
from controller import Controller
from fracture_obj.fracture_initialization import Geometry, InitializationParameters
from utilities.utility import setup_logging_to_console
from utilities.postprocess_fracture import load_fractures

# setting up the verbosity level of the log at console
setup_logging_to_console(verbosity_level='info')
log = logging.getLogger('PyFrac.controller.run')

# creating mesh
Mesh = CartesianMesh(0.5, 0.5, 61, 61)

# solid properties
nu = 0.4                            # Poisson's ratio
youngs_mod = 3.3e10                 # Young's modulus
Eprime = youngs_mod / (1 - nu**2)   # plain strain modulus
K1c = 1e6                           # Fracture toughness
Cl = 5.875e-6                       # Carter's leak off coefficient

# material properties
Solid = MaterialProperties(Mesh,
                           Eprime,
                           K1c,
                           Carters_coef=Cl)

# injection parameters
Q0 = 0.01  # injection rate
ts = 2700 #
inj = np.asarray([[0.0, ts],
                 [Q0, 0]])  # injection rate
Injection = InjectionProperties(inj, Mesh)

# fluid properties
viscosity = 2.15e-3
Fluid = FluidProperties(viscosity=viscosity)

# value of the trajectory parameter phi
phi = (2 * Cl) ** 4 * Eprime ** 11 * (12 * viscosity) ** 3 * Q0 / ((32 / np.pi) ** (1/2) * K1c) ** 14
log.info("We calculate the trajectory parameter phi from Madyarova (2003) as:")
log.info("phi = " + str(phi))

# value of the closure parameters according to Peirce, (2022)
phi_v = ((Eprime ** 21 * (12 * viscosity) ** 5 * (2 * Cl) ** 10 * (Q0 * ts)) / ((32 / np.pi) ** (1/2) * K1c) ** 26) ** (9/65)
omega = ts / ((12 * viscosity) ** 4 * Q0 ** 6 / ((2 * Cl) ** 18 * Eprime ** 4)) ** (1/7)
log.info("And the parameters governing closure according to Peirce (202) as:")
log.info("phi^v = " + str(phi_v) + " and omega = " + str(omega))

# simulation properties
simulProp = SimulationProperties()
simulProp.finalTime = 1e4                           # the time at which the simulation stops
simulProp.saveTSJump, simulProp.plotTSJump = 1, 5   # save and plot after every 5 time steps
simulProp.set_outputFolder("./Data/MtoK_leakoff")   # the disk address where the files are saved
simname = 'phi_0o1'
simulProp.set_simulation_name(simname)              # set the name of the simulation
simulProp.plotVar = ['regime', 'w']

# initializing fracture
Fr_geometry = Geometry('radial')
init_param = InitializationParameters(Fr_geometry, regime='M', time=1e-3)

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
# Remove user interaction in case of batch_run
controller.PstvInjJmp = True if os.path.isfile('./batch_run.txt') else None
# run the simulation
controller.run()


####################
# plotting results #
####################

if not os.path.isfile('./batch_run.txt'): # We only visualize for runs of specific examples

    from utilities.visualization import *
    from utilities.postprocess_fracture import get_closure_geometry
    from scipy.optimize import curve_fit

    # loading simulation results
    Fr_list, properties = load_fractures("./Data/MtoK_leakoff",
                                         sim_name=simname)
    time_srs = get_fracture_variable(Fr_list,
                                     'time')

    # We obtain the fracture opening at the injection point
    # Note that we correct for the numerical minimum opening
    w0 = np.asarray(get_fracture_variable_at_point(Fr_list, variable="w", point=[properties[2].sourceCoordinates])[0])\
         - properties[0].wc

    # We obtain the evolution of the mean radius of the fracture
    r_mean = get_fracture_variable(Fr_list, variable='d_mean')

    # We also obtain all necessary information on fracture closure
    closure_data = get_closure_geometry(Fr_list, properties[2].sourceCoordinates)

    # We want to get the fracture radius during propagation, arrest, and closure. We obtain the moment when the
    # fracture starts receding.
    index_rec = np.absolute(time_srs - np.mean([closure_data['tr_breadth'], closure_data['tr_height']])).argmin()
    # and we combine the solutions
    r_tot = np.hstack((r_mean[:index_rec], closure_data['closing_radius_avg'][index_rec:]))

    # We also want to investigate fracture closure. To this end we calcualte several values form Peirce, (2022)
    t_prime = (closure_data['tc'] - np.asarray(time_srs[index_rec:])) / \
              np.mean([closure_data['tr_breadth'], closure_data['tr_height']])  # dimensinoless closure time

    # we want to plot the powerlaws
    index_closure = np.absolute(time_srs[index_rec:] - closure_data['tc']).argmin()
    len_fit = 5

    # We also fit the data to the square root for the radius
    # Define the square-root function
    def square_fit(x, a):
        y = a * x ** (1/2)
        return y
    # Define the linear funtion
    def line_fit(x, a):
        y = a * x
        return y

    # Make the fits.
    parameters_r, _ = curve_fit(square_fit, t_prime[index_closure - len_fit:index_closure],
                                r_tot[index_rec:][index_closure - len_fit:index_closure] / r_tot[index_rec])
    parameters_w, _ = curve_fit(line_fit, t_prime[index_closure - len_fit:index_closure],
                                w0[index_rec:][index_closure - len_fit:index_closure] / w0[index_rec])

    # plotting efficiency
    plot_prop = PlotProperties(graph_scaling='loglog',
                               line_style='.')
    label = LabelProperties('efficiency')
    label.legend = 'fracturing efficiency'
    Fig_eff = plot_fracture_list(Fr_list,
                               variable='efficiency',
                               plot_prop=plot_prop,
                               labels=label)


    # We plot the evolution of fracture radius and opening
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Plots of radius and opening evolution')
    fig.set_figwidth(10)
    ax1.scatter(time_srs, r_tot)
    ax1.set(xlabel='time since injection (s)', ylabel='fracture radius (m)')
    ax2.scatter(time_srs, w0 * 1000)  # to get the opening in [mm]
    ax2.set(xlabel='time since injection (s)', ylabel='fracture opening at centre wo (mm)')

    # We plot the emergence of the sunset solution by plotting the opening towards closure.
    fig2, (ax3, ax4) = plt.subplots(1, 2)
    fig2.suptitle('Plots of radius and opening evolution')
    fig2.set_figwidth(10)
    ax3.scatter(t_prime, r_tot[index_rec:] / r_tot[index_rec])
    ax3.plot(t_prime, square_fit(t_prime, parameters_r[0]), label='Fitted line')
    ax4.scatter(t_prime,  w0[index_rec:] / w0[index_rec])
    ax4.plot(t_prime, line_fit(t_prime, parameters_w[0]), 'r', label='Fitted line')
    ax4.set(xlabel='t`', ylabel='wo / wo at recession')
    ax3.set(xlabel='t`', ylabel='fracture radius / radius when recession starts')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_ylim([0.1, max(r_tot[index_rec:] / r_tot[index_rec]) * 1.1])

    plt.show(block=True)