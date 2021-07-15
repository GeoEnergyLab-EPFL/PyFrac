# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo on Fri Apr 17 23:16:25 2020.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""


# post processing function

def get_info(Fr_list_A):  # get L(t) and x_max(t) and p(t)
    double_L_A = [];
    x_max_A = [];
    p_A = [];
    w_A=[];
    time_simul_A = [];
    center_indx = (Fr_list_A[0]).mesh.locate_element( 0., 0.)
    for frac_sol in Fr_list_A:
        # we are at a give time step now,
        # I am getting double_L_A, x_max_A
        x_max_temp = 0
        y_min_temp = 0
        y_max_temp = 0
        for i in range(frac_sol.Ffront.shape[0]):
            segment = frac_sol.Ffront[i]
            # to find the x_max at this time:
            if segment[0] > x_max_temp:
                x_max_temp = segment[0]
            if segment[2] > x_max_temp:
                x_max_temp = segment[2]
            # to find the y_max at this time:
            if segment[1] > y_max_temp:
                y_max_temp = segment[1]
            if segment[3] > y_max_temp:
                y_max_temp = segment[3]
            # to find the y_min at this time:
            if segment[1] < y_min_temp:
                y_min_temp = segment[1]
            if segment[3] < y_min_temp:
                y_min_temp = segment[3]

        double_L_A.append(y_max_temp - y_min_temp)
        x_max_A.append(x_max_temp)

        p_A.append(frac_sol.pFluid.max() / 1.e6)
        w_A.append(frac_sol.w[center_indx])
        time_simul_A.append(frac_sol.time)
    return double_L_A, x_max_A, p_A,w_A,  time_simul_A



# imports
import os
import time
import numpy as np
from datetime import datetime

# local imports
from mesh import CartesianMesh
from properties import MaterialProperties, FluidProperties, InjectionProperties, SimulationProperties
from fracture import Fracture
from controller import Controller
from fracture_initialization import Geometry, InitializationParameters
from utility import append_new_line
from pypart import Bigwhamio
import math
from utility import setup_logging_to_console
from Hdot import Hdot_3DR0opening
from Hdot import gmres_counter

# setting up the verbosity level of the log at console
setup_logging_to_console(verbosity_level='debug')

########## OPTIONS #########
run = True
use_iterative = True
use_HMAT = True
use_direct_TOEPLITZ = False
############################
if run:
    # creating mesh
    Mesh = CartesianMesh(3, 18, 151, 901)

    # solid properties
    nu = 0.4  # Poisson's ratio
    youngs_mod = 3.3e10  # Young's modulus
    Eprime = youngs_mod / (1 - nu ** 2)  # plain strain modulus
    properties = [youngs_mod, nu]


    def smoothing(K1, K2, r, delta, x, Lx):
        # instead of having -10/10, take the MESHNAME.Ly/Lx (if mesh square)
        #### LINEAR ####
        x = np.abs(x)
        a = (K2 - K1) / ((r + delta) - (r - delta))
        b = K2 - a * (r + delta)
        if -1.1* Lx <= x < r - delta:
            return K1
        elif r - delta <= x <= r + delta:
            return a * x + b
        elif r + delta < x <= 1.1 *Lx:
            return K2
        else:
            print("ERROR")
        #### EXPONENTIAL ####
        # x_b = 400*(np.abs(x)-r)
        # if x >= 0.:
        #     return K1 + np.exp(x_b)/(np.exp(x_b)+1)*(K2-K1)
        # else:
        #     return K1 + np.exp(x_b)/(np.exp(x_b)+1)*(K2-K1)


    def K1c_func(x,y):
        """ The function providing the toughness"""
        K_Ic = 0.5e6  # fracture toughness
        r = 1.5
        delta = 0.005
        Lx = 10
        # if np.abs(x) > r:
        #     return 2.*K_Ic
        # else:
        #     return K_Ic
        return smoothing(K_Ic, 1.9*K_Ic, r, delta, x, Lx)

    # def K1c_func(x, y):
    #     """ The function providing the toughness
    #
    #     It consist of a periodic layer of constant height H characterized either by a toughness of Kmax or Kmin.
    #     A smoothing between the two values is made over a distance epsilon.
    #
    #     """
    #     K_Ic_min = 0.5e6  # fracture toughness
    #     K_Ic_max = 1.2 * K_Ic_min  # fracture toughness
    #     H = 1.5 # Height of the layer
    #     epsilon = 0.1 * H  # linear smoothing distance
    #     delta = K_Ic_max - K_Ic_min
    #     Kmed = (K_Ic_max + K_Ic_min)/2.
    #     localvar = np.sin(np.pi * x / H)
    #
    #     if np.abs(localvar) > np.sin(np.pi * (epsilon + H/2.) / H) :
    #         y_jump = Kmed
    #         if ((x+H/2.) % H) > 0.5:
    #             x_jump = ((x+H/2.) // H + 1) * H - H/2.
    #         else:
    #             x_jump = ((x + H / 2.) // H) * H - H / 2.
    #         m = np.sign(localvar) * delta / 2. / epsilon
    #         q = y_jump - m * x_jump
    #         return m * x + q
    #     else:
    #         return Kmed + delta/2. * np.sign(np.sin(np.pi * x / H - np.pi/2.))


    def sigmaO_func(x, y):
        return 0
        # """ The function providing the stress"""
        # if (np.floor(abs(y)) % 3) > 1 and abs(y) >0.:
        #     return 48e6
        # else:
        #     return 36e6
        # # comment the following section if you would like to consider field of stress
        # # caracterized by the presence of less heterogeneities.
        # lx = 0.20
        # ly = 0.20
        # if math.trunc(abs(x) / lx) >0:
        #     if math.trunc(abs(x) / lx) %2 == 0:
        #         x = abs(x) - (math.trunc(abs(x) / lx)) * lx
        #     else :
        #         x = abs(x) - (math.trunc(abs(x) / lx) + 1) * lx
        #
        # if math.trunc(abs(y) / ly) > 0:
        #     if math.trunc(abs(y) / ly) %2 == 0:
        #         y = abs(y) - (math.trunc(abs(y) / ly)) * ly
        #     else :
        #         y = abs(y) - (math.trunc(abs(y) / ly)+1) * ly
        # # comment up to here
        #
        #
        # """ The function providing the confining stress"""
        # R=0.05
        # x1=0.
        # y1=0.2
        #
        # if (abs(x)-x1)**2+(abs(y)-y1)**2 < R**2:
        #    return 60.0e6
        # if (abs(x)-y1)**2+(abs(y)-x1)**2 < R**2:
        #    return 60.0e6
        # else:
        #    return 5.0e6

    Solid = MaterialProperties(Mesh,
                              Eprime,
                              K1c_func=K1c_func,
                              confining_stress_func = sigmaO_func,
                              confining_stress=0.,
                              minimum_width=0.)

    if use_iterative:
        if use_HMAT:
            # set the Hmatrix for elasticity
            begtime_HMAT = time.time()
            C = Hdot_3DR0opening()
            max_leaf_size = 100
            eta = 10
            eps_aca = 0.001
            data = [max_leaf_size, eta, eps_aca, properties, Mesh.VertexCoor, Mesh.Connectivity, Mesh.hx, Mesh.hy]
            C.set(data)
            endtime_HMAT = time.time()
            compute_HMAT = endtime_HMAT - begtime_HMAT
            append_new_line('./Data/radial_VC_gmres/building_HMAT.txt', str(compute_HMAT))
            print("Compression Ratio of the HMAT : ", C.compressionratio)

        else:
            from elasticity import load_isotropic_elasticity_matrix_toepliz
            C = load_isotropic_elasticity_matrix_toepliz(Mesh, Eprime)

    # injection parameters
    Q0 = 0.001
    Injection = InjectionProperties(Q0, Mesh)

    # fluid properties
    Fluid = FluidProperties(viscosity=1.1e-6)

    # simulation properties
    simulProp = SimulationProperties()
    simulProp.finalTime = 105.12  # the time at which the simulation stops
    simulProp.tmStpPrefactor = 0.9  # decrease the pre-factor due to explicit front tracking
    simulProp.gmres_tol = 1e-15
    simulProp.saveToDisk = True
    simulProp.set_volumeControl(True)
    if use_iterative: simulProp.volumeControlGMRES = True
    simulProp.bckColor = 'K1c'
    #simulProp.bckColor = 'sigma0'
    simulProp.set_outputFolder("./Data/radial_VC_gmres")  # the disk address where the files are saved
    simulProp.set_tipAsymptote('K')  # the tip asymptote is evaluated with the toughness dominated assumption
    simulProp.plotVar = ['footprint', 'custom']#,'regime']
    simulProp.frontAdvancing = 'implicit'  # <--- mandatory use
    simulProp.projMethod = 'LS_continousfront'  # <--- mandatory use
    simulProp.set_solTimeSeries( np.concatenate((np.arange(0.,0.3,0.01),np.arange(0.3,105.12,0.05))))
    simulProp.force_time_schedule = True

    simulProp.customPlotsOnTheFly = True
    simulProp.LHyst__ = []
    simulProp.tHyst__ = []

    # initialization parameters
    Fr_geometry = Geometry('radial', radius=0.8)

    if not simulProp.volumeControlGMRES:
        if use_direct_TOEPLITZ:
            simulProp.useBlockToeplizCompression = True
            from elasticity import load_isotropic_elasticity_matrix_toepliz

            C = load_isotropic_elasticity_matrix_toepliz(Mesh, Eprime)
        else:
            from elasticity import load_isotropic_elasticity_matrix

            C = load_isotropic_elasticity_matrix(Mesh, Eprime)

    init_param = InitializationParameters(Fr_geometry, regime='K')

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
                            simulProp,
                            C=C)

    # run the simulation
    controller.run()
####################
# plotting results #
####################

if not os.path.isfile('./batch_run.txt'):  # We only visualize for runs of specific examples

    from visualization import *

    # # loading simulation results
    Fr_list, properties = load_fractures(address="./Data/radial_VC_gmres",step_size=1)                  # load all fractures
    time_srs = get_fracture_variable(Fr_list, variable='time')                                                 # list of times
    Solid, Fluid, Injection, simulProp = properties


    # plot fracture radius
    my_list = []
    for i in np.arange(0,len(Fr_list),10):
        my_list.append(Fr_list[i])
    plot_prop = PlotProperties()
    Fig_R = plot_fracture_list(my_list,
                               variable='footprint',
                               plot_prop=plot_prop)
    Fig_R = plot_fracture_list(my_list,
                               fig=Fig_R,
                               variable='mesh',
                               mat_properties=properties[0],
                               backGround_param='K1c',
                               plot_prop=plot_prop)

    #
    # # plot fracture radius
    # plot_prop = PlotProperties()
    # plot_prop.lineStyle = '.'               # setting the linestyle to point
    # plot_prop.graphScaling = 'loglog'       # setting to log log plot
    # Fig_R = plot_fracture_list(Fr_list,
    #                            variable='d_mean',
    #                            plot_prop=plot_prop)
    #
    # # plot analytical radius
    # Fig_R = plot_analytical_solution(regime='K',
    #                                  variable='d_mean',
    #                                  mat_prop=Solid,
    #                                  inj_prop=Injection,
    #                                  fluid_prop=Fluid,
    #                                  time_srs=time_srs,
    #                                  fig=Fig_R)
    #
    # # # plot width at center
    # Fig_w = plot_fracture_list_at_point(Fr_list,
    #                                     variable='w',
    #                                     plot_prop=plot_prop)
    # # # plot analytical width at center
    # Fig_w = plot_analytical_solution_at_point('K',
    #                                           'w',
    #                                           Solid,
    #                                           Injection,
    #                                           fluid_prop=Fluid,
    #                                           time_srs=time_srs,
    #                                           fig=Fig_w)
    #
    #
    # Fig_pf = plot_fracture_list_at_point(Fr_list,
    #                                     variable='pn',
    #                                     plot_prop=plot_prop)
    #
    # Fig_pf = plot_analytical_solution_at_point('K',
    #                                           'pn',
    #                                           Solid,
    #                                           Injection,
    #                                           fluid_prop=Fluid,
    #                                           time_srs=time_srs,
    #                                           fig=Fig_pf)
    #
    #
    # # plot slice
    # #ext_pnts = np.empty((2, 2), dtype=np.float64)
    # my_X = 0.
    # my_Y = 0.
    # ext_pnts = np.empty((2, 2), dtype=np.float64)
    # Fig_WS = plot_fracture_list_slice(Fr_list,
    #                                   variable='w',
    #                                   projection='2D',
    #                                   plot_cell_center=True,
    #                                   extreme_points=ext_pnts,
    #                                   orientation='horizontal',
    #                                   point1=[my_X, my_Y]
    #                                   )

    #################################

    # loading simulation results A
    Fr_list_A, properties_A = load_fractures(address="./Data/radial_VC_gmres",step_size=1)                  # load all fractures
    time_srs_A = get_fracture_variable(Fr_list_A, variable='time')                                                 # list of times
    Solid_A, Fluid_A, Injection_A, simulProp_A = properties_A
    double_L_A, x_max_A, p_A, w_A,  time_simul_A = get_info(Fr_list_A)

    print("\n get w(x) with x passing through a specific point for different times... ")
    # my_X = 0.
    # my_Y = 0.
    # ext_pnts = np.empty((2, 2), dtype=np.float64)
    # fracture_list_slice_A = plot_fracture_list_slice(Fr_list_A,
    #                                                variable='w',
    #                                                projection='2D',
    #                                                plot_cell_center=True,
    #                                                extreme_points=ext_pnts,
    #                                                orientation='horizontal',
    #                                                point1=[my_X, my_Y],
    #                                                export2Json=True,
    #                                                export2Json_assuming_no_remeshing=True)
    # loading simulation results B
    # Fr_list_B, properties_B = load_fractures(address="./Data/sim_B",step_size=1)                  # load all fractures
    # time_srs_B = get_fracture_variable(Fr_list_A, variable='time')                                                 # list of times
    # Solid_B, Fluid_B, Injection_B, simulProp_B = properties_B
    # double_L_B, x_max_B, p_B, time_simul_B = get_info(Fr_list_B)
    # # plot slice
    # print("\n get w(x) with x passing through a specific point for different times... ")
    # my_X = 0.
    # my_Y = 0.
    # ext_pnts = np.empty((2, 2), dtype=np.float64)
    # fracture_list_slice_B = plot_fracture_list_slice(Fr_list_B,
    #                                                variable='w',
    #                                                projection='2D',
    #                                                plot_cell_center=True,
    #                                                extreme_points=ext_pnts,
    #                                                orientation='horizontal',
    #                                                point1=[my_X, my_Y],
    #                                                export2Json=True,
    #                                                export2Json_assuming_no_remeshing=True)

    #### BUILDING OUR PLOTS ###
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # plot the pressure vs time
    xlabel = 'time [s]'
    ylabel = 'Pressure [MPa]'
    fig, ax = plt.subplots()
    ax.scatter(time_simul_A, p_A, color='k')
    p_ana = []
    for i in range(len(time_simul_A)):
        p_ana.append(0.35)
    ax.plot(time_simul_A, p_ana, color='g')
    # ax.scatter(time_simul_B, p_B, color='g')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.show()

    # plot 2L vs time
    xlabel = 'time [s]'
    ylabel = '2L [m]'
    fig, ax = plt.subplots()
    ax.scatter(time_simul_A, double_L_A, color='k')
    doubleL_scaling = []
    for i in range(len(time_simul_A)):
        doubleL_scaling.append(10*time_simul_A[i]**(2/3))
    ax.plot(time_simul_A, doubleL_scaling, color='g')

    doubleL_radial = []
    for i in range(len(time_simul_A)):
        doubleL_radial.append(10*time_simul_A[i]**(2/5))
    ax.plot(time_simul_A, doubleL_radial, color='b')
    # ax.scatter(time_simul_B, p_B, color='g')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.show()


    # plot x_max vs time
    xlabel = 'time [s]'
    ylabel = 'x max [m]'
    fig, ax = plt.subplots()
    ax.scatter(time_simul_A, x_max_A, color='k')
    p_const = []
    for i in range(len(time_simul_A)):
        p_const.append(1.6)
    ax.plot(time_simul_A, p_const, color='g')
    # ax.scatter(time_simul_B, p_B, color='g')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.show()


    # plot w_cemter vs time
    xlabel = 'time [s]'
    ylabel = 'w center [m]'
    fig, ax = plt.subplots()
    ax.scatter(time_simul_A, w_A, color='k')
    w_scaling = []
    w_radial = []
    for i in range(len(time_simul_A)):
        w_scaling.append(0.0000625*time_simul_A[i]**(1/3))
    ax.plot(time_simul_A, w_scaling, color='g')

    for i in range(len(time_simul_A)):
        w_radial.append(0.0000525*time_simul_A[i]**(1/5))
    ax.plot(time_simul_A, w_radial, color='r')
    # ax.scatter(time_simul_B, p_B, color='g')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale('log')
    ax.set_xscale('log')


    V = []
    for fr in Fr_list_A:
        V.append((np.sum(fr.w)*fr.mesh.hx*fr.mesh.hy-fr.time*Injection_A.injectionRate.max())/fr.time*Injection_A.injectionRate.max())
    xlabel = 'time [s]'
    ylabel = 'volume res [m]'
    fig, ax = plt.subplots()
    ax.scatter(time_simul_A, V, color='k')
    w_scaling = []
    w_radial = []
    # ax.scatter(time_simul_B, p_B, color='g')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale('log')
    plt.show()


    plt.show(block=True)