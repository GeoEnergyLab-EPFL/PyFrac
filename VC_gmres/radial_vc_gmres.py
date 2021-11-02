# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo on Fri Apr 17 23:16:25 2020.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# imports
import os
import time

# local imports
from mesh.mesh import CartesianMesh
from solid.solid_prop import MaterialProperties
from fluid.fluid_prop import FluidProperties
from properties import InjectionProperties, SimulationProperties
from fracture.fracture import Fracture
from controller import Controller
from fracture.fracture_initialization import Geometry, InitializationParameters
from utilities.utility import append_new_line
from utilities.utility import setup_logging_to_console
from systems.Hdot import Hdot_3DR0opening
from utilities.postprocess_fracture import load_fractures

# setting up the verbosity level of the log at console
setup_logging_to_console(verbosity_level='debug')

########## OPTIONS #########
run = False
use_iterative = True #GMRES use or not
use_HMAT = False
use_direct_TOEPLITZ = True
############################

if run:
    # creating mesh
    Mesh = CartesianMesh(10, 10, 101, 101)

    # solid properties
    nu = 0.4                            # Poisson's ratio
    youngs_mod = 3.3e10                 # Young's modulus
    Eprime = youngs_mod / (1 - nu ** 2) # plain strain modulus
    K_Ic = 0.5e6                          # fracture toughness
    properties = [youngs_mod, nu]

    Solid = MaterialProperties(Mesh,
                               Eprime,
                               toughness= K_Ic ,
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
            from solid.elasticity_isotropic import load_isotropic_elasticity_matrix_toepliz

            C = load_isotropic_elasticity_matrix_toepliz(Mesh, Eprime)

    # injection parameters
    Q0 = 0.001
    Injection = InjectionProperties(Q0, Mesh)

    # fluid properties
    Fluid = FluidProperties(viscosity=1.1e-3)

    # simulation properties
    simulProp = SimulationProperties()
    simulProp.finalTime = 50  # the time at which the simulation stops
    simulProp.tmStpPrefactor = 0.8  # decrease the pre-factor due to explicit front tracking
    simulProp.plotTSJump = 10
    simulProp.gmres_tol = 1e-13
    simulProp.saveToDisk = True
    simulProp.set_volumeControl(True)
    if use_iterative: simulProp.volumeControlGMRES = True
    simulProp.bckColor = 'K1c'
    simulProp.set_outputFolder("./Data/radial_VC_gmres")  # the disk address where the files are saved
    simulProp.set_tipAsymptote('K')  # the tip asymptote is evaluated with the toughness dominated assumption
    simulProp.plotVar = ['footprint']
    simulProp.frontAdvancing = 'implicit'  # <--- mandatory use
    simulProp.projMethod = 'LS_continousfront'  # <--- mandatory use
    #simulProp.force_time_schedule = True
    # simulProp.set_solTimeSeries( np.asarray([0.34029377788929954, 0.3910039421949774 , 0.44932063114650705 ,
    #                                         0.5163848234407662 , 0.5935086445791642 , 0.6822010388883218 ,
    #                                         0.7841972923438532 , 0.9014929838177141 , 1.0363830290126543 ,
    #                                         1.1915065809868355 , 1.3698986657571441 , 1.5750495632429984 ,
    #                                         1.81097309535173 , 2.0822851572767718 , 2.39429402849057 ,
    #                                         2.753104230386438 , 3.1657359625666865 , 3.6172490221886555 ,
    #                                         4.111016322286733 , 4.645272813065946 , 5.23091874141734 ,
    #                                         5.852718611327669 , 6.519015764803001 , 6.810536234012597 ,
    #                                         7.0]))
    # initialization parameters
    Fr_geometry = Geometry('radial', radius=1)

    if not simulProp.volumeControlGMRES:
        if use_direct_TOEPLITZ:
            simulProp.useBlockToeplizCompression = True
            from solid.elasticity_isotropic import load_isotropic_elasticity_matrix_toepliz
            C = load_isotropic_elasticity_matrix_toepliz(Mesh, Eprime)
        else:
            from solid.elasticity_isotropic import load_isotropic_elasticity_matrix
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
                            C = C)

    # run the simulation
    controller.run()
####################
# plotting results #
####################

# if not os.path.isfile('./batch_run.txt'):  # We only visualize for runs of specific examples
#
#     from visualization import *
    #
    # # loading simulation results
    # Fr_list, properties = load_fractures(address="./Data/radial_VC_gmres",step_size=1)                       # load all fractures
    # time_srs = get_fracture_variable(Fr_list, variable='time')                                                 # list of times
    # Solid, Fluid, Injection, simulProp = properties
    #
    #
    # # plot fracture radius
    # plot_prop = PlotProperties()
    # Fig_R = plot_fracture_list(Fr_list,
    #                            variable='footprint',
    #                            plot_prop=plot_prop)
    # Fig_R = plot_fracture_list(Fr_list,
    #                            fig=Fig_R,
    #                            variable='mesh',
    #                            mat_properties=properties[0],
    #                            backGround_param='sigma0',
    #                            plot_prop=plot_prop)
    #
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


    #plt.show(block=True)
####################
# plotting results #
####################
if not os.path.isfile('./batch_run.txt'):  # We only visualize for runs of specific examples

    from utilities.visualization import *
    #Fr_list, properties = load_fractures(address="./Data/sim_red", step_size=1)
    Fr_list, properties = load_fractures(address="./Data/radial_VC_gmres", step_size=1)
    time_srs = get_fracture_variable(Fr_list, variable='time')                                                 # list of times
    Solid, Fluid, Injection, simulProp = properties
    plot_prop = PlotProperties()
    plot_prop.lineStyle = '.'               # setting the linestyle to point
    plot_prop.graphScaling = 'loglog'       # setting to log log plot
    plot_prop.lineColor= (1.0,0.,0.)
    labels = LabelProperties('d_mean', 'whole mesh', '1D')
    labels.legend='Judit, look you can specify the text here'
    Fig_R = plot_fracture_list(Fr_list,
                               variable='d_mean',
                               plot_prop=plot_prop,
                               labels=labels)

    # Here I am getting the numerical radius
    var_val_list, time_list = get_fracture_variable(Fr_list,
                                                    'd_mean',
                                                    edge=4,
                                                    return_time=True)
    var_val_copy = copy.deepcopy(var_val_list)
    for i in range(len(var_val_copy)):
        var_val_copy[i] /= labels.unitConversion

    # Here I am getting the relative error
    # R = (3 / 2 ** 0.5 / np.pi * Q0 * Eprime * t / Kprime) ** 0.4
    Kprime = Solid.Kprime.max()
    Eprime = Solid.Eprime
    Q0 = Injection.injectionRate.max()
    rel_err = []
    for i in range(len(var_val_copy)):
        current_t = time_list[i]
        analytical_R = (3 / 2 ** 0.5 / np.pi * Q0 * Eprime * current_t / Kprime) ** 0.4
        rel_err_i = 100 * np.abs(var_val_copy[i] - analytical_R) / analytical_R
        rel_err.append(rel_err_i)

    # plot the relative error
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    ser = pd.Series(index=time_list, data=rel_err)
    df = ser.to_frame()
    df.reset_index(inplace=True)
    xlabel = 'time [s]'
    ylabel = 'relative error mean radius [%]'
    df.columns = [xlabel, ylabel]
    df.plot(kind='scatter', x=xlabel, y=ylabel, title = 'relative error VS time',legend=True, logx=True)
    plt.show()


    # Fr_list, properties = load_fractures(address="./Data/sim_blue", step_size=1)
    #
    # plot_prop.lineColor =  (0.0,0.,1.)
    # plot_prop.lineStyle = '+'               # setting the linestyle to point
    # labels = LabelProperties('d_mean', 'whole mesh', '1D')
    # labels.legend='Also the text here'
    # Fig_R = plot_fracture_list(Fr_list,
    #                            variable='d_mean',
    #                            plot_prop=plot_prop,
    #                            fig=Fig_R,
    #                            labels=labels)


    # plot analytical radius
    Fig_R = plot_analytical_solution(regime='K',
                                     variable='d_mean',
                                     mat_prop=Solid,
                                     inj_prop=Injection,
                                     fluid_prop=Fluid,
                                     time_srs=time_srs,
                                     fig=Fig_R)


    # Here I am getting the numerical radius
    var_val_list, time_list = get_fracture_variable(Fr_list,
                                                    'pn',
                                                    edge=4,
                                                    return_time=True)
    p_list = []
    for pres in var_val_list:
        p_list.append(pres.max())

    # Here I am getting the relative error
    # P = np.pi / 8 * (np.pi / 12) ** (1 / 5) * (Kprime ** 6 / (Eprime * Q0 * t)) ** (1 / 5)
    Kprime = Solid.Kprime.max()
    Eprime = Solid.Eprime
    Q0 = Injection.injectionRate.max()
    rel_err_p = []
    for i in range(len(var_val_list)):
        current_t = time_list[i]
        analytical_P = np.pi / 8 * (np.pi / 12) ** (1 / 5) * (Kprime ** 6 / (Eprime * Q0 * current_t)) ** (1 / 5)
        rel_err_i = 100 * np.abs(p_list[i] - analytical_P) / analytical_P
        rel_err_p.append(rel_err_i)

    # plot the relative error
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    ser = pd.Series(index=time_list, data=rel_err_p)
    df = ser.to_frame()
    df.reset_index(inplace=True)
    xlabel = 'time [s]'
    ylabel = 'relative error Pressure [%]'
    df.columns = [xlabel, ylabel]
    df.plot(kind='scatter', x=xlabel, y=ylabel, title = 'relative error VS time',legend=True, logx=True)
    plt.show()
    plt.show(block=True)