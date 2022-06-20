# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Brice Lecampion -
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2022.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# External imports
import numpy as np
import os

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


run=False
post_process=True
export_results=True
plotting = True

# setting up the verbosity level of the log at console
setup_logging_to_console(verbosity_level='info')

# creating mesh
Lr=0.425
Mesh = CartesianMesh(Lr, Lr, 81, 81)

# dimensionless simulation
# as per Peirce 2022
# we set all parameters to 1 such that t_m\tilde{m} =1
#
t_s=1
phi_v=0.1
K_prime=(t_s*(phi_v**(-65/9.)))**(1./26.)
# end time
t_end=1.8*t_s

# fluid properties
viscosity = 1. / 12  # mu' =1.
Fluid = FluidProperties(viscosity=viscosity)
# injection parameters
Q0 = 1.0  # injection rate

Rate_history=np.asarray([[0.0, t_s,1.01,1.25,1.26,1.4],[Q0, 0.,0.,0.,0,0.]])
Injection = InjectionProperties(Rate_history, Mesh)

# solid properties
Eprime = 1   # plain strain modulus
Cl = 0.5       # C'=2Cl=1 Carter's leak off coefficient
K1c =  K_prime/np.sqrt(32./np.pi)         #
min_width=1.e-6
# material properties
Solid = MaterialProperties(Mesh,
                           Eprime,
                           K1c,
                           Carters_coef=Cl,minimum_width=min_width)

baseName="MtoK_closure_w_1_phi_01-fine-new" #"Radial_closure_om_1_phi_01-fine"
foldername="./"+baseName+"/"

# simulation properties
simulProp = SimulationProperties()
simulProp.finalTime = t_end                           # the time at which the simulation stops
simulProp.saveTSJump, simulProp.plotTSJump = 1, 5   # save 1 and plot after every 5 time steps
simulProp.set_outputFolder(foldername)   # the disk address where the files are saved
simulProp.frontAdvancing = 'predictor-corrector'               # setting up predictor-corrector front advancing
simulProp.plotVar = ['w']


if run :
    # initializing fractures
    Fr_geometry = Geometry('radial')
    init_param = InitializationParameters(Fr_geometry, regime='M', time=0.001)

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


####################
# Post_Processing the results #
####################
if post_process:

    myJsonName=foldername+"Exported-results.json"
    from utilities.visualization import *
    from utilities.postprocess_fracture import *

    # loading simulation results
    Fr_list, properties = load_fractures(foldername)
    time_srs = get_fracture_variable(Fr_list,'time')
    eta = get_fracture_variable(Fr_list,'efficiency')
    d_list = get_fracture_variable(Fr_list, variable='d_mean', return_time=False)
    # net pressure at inlet
    p_list=get_fracture_variable(Fr_list, variable='net pressure', return_time=False)
    p_inlet=[p_list[e][Injection.sourceElem[0]] for e in range(len(p_list))]

    # need to find t_s and get R_s !
    i_ts = np.where(np.asarray(time_srs) == t_s)[0][0]
    R_s = d_list[i_ts]
    R_a = np.asarray(d_list).max()
    R_list = np.asarray(d_list) / R_s
    time_srs_a = np.asarray(time_srs)

    # post-process of closure radius
    # injection point is at 0,0 here
    min_rc = np.zeros(len(Fr_list))
    mean_rc = np.zeros(len(Fr_list))
    for i in range(len(Fr_list)):
        test = Fr_list[i]
        closed_elt = test.closed
        me = test.mesh
        c_c = me.CenterCoor[closed_elt]  # cell_center
        r_c = np.asarray([np.linalg.norm(c_c[e]) for e in range(len(c_c))])
        if r_c.size > 0:
            min_rc[i] = r_c.min() #/ R_s
            mean_rc[i] = np.mean(np.extract(r_c <= min_rc[i] + me.cellDiag / 3., r_c)) #/R_s

    ext_pnts = np.empty((2, 2), dtype=np.float64)
    fracture_list_slice_w_h = plot_fracture_list_slice(Fr_list,
                                                   variable='w',
                                                   projection='2D',
                                                   plot_cell_center=True,
                                                   extreme_points=ext_pnts,
                                                   orientation='horizontal',
                                                   point1=[-Lr, 0.0],
                                                   point2=[Lr, 0.], export2Json=True)

    fracture_list_slice_w_v = plot_fracture_list_slice(Fr_list,
                                                   variable='w',
                                                   projection='2D',
                                                   plot_cell_center=True,
                                                   extreme_points=ext_pnts,
                                                   orientation='vertical',
                                                   point1=[ 0.0,-Lr],
                                                   point2=[0.,Lr], export2Json=True)
    fracture_list_slice_p_h = plot_fracture_list_slice(Fr_list,
                                                   variable='pn',
                                                   projection='2D',
                                                   plot_cell_center=True,
                                                   extreme_points=ext_pnts,
                                                   orientation='horizontal',
                                                   point1=[-Lr, 0.0],
                                                   point2=[Lr, 0.], export2Json=True)

    fracture_list_slice_p_v = plot_fracture_list_slice(Fr_list,
                                                   variable='pn',
                                                   projection='2D',
                                                   plot_cell_center=True,
                                                   extreme_points=ext_pnts,
                                                   orientation='vertical',
                                                   point1=[ 0.0,-Lr],
                                                   point2=[0.,Lr], export2Json=True)

    if export_results :
        from utilities.postprocess_fracture import append_to_json_file

        append_to_json_file(myJsonName, time_srs, 'append2keyASnewlist', key='Time',
                            delete_existing_filename=True)

        append_to_json_file(myJsonName, eta, 'append2keyASnewlist', key='Efficiency')

        append_to_json_file(myJsonName, d_list, 'append2keyASnewlist', key='Radius')

        append_to_json_file(myJsonName, min_rc.tolist(), 'append2keyASnewlist', key='Closure Radius min')

        append_to_json_file(myJsonName, mean_rc.tolist(), 'append2keyASnewlist', key='Closure Radius mean')

        towrite = {'intersectionHslice width': fracture_list_slice_w_h}
        append_to_json_file(myJsonName, towrite, 'extend_dictionary')
        towrite = {'intersectionVslice width': fracture_list_slice_w_v}
        append_to_json_file(myJsonName, towrite, 'extend_dictionary')
        towrite = {'intersectionHslice p': fracture_list_slice_p_h}
        append_to_json_file(myJsonName, towrite, 'extend_dictionary')
        towrite = {'intersectionVslice p': fracture_list_slice_p_v}
        append_to_json_file(myJsonName, towrite, 'extend_dictionary')

    if plotting :
     # plotting efficiency
        plot_prop = PlotProperties(#graph_scaling='loglog',
                               line_style='.')
        label = LabelProperties('efficiency')
        label.legend = 'fracturing efficiency'
        Fig_eff = plot_fracture_list(Fr_list,
                               variable='efficiency',
                               plot_prop=plot_prop,
                               labels=label)
        plt.show(block=True)

        label = LabelProperties('d_mean')
        label.legend = 'radius'
        Fig_r = plot_fracture_list(Fr_list,
                                   variable='d_mean',
                                   plot_prop=plot_prop,
                                   labels=label)
        plt.show(block=True)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(time_srs,R_list,'r')
        ax.plot(time_srs[i_ts:],mean_rc[i_ts:],'b')
        ax.plot(time_srs[i_ts:],min_rc[i_ts:],'.b')
        plt.xlabel("Time (s)")
        plt.ylabel("R/R_s (m)")
        #ax.legend(["analytical solution","numerics"])
        plt.show()


        fig, ax = plt.subplots()
        ax.plot(time_srs[0:],p_inlet[0:],'r')
        plt.xlabel("Time (s)")
        plt.ylabel("p(0,t) (Pa)")
    #ax.legend(["analytical solution","numerics"])
        plt.show()

