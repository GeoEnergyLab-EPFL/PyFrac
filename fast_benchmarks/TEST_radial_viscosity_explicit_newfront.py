# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo on 01.09.20.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# local imports
from fast_benchmarks.TEST_tools import run_radial_vertex
import numpy as np

def test_radial_M_expl_newfront():

    # setting the parameters
    my_front_reconstruction = 'LS_continousfront'
    my_front_advancement = 'explicit'
    my_vertex = 'M'
    my_param  =    {"Lx": 0.3, "Ly": 0.3, "Nx": 41, "Ny": 41,
                    "nu": 0.4, "youngs_mod": 3.3e10, "K_Ic": 0.5,
                    "Cl": 0.,  "Q0": 0.001,          "viscosity": 1.1e-3,
                    "finalTime": 1e5, "initialR": 0.1}

    # running the simulation
    # exitcode, outputfolder = run_radial_vertex(my_front_reconstruction,my_front_advancement,my_vertex,my_param)
    #
    # assert exitcode == True, " error during the computation of the numerical solution"
    outputfolder = "./Temp_Data/M_radial_explicit_LS_continousfront"

    ########################
    # checking the results #
    ########################

    from postprocess_fracture import get_fracture_dimensions_analytical_with_properties
    from postprocess_fracture import get_HF_analytical_solution_at_point
    from postprocess_fracture import get_fracture_variable_at_point
    from visualization import load_fractures
    from visualization import get_fracture_variable
    from visualization import plot_fracture_list_slice
    from properties import LabelProperties

    # loading simulation results
    Fr_list, properties = load_fractures(address=outputfolder)       # load all fractures
    Solid, Fluid, Injection, simulProp = properties
    time_srs = get_fracture_variable(Fr_list, variable='time')

    #
    # comparing the analytical and the numerical radius
    Rmean_num_list = []
    Rmax_num_list = []
    Rmin_num_list = []
    R_ana_list = []
    for i in Fr_list:
        front_intersect_dist = np.sqrt((i.Ffront[::, [0, 2]].flatten() - i.mesh.CenterCoor[i.source[0]][0]) ** 2
                                     + (i.Ffront[::, [1, 3]].flatten() - i.mesh.CenterCoor[i.source[0]][1]) ** 2)
        Rmean_num_list.append(np.mean(front_intersect_dist))
        Rmax_num_list.append(np.max(front_intersect_dist))
        Rmin_num_list.append(np.min(front_intersect_dist))
        x_len, y_len = get_fracture_dimensions_analytical_with_properties('M',
                                                                          i.time,
                                                                          Solid,
                                                                          Injection,
                                                                          fluid_prop=Fluid,
                                                                          h=None,
                                                                          samp_cell=None,
                                                                          gamma=None)
        R_ana_list.append(x_len)

    for i in range(len(R_ana_list)):
        Diff=np.abs(R_ana_list[i]-Rmean_num_list[i])/R_ana_list[i]
        assert Diff < 5/100, "At time step n. "+str(i)+" the solution for the mean Radius is far from the analytical more than 5%"
        Diff=np.abs(R_ana_list[i]-Rmax_num_list[i])/R_ana_list[i]
        assert Diff < 5/100, "At time step n. "+str(i)+" the solution for the max Radius is far from the analytical more than 5%"
        Diff=np.abs(R_ana_list[i]-Rmin_num_list[i])/R_ana_list[i]
        assert Diff < 5/100, "At time step n. "+str(i)+" the solution for the min Radius is far from the analytical more than 5%"



    #
    # comparing the analytical and the numerical width at the fracture center
    w_num_point_values, time_list = get_fracture_variable_at_point(Fr_list,
                                                            'w',
                                                            point=[0., 0.])

    w_ana_point_values = get_HF_analytical_solution_at_point('M',
                                                             'w',
                                                             [0.,0.],
                                                             Solid,
                                                             Injection,
                                                             fluid_prop=Fluid,
                                                             time_srs=time_list,
                                                             h=None,
                                                             samp_cell=None,
                                                             gamma=None)

    for i in range(1,len(w_num_point_values)):
        Diff = np.abs(w_ana_point_values[i]-w_num_point_values[i])/w_ana_point_values[i]
        assert Diff < 5/100, "At time step n. "+str(i)+" the solution for the Fracture opening is far from the analytical more than 5%"

    ####### useful for debugging ###
    # from visualization import *
    # plot_prop = PlotProperties()
    # plot_prop.lineStyle = '.'  # setting the line style to point
    # plot_prop.graphScaling = 'loglog'  # setting to log log plot
    # Fig_w = plot_fracture_list_at_point(Fr_list,
    #                                     variable='w',
    #                                     plot_prop=plot_prop)
    # # plot analytical width at center
    # Fig_w = plot_analytical_solution_at_point('M',
    #                                           'w',
    #                                           Solid,
    #                                           Injection,
    #                                           fluid_prop=Fluid,
    #                                           time_srs=time_srs,
    #                                           fig=Fig_w)


    #
    # comparing the analytical and the numerical slice
    time_srs = np.asarray([2, 200, 5000, 30000, 100000])
    Fr_list, properties = load_fractures(address=outputfolder, time_srs=time_srs)
    #get the true times
    time_srs = get_fracture_variable(Fr_list, variable='time')

    # comparing w
    for i in range(len(Fr_list)):
        ext_pnts = np.empty((2, 2), dtype=np.float64)
        numerical_results_dict = plot_fracture_list_slice([Fr_list[i]],
                                          variable='w',
                                          projection='2D',
                                          plot_cell_center=True,
                                          extreme_points=ext_pnts,
                                          export2Json=True,
                                          export2Json_assuming_no_remeshing=False)

        from postprocess_fracture import get_HF_analytical_solution
        labels = LabelProperties('w', 'slice', '2D')
        analytical_list, mesh_list = get_HF_analytical_solution('M',
                                                                  'w',
                                                                  Solid,
                                                                  Injection,
                                                                  mesh=Fr_list[i].mesh,
                                                                  fluid_prop=Fluid,
                                                                  time_srs=[time_srs[i]],
                                                                  length_srs=None,
                                                                  h=None,
                                                                  samp_cell=None,
                                                                  gamma=None)
        for j in range(len(analytical_list)):
            analytical_list[j] /= labels.unitConversion
        sampling_cells = np.asarray(numerical_results_dict['sampling_cells_0'],dtype=int)
        analytical_w = np.asarray(analytical_list[0])[sampling_cells]
        numerical_w = np.asarray(numerical_results_dict['w_0'])
        diff = 0.
        diff_i = []
        diff_i_limit = np.array([7. / 1000., 1.1 /100., 1.7/100., 1.1 /100., 1.8/100])
        diff_limit =   np.array([8. / 100.,  14./100.,  0.18,     0.2,       0.19])

        for j in range(numerical_w.size):
            diff = diff + abs(numerical_w[j]-analytical_w[j])
            diff_i.append(abs(numerical_w[j]-analytical_w[j]))
        diff_i = np.asarray(diff_i)
        assert diff_i.max() < diff_i_limit[i]
        assert diff < diff_limit[i]


    for i in range(len(Fr_list)):
        ext_pnts = np.empty((2, 2), dtype=np.float64)
        numerical_results_dict = plot_fracture_list_slice([Fr_list[i]],
                                          variable='pn',
                                          projection='2D',
                                          plot_cell_center=True,
                                          extreme_points=ext_pnts,
                                          export2Json=True,
                                          export2Json_assuming_no_remeshing=False)

        # from postprocess_fracture import get_HF_analytical_solution
        # labels = LabelProperties('pf', 'slice', '2D')
        # analytical_list, mesh_list = get_HF_analytical_solution('M',
        #                                                           'pn',
        #                                                           Solid,
        #                                                           Injection,
        #                                                           mesh=Fr_list[i].mesh,
        #                                                           fluid_prop=Fluid,
        #                                                           time_srs=[time_srs[i]],
        #                                                           length_srs=None,
        #                                                           h=None,
        #                                                           samp_cell=None,
        #                                                           gamma=None)
        # for j in range(len(analytical_list)):
        #     analytical_list[j] /= labels.unitConversion
        # sampling_cells = np.asarray(numerical_results_dict['sampling_cells_0'],dtype=int)
        # analytical_p = np.asarray(analytical_list[0])[sampling_cells]
        # numerical_p = np.asarray(numerical_results_dict['pn_0'])
        # diff = 0.
        # diff_i = []
        # diff_i_limit = np.array([3.1 / 10., 1.1 /100., 1.7/100., 1.1 /100., 1.8/100])
        # diff_limit =   np.array([1.4,       14./100.,  0.18,     0.2,       0.19])
        # for j in range(numerical_p.size):
        #     diff = diff + abs(numerical_p[j]-analytical_p[j])
        #     diff_i.append(abs(numerical_p[j]-analytical_p[j]))
        # diff_i = np.asarray(diff_i)
        # assert diff_i.max() < diff_i_limit[i]
        # assert diff < diff_limit[i]

    ####### useful for debugging ###
    # from visualization import *
    # # plot slice
    # ext_pnts = np.empty((2, 2), dtype=np.float64)
    # Fig_WS = plot_fracture_list_slice(Fr_list,
    #                                   variable='w',
    #                                   projection='2D',
    #                                   plot_cell_center=True,
    #                                   extreme_points=ext_pnts)
    # # plot slice analytical
    # Fig_WS = plot_analytical_solution_slice('M',
    #                                         'w',
    #                                         Solid,
    #                                         Injection,
    #                                         fluid_prop=Fluid,
    #                                         fig=Fig_WS,
    #                                         time_srs=time_srs,
    #                                         point1=ext_pnts[0],
    #                                         point2=ext_pnts[1])

    from .TEST_tools import remove
    remove(outputfolder)