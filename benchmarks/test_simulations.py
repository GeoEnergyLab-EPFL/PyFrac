# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo on 01.09.20.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""

import shutil
import os
from benchmarks.simulparam_and_tolerances import *
from mesh import CartesianMesh
from properties import MaterialProperties, FluidProperties, InjectionProperties, SimulationProperties
from fracture import Fracture
from controller import Controller
from fracture_initialization import Geometry, InitializationParameters
import numpy as np
from visualization import *

def pytest_generate_tests(metafunc):
    # called once per each test function
    funcarglist = metafunc.cls.params[metafunc.function.__name__]
    argnames = sorted(funcarglist[0])
    metafunc.parametrize(
        argnames, [[funcargs[name] for name in argnames] for funcargs in funcarglist]
    )


class TestClass:
    # a map specifying one or multiple argument sets for each test method
    # in this case we have only the test method "test_simulnames" but you can add more of them
    # in this case we have only 1 parameter passed to the test method  but you can add more of them
    params = {
        "test_radial_vertex": [dict(testname='radial_M_explicit_newfront'),
                            dict(testname='radial_M_explicit_oldfront'),
                            dict(testname='radial_M_implicit_newfront'),
                            dict(testname='radial_M_implicit_oldfront'),
                            dict(testname='radial_K_explicit_newfront'),
                            dict(testname='radial_K_explicit_oldfront'),
                            dict(testname='radial_K_implicit_newfront'),
                            dict(testname='radial_K_implicit_oldfront')
                            ]
    }

    def remove(self, path):
        """ param <path> could either be relative or absolute. """
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)  # remove the file
        elif os.path.isdir(path):
            shutil.rmtree(path)  # remove dir and all contains
        else:
            print("File or dir {} has not been removed.".format(path))
            # raise ValueError("file {} is not a file or dir.".format(path))

    def run_radial_simul(self, my_front_reconstruction, my_front_advancement, my_vertex_or_path, my_param):
        # setting up the verbosity level of the log at console
        # setup_logging_to_console(verbosity_level='error')

        outputfolder = "./Temp_Data/" + my_vertex_or_path + "_radial_" + my_front_advancement + "_" + my_front_reconstruction
        self.remove(outputfolder)

        # creating mesh
        Mesh = CartesianMesh(my_param['Lx'], my_param['Ly'], my_param['Nx'], my_param['Ny'])

        # solid properties
        nu = my_param['nu']  # Poisson's ratio
        youngs_mod = my_param['youngs_mod']  # Young's modulus
        Eprime = youngs_mod / (1 - nu ** 2)  # plain strain modulus
        K_Ic = my_param['K_Ic']  # fracture toughness
        Cl = my_param['Cl']  # Carter's leak off coefficient

        # material properties
        Solid = MaterialProperties(Mesh, Eprime, K_Ic, Carters_coef=Cl)

        # injection parameters
        Q0 = my_param['Q0']  # injection rate
        Injection = InjectionProperties(Q0, Mesh)

        # fluid properties
        Fluid = FluidProperties(viscosity=my_param['viscosity'])

        # simulation properties
        simulProp = SimulationProperties()
        simulProp.finalTime = my_param['finalTime']  # the time at which the simulation stops
        simulProp.set_tipAsymptote(my_vertex_or_path)  # tip asymptote is evaluated with the viscosity dominated assumption
        simulProp.frontAdvancing = my_front_advancement  # to set explicit front tracking
        simulProp.plotFigure = False
        simulProp.set_solTimeSeries(np.asarray([2, 200, 5000, 30000, 100000]))
        simulProp.saveTSJump, simulProp.plotTSJump = 5, 5  # save and plot after every five time steps
        simulProp.set_outputFolder(outputfolder)
        simulProp.projMethod = my_front_reconstruction
        simulProp.log2file = False

        # initialization parameters
        Fr_geometry = Geometry('radial', radius=my_param['initialR'])
        init_param = InitializationParameters(Fr_geometry, regime=my_vertex_or_path)

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
        exitcode = controller.run()
        return exitcode, outputfolder

    def compute(self, testname, run):
        if testname in testnames.keys():
            my_front_advancement = testnames[testname]['front_advancement']
            my_front_reconstruction = testnames[testname]['front_reconstruction']
            if 'vertex' in testnames[testname].keys():
                my_vertex_or_path = testnames[testname]['vertex']
            elif 'path' in testnames[testname].keys():
                my_vertex_or_path = testnames[testname]['path']
            else:  SystemExit('You must specify vertex or path inside simulparam_and_tolerances !')
            my_param = simulparam[testnames[testname]['simulparam']]
            toll = tolerances[testname]
        else:
            SystemExit('Simulation not present !')

        if run:
            # running the simulation
            exitcode, outputfolder = self.run_radial_simul(my_front_reconstruction, my_front_advancement, my_vertex_or_path, my_param)
            #
            assert exitcode == True, " error during the computation of the numerical solution"
        else:
            outputfolder = "./Temp_Data/" + my_vertex_or_path + "_radial_" + my_front_advancement + "_" + my_front_reconstruction

        if not run:
            delete = False
        else:
            delete = True

        return outputfolder, my_vertex_or_path, toll, delete

    def test_radial_vertex(self, testname):
        ########################
        # running a simulation #
        ########################
        outputfolder, my_vertex, toll, delete = self.compute(testname, True) #-->>true if you want to rerun because you do not have results to check

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
        Fr_list, properties = load_fractures(address=outputfolder)  # load all fractures
        Solid, Fluid, Injection, simulProp = properties
        time_srs = get_fracture_variable(Fr_list, variable='time')

        #############################################
        # comparing analytical and numerical radius #
        #############################################
        Rmean_num_list = []
        Rmax_num_list = []
        Rmin_num_list = []
        R_ana_list = []
        for i in Fr_list:
            front_intersect_dist = np.sqrt((i.Ffront[::, [0, 2]].flatten() - i.mesh.CenterCoor[i.source[0]][0]) ** 2
                                           + (i.Ffront[::, [1, 3]].flatten() - i.mesh.CenterCoor[i.source[0]][
                1]) ** 2)
            Rmean_num_list.append(np.mean(front_intersect_dist))
            Rmax_num_list.append(np.max(front_intersect_dist))
            Rmin_num_list.append(np.min(front_intersect_dist))
            x_len, y_len = get_fracture_dimensions_analytical_with_properties(my_vertex,
                                                                              i.time,
                                                                              Solid,
                                                                              Injection,
                                                                              fluid_prop=Fluid,
                                                                              h=None,
                                                                              samp_cell=None,
                                                                              gamma=None)
            R_ana_list.append(x_len)

        for i in range(len(R_ana_list)):
            Diff = np.abs(R_ana_list[i] - Rmean_num_list[i]) / R_ana_list[i]
            assert Diff < toll['radius_toll'], "At time step n. " + str(
                i) + " the solution for the mean Radius is too far from the analytical"
            Diff = np.abs(R_ana_list[i] - Rmax_num_list[i]) / R_ana_list[i]
            assert Diff < toll['radius_toll'], "At time step n. " + str(
                i) + " the solution for the max Radius is too far from the analytical"
            Diff = np.abs(R_ana_list[i] - Rmin_num_list[i]) / R_ana_list[i]
            assert Diff < toll['radius_toll'], "At time step n. " + str(
                i) + " the solution for the min Radius is too far from the analytical"

        #################################################################
        # comparing analytical and numerical fracture opening at center #
        #################################################################
        w_num_point_values, time_list = get_fracture_variable_at_point(Fr_list,
                                                                       'w',
                                                                       point=[0., 0.])

        w_ana_point_values = get_HF_analytical_solution_at_point(my_vertex,
                                                                 'w',
                                                                 [0., 0.],
                                                                 Solid,
                                                                 Injection,
                                                                 fluid_prop=Fluid,
                                                                 time_srs=time_list,
                                                                 h=None,
                                                                 samp_cell=None,
                                                                 gamma=None)

        for i in range(1, len(w_num_point_values)):
            Diff = np.abs(w_ana_point_values[i] - w_num_point_values[i]) / w_ana_point_values[i]
            assert Diff < toll['w_center_toll'], "At time step n. " + str(
                i) + " the solution for the Fracture opening is too far from the analytical"

        ####### useful for debugging ###
        # from visualization import *
        # plot_prop = PlotProperties()
        # plot_prop.lineStyle = '.'  # setting the line style to point
        # plot_prop.graphScaling = 'loglog'  # setting to log log plot
        # Fig_w = plot_fracture_list_at_point(Fr_list,
        #                                     variable='w',
        #                                     plot_prop=plot_prop)
        # # plot analytical width at center
        # Fig_w = plot_analytical_solution_at_point( my_vertex,
        #                                           'w',
        #                                           Solid,
        #                                           Injection,
        #                                           fluid_prop=Fluid,
        #                                           time_srs=time_srs,
        #                                           fig=Fig_w)

        #####################################################################
        # comparing analytical and numerical fracture opening along a slice #
        #####################################################################
        time_srs = np.asarray([2, 200, 5000, 30000, 100000])
        Fr_list, properties = load_fractures(address=outputfolder, time_srs=time_srs)
        # get the true times
        time_srs = get_fracture_variable(Fr_list, variable='time')

        # comparing w
        diff_max_vs_time = []
        diff_total_vs_time = []
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
            analytical_list, mesh_list = get_HF_analytical_solution(my_vertex,
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
            sampling_cells = np.asarray(numerical_results_dict['w_sampling_cells_0'], dtype=int)
            analytical_w = np.asarray(analytical_list[0])[sampling_cells]
            numerical_w = np.asarray(numerical_results_dict['w_0'])
            diff = 0.
            diff_i = []

            for j in range(numerical_w.size):
                diff = diff + abs(numerical_w[j] - analytical_w[j])
                diff_i.append(abs(numerical_w[j] - analytical_w[j]))
            diff_i = np.asarray(diff_i)
            diff_max_vs_time.append(diff_i.max())
            diff_total_vs_time.append(diff)
        #print('here')
        for i in range(len(Fr_list)):
            assert diff_max_vs_time[i] < toll['w_section_toll_max_value'][i]
            assert diff_total_vs_time[i] < toll['w_section_toll_cumulative_value'][i]

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
        # Fig_WS = plot_analytical_solution_slice( my_vertex,
        #                                         'w',
        #                                         Solid,
        #                                         Injection,
        #                                         fluid_prop=Fluid,
        #                                         fig=Fig_WS,
        #                                         time_srs=time_srs,
        #                                         point1=ext_pnts[0],
        #                                         point2=ext_pnts[1])

        #####################################################################
        # comparing analytical and numerical pressure along a slice         #
        #####################################################################
        diff_max_vs_time = []
        diff_total_vs_time = []
        for i in range(len(Fr_list)):
            ext_pnts = np.empty((2, 2), dtype=np.float64)
            numerical_results_dict = plot_fracture_list_slice([Fr_list[i]],
                                                              variable='pn',
                                                              projection='2D',
                                                              plot_cell_center=True,
                                                              extreme_points=ext_pnts,
                                                              export2Json=True,
                                                              export2Json_assuming_no_remeshing=False)

            from postprocess_fracture import get_HF_analytical_solution
            labels = LabelProperties('pn', 'slice', '2D')
            analytical_list, mesh_list = get_HF_analytical_solution(my_vertex,
                                                                    'pn',
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
            sampling_cells = np.asarray(numerical_results_dict['pn_sampling_cells_0'], dtype=int)
            analytical_p = np.asarray(analytical_list[0])[sampling_cells]
            numerical_p = np.asarray(numerical_results_dict['pn_0'])
            diff = 0.
            diff_i = []
            for j in range(numerical_p.size):
                diff = diff + abs(numerical_p[j] - analytical_p[j])
                diff_i.append(abs(numerical_p[j] - analytical_p[j]))
            diff_i = np.asarray(diff_i)
            diff_max_vs_time.append(diff_i.max())
            diff_total_vs_time.append(diff)
        #print('here')
        for i in range(len(Fr_list)):
            assert diff_max_vs_time[i] < toll['p_section_toll_max_value'][i]
            assert diff_total_vs_time[i] < toll['p_section_toll_cumulative_value'][i]

        # ####### useful for debugging ###
        # plot slice
        # ext_pnts = np.empty((2, 2), dtype=np.float64)
        # Fig_WS = plot_fracture_list_slice(Fr_list,
        #                                   variable='pn',
        #                                   projection='2D',
        #                                   plot_cell_center=True,
        #                                   extreme_points=ext_pnts)
        # # plot slice analytical
        # Fig_WS = plot_analytical_solution_slice( my_vertex,
        #                                         'pn',
        #                                         Solid,
        #                                         Injection,
        #                                         fluid_prop=Fluid,
        #                                         fig=Fig_WS,
        #                                         time_srs=time_srs,
        #                                         point1=ext_pnts[0],
        #                                         point2=ext_pnts[1])
        if delete: self.remove(outputfolder)
