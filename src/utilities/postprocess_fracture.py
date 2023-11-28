# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on 12.06.17.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# External imports
import numpy as np
import logging
from scipy.interpolate import griddata
import dill
import os
import re
import sys
import json
from typing import Dict
import copy

# Internal Imports
from systems.make_sys_common_fun import calculate_fluid_flow_characteristics_laminar
from utilities.utility import ReadFracture
from HF_reference_solutions import HF_analytical_sol, get_fracture_dimensions_analytical
from utilities.labels import *

if 'win32' in sys.platform or 'win64' in sys.platform:
    slash = '\\'
else:
    slash = '/'

#-----------------------------------------------------------------------------------------------------------------------
def convert_meshDict_to_mesh(fracture_list):
    """

    :return:
    """
    from mesh_obj.mesh import CartesianMesh
    from fracture_obj.fracture import Fracture


    if isinstance(fracture_list, list):
        for num, fr in enumerate(fracture_list):
            if isinstance(fr.mesh, Dict):
                mesh_dict = copy.deepcopy(fr.mesh)
                fracture_list[num].mesh = CartesianMesh(mesh_dict['domain Limits'][[2, 3]].tolist(),
                                                        mesh_dict['domain Limits'][[0, 1]].tolist(),
                                                        mesh_dict['nx'], mesh_dict['ny'])

    else:
        if isinstance(fracture_list.mesh, Dict):
            mesh_dict = copy.deepcopy(fracture_list.mesh)
            fracture_list.mesh = CartesianMesh(mesh_dict['domain Limits'][[2, 3]].tolist(),
                                               mesh_dict['domain Limits'][[0, 1]].tolist(),
                                               mesh_dict['nx'], mesh_dict['ny'])

#-----------------------------------------------------------------------------------------------------------------------

def load_fractures(address=None, sim_name='simulation', time_period=0.0, time_srs=None, step_size=1, load_all=False,
                   max_time=np.inf, load_all_meshes=True):
    """
    This function returns a list of the fractures. If address and simulation name are not provided, results from the
    default address and having the default name will be loaded.

    Args:
        address (string):               -- the folder address containing the saved files. If it is not provided,
                                           simulation from the default folder (_simulation_data_PyFrac) will be loaded.
        sim_name (string):              -- the simulation name from which the fractures are to be loaded. If not
                                           provided, simulation with the default name (Simulation) will be loaded.
        time_period (float):            -- time period between two successive fractures to be loaded. if not provided,
                                           all fractures will be loaded.
        time_srs (ndarray):             -- if provided, the fracture stored at the closest time after the given times
                                           will be loaded.
        step_size (int):                -- the number of time steps to skip before loading the next fracture. If not
                                           provided, all of the fractures will be loaded.
        load_all (bool):                -- avoid jumping time steps too close to each other
        max_time (float):               -- the simulation gets loaded onlz up to that specific time.
        load_all_meshes (bool):         -- boolean to decide if the meshes should be loaded.

    Returns:
        fracture_list(list):            -- a list of fractures.

    """
    from mesh_obj.mesh import CartesianMesh

    log = logging.getLogger('PyFrac.load_fractures')
    log.info('Returning fractures...')

    if address is None:
        address = '.' + slash + '_simulation_data_PyFrac'

    if address[-1] != slash:
        address = address + slash

    if isinstance(time_srs, float) or isinstance(time_srs, int):
        time_srs = np.array([time_srs])
    elif isinstance(time_srs, list):
        time_srs = np.array(time_srs)

    if re.match('\d+-\d+-\d+__\d+_\d+_\d+', sim_name[-20:]):
        sim_full_name = sim_name
    else:
        simulations = os.listdir(address)
        time_stamps = []
        for i in simulations:
            if re.match(sim_name + '__\d+-\d+-\d+__\d+_\d+_\d+', i):
                time_stamps.append(i[-20:])
        if len(time_stamps) == 0:
            raise ValueError('Simulation not found! The address might be incorrect.')

        Tmst_sorted = sorted(time_stamps)
        sim_full_name = sim_name + '__' + Tmst_sorted[-1]
    sim_full_path = address + sim_full_name
    properties_file = sim_full_path + slash + 'properties'
    try:
        with open(properties_file, 'rb') as inp:
            properties = dill.load(inp)
    except FileNotFoundError:
        raise SystemExit('Data not found. The address might be incorrect')

    fileNo = 0
    next_t = 0.0
    t_srs_indx = 0
    fracture_list = []

    t_srs_given = isinstance(time_srs, np.ndarray) #time series is given
    if t_srs_given:
        if len(time_srs) == 0:
            return fracture_list
        next_t = time_srs[t_srs_indx]

    # time at wich the first fracture file was modified
    while fileNo < 1e5:

        # trying to load next file. exit loop if not found
        try:
            ff = ReadFracture(sim_full_path + slash + sim_full_name + '_file_' + repr(fileNo))
        except FileNotFoundError:
            break

        fileNo += step_size

        if ff.time == 0.:
            ff.time = 1.e-16

        if load_all:
            log.info('Returning fracture at ' + repr(ff.time) + ' s')
            fracture_list.append(ff)
        else:
            if 1. - next_t / ff.time >= -1e-8:
                # if the current fracture time has advanced the output time period
                log.info('Returning fracture at ' + repr(ff.time) + ' s')
                log.info('Returning file number ' + repr(fileNo - 1))

                fracture_list.append(ff)

                if t_srs_given:
                    if t_srs_indx < len(time_srs) - 1:
                        t_srs_indx += 1
                        next_t = time_srs[t_srs_indx]
                    if ff.time > max(time_srs):
                        break
                else:
                    next_t = ff.time + time_period
                    if next_t >= max_time:
                        break

    if fileNo >= 1e5:
        raise SystemExit('too many files.')

    if len(fracture_list) == 0:
        raise ValueError("Fracture list is empty")

    #--- instantiate the list ---#
    intDict = []

    #--- do the first instance ---#
    if isinstance(fracture_list[0].mesh, Dict):
        intDict = copy.deepcopy(fracture_list[0].mesh)
        convert_meshDict_to_mesh(fracture_list[0])
        distinct_mesh = fracture_list[0].mesh
    else:
        distinct_mesh = fracture_list[0].mesh

    fracture_list[0].mesh = distinct_mesh
    mesh_ind = 0

    #--- loop over the rest ---#
    for num, fr in enumerate(fracture_list):
        if num != 0:
            if isinstance(fr.mesh, Dict):
                if not ((fr.mesh["domain Limits"] == intDict["domain Limits"]).all() and
                        fr.mesh["nx"] == intDict["nx"] and fr.mesh["ny"] == intDict["ny"]) or intDict == []:
                    intDict = copy.deepcopy(fr.mesh)
                    convert_meshDict_to_mesh(fr)
                    distinct_mesh = fr.mesh
                    mesh_ind = num
                else:
                    fr.mesh = mesh_ind

            else:
                if fr.mesh != distinct_mesh:
                    intDict = {'domain Limits' : fr.mesh.domainLimits,
                               'nx': fr.mesh.nx,
                               'ny': fr.mesh.ny}
                    distinct_mesh = fr.mesh
                    mesh_ind = num
                else:
                    fr.mesh = mesh_ind

    # saving the reference to the proper mesh! (i am not making a copy of the mesh per se)
    for i in range(len(fracture_list)):
        fr_i= fracture_list[i]
        if isinstance(fr_i.mesh, int):
            fracture_list[i].mesh = fracture_list[fr_i.mesh].mesh

    return fracture_list, properties

#-----------------------------------------------------------------------------------------------------------------------

def get_fracture_variable(fracture_list, variable, edge=4, return_time=False):
    """ This function returns the required variable from a fracture list.

    Args:
        fracture_list (list):       -- the fracture list from which the variable is to be extracted.
        variable (string):          -- the variable to be extracted. See :py:data:`labels.supported_variables` of the
                                        :py:mod:`labels` module for a list of supported variables.
        edge (int):                 -- the edge of the cell that will be plotted. This is for variables that
                                       are evaluated on the cell edges instead of cell center. It can have a
                                       value from 0 to 4 (0->left, 1->right, 2->bottom, 3->top, 4->average).
        return_time (bool):         -- if True, the times at which the fractures are stored will also be returned.

    Returns:
        - variable_list (list)      -- a list containing the extracted variable from each of the fracture. The \
                                       dimension and type of each member of the list depends upon the variable type.
        - time_srs (list)           -- a list of times at which the fractures are stored.
    """

    variable_list = init_list_of_objects(len(fracture_list))
    time_srs = init_list_of_objects(len(fracture_list))

    if variable == 'time' or variable == 't':
        for i in range(len(fracture_list)):
            variable_list[i] = fracture_list[i].time
            time_srs[i] = fracture_list[i].time

    elif variable == 'width' or variable == 'w' or variable == 'surface':
        for i in range(len(fracture_list)):
            variable_list[i] = fracture_list[i].w
            time_srs[i] = fracture_list[i].time

    elif variable == 'fluid pressure' or variable == 'pf':
        for i in range(len(fracture_list)):
            variable_list[i] = fracture_list[i].pFluid
            time_srs[i] = fracture_list[i].time

    elif variable == 'net pressure' or variable == 'pn':
        for i in range(len(fracture_list)):
            variable_list[i] = fracture_list[i].pNet
            if len(fracture_list[i].closed) != 0:
                variable_list[i][fracture_list[i].closed] = 0
            time_srs[i] = fracture_list[i].time

    elif variable == 'front velocity' or variable == 'v':
        for i in fracture_list:
            if isinstance(i.mesh, int):
                fr_mesh = fracture_list[i.mesh].mesh
            else:
                fr_mesh = i.mesh
            vel = np.full((fr_mesh.NumberOfElts, ), np.nan)
            vel[i.EltTip] = i.v
            variable_list.append(vel)
            time_srs.append(i.time)

    elif variable == 'Reynolds number' or variable == 'Re':
        if fracture_list[-1].ReynoldsNumber is None:
            raise SystemExit(err_var_not_saved)
        for i in range(len(fracture_list)):
            if edge < 0 or edge > 4:
                raise ValueError('Edge can be an integer between and including 0 and 4.')
            if edge < 4:
                variable_list.append(fracture_list[i].ReynoldsNumber[edge])
                time_srs[i] = fracture_list[i].time
            elif fracture_list[i].ReynoldsNumber is not None:
                variable_list.append(np.mean(fracture_list[i].ReynoldsNumber, axis=0))
                time_srs[i] = fracture_list[i].time
            else:
                if isinstance(i.mesh, int):
                    fr_mesh = fracture_list[i.mesh].mesh
                else:
                    fr_mesh = i.mesh
                variable_list.append(np.full((fr_mesh.NumberOfElts, ), np.nan))

    elif variable == 'fluid flux' or variable == 'ff':
        if fracture_list[-1].fluidFlux is None:
            raise SystemExit(err_var_not_saved)
        for i in range(len(fracture_list)):
            if edge < 0 or edge > 4:
                raise ValueError('Edge can be an integer between and including 0 and 4.')
            if edge < 4:
                variable_list[i] = fracture_list[i].fluidFlux[edge]
                time_srs[i] = fracture_list[i].time
            elif fracture_list[i].fluidFlux is not None:
                variable_list[i] = np.mean(fracture_list[i].fluidFlux, axis=0)
                time_srs[i] = fracture_list[i].time
            else:
                if isinstance(i.mesh, int):
                    fr_mesh = fracture_list[i.mesh].mesh
                else:
                    fr_mesh = i.mesh
                variable_list.append(np.full((fr_mesh.NumberOfElts,), np.nan))

    elif variable == 'fluid velocity' or variable == 'fv':
        if fracture_list[-1].fluidVelocity is None:
            raise SystemExit(err_var_not_saved)
        for i in range(len(fracture_list)):
            if edge < 0 or edge > 4:
                raise ValueError('Edge can be an integer between and including 0 and 4.')
            if edge < 4:
                variable_list[i] = fracture_list[i].fluidVelocity[edge]
                time_srs[i] = fracture_list[i].time
            elif fracture_list[i].fluidVelocity is not None:
                variable_list[i] = np.mean(fracture_list[i].fluidVelocity, axis=0)
                time_srs[i] = fracture_list[i].time
            else:
                if isinstance(i.mesh, int):
                    fr_mesh = fracture_list[i.mesh].mesh
                else:
                    fr_mesh = i.mesh
                variable_list.append(np.full((fr_mesh.NumberOfElts, ), np.nan))

    elif variable == 'pressure gradient x' or variable == 'dpdx':
        for i in fracture_list:
            # get the mesh (either stored there or retrieve from location)
            if isinstance(i.mesh, int):
                fr_mesh = fracture_list[i.mesh].mesh
            else:
                fr_mesh = i.mesh
            dpdxLft = (i.pNet[i.EltCrack] - i.pNet[fr_mesh.NeiElements[i.EltCrack, 0]]) \
                      * i.InCrack[fr_mesh.NeiElements[i.EltCrack, 0]]
            dpdxRgt = (i.pNet[fr_mesh.NeiElements[i.EltCrack, 1]] - i.pNet[i.EltCrack]) \
                      * i.InCrack[fr_mesh.NeiElements[i.EltCrack, 1]]
            dpdx = np.full((fr_mesh.NumberOfElts, ),0.0)
            dpdx[i.EltCrack] = np.mean([dpdxLft, dpdxRgt], axis=0)
            variable_list.append(dpdx)
            time_srs.append(i.time)

    elif variable == 'pressure gradient y' or variable == 'dpdy':
        for i in fracture_list:
            if isinstance(i.mesh, int):
                fr_mesh = fracture_list[i.mesh].mesh
            else:
                fr_mesh = i.mesh
            dpdyBtm = (i.pNet[i.EltCrack] - i.pNet[fr_mesh.NeiElements[i.EltCrack, 2]]) \
                      * i.InCrack[fr_mesh.NeiElements[i.EltCrack, 2]]
            dpdxtop = (i.pNet[fr_mesh.NeiElements[i.EltCrack, 3]] - i.pNet[i.EltCrack]) \
                      * i.InCrack[fr_mesh.NeiElements[i.EltCrack, 3]]
            dpdy = np.full((fr_mesh.NumberOfElts, ),0.0)
            dpdy[i.EltCrack] = np.mean([dpdyBtm, dpdxtop], axis=0)
            variable_list.append(dpdy)
            time_srs.append(i.time)

    elif variable == 'fluid flux as vector field' or variable == 'ffvf':
        if fracture_list[-1].fluidFlux_components is None:
            raise SystemExit(err_var_not_saved)
        for i in range(len(fracture_list)):
            if edge < 0 or edge > 4:
                raise ValueError('Edge can be an integer between and including 0 and 4.')
            if edge < 4:
                variable_list[i] = fracture_list[i].fluidFlux_components[edge]
                time_srs[i] = fracture_list[i].time
            elif i.fluidFlux_components is not None:
                variable_list[i] = fracture_list[i].fluidFlux_components
                time_srs[i] = fracture_list[i].time
            else:
                if isinstance(i.mesh, int):
                    fr_mesh = fracture_list[i.mesh].mesh
                else:
                    fr_mesh = i.mesh
                variable_list.append(np.full((fr_mesh.NumberOfElts, ), np.nan))

    elif variable == 'fluid velocity as vector field' or variable == 'fvvf':
        if fracture_list[-1].fluidVelocity_components is None:
            raise SystemExit(err_var_not_saved)
        for i in range(len(fracture_list)):
            if edge < 0 or edge > 4:
                raise ValueError('Edge can be an integer between and including 0 and 4.')
            if edge < 4:
                variable_list[i] = fracture_list[i].fluidVelocity_components[edge]
                time_srs[i] = fracture_list[i].time
            elif i.fluidFlux_components is not None:
                variable_list[i] = fracture_list[i].fluidVelocity_components
                time_srs[i] = fracture_list[i].time
            else:
                if isinstance(i.mesh, int):
                    fr_mesh = fracture_list[i.mesh].mesh
                else:
                    fr_mesh = i.mesh
                variable_list.append(np.full((fr_mesh.NumberOfElts,), np.nan))

    elif variable == 'effective viscosity' or variable == 'ev':
        if fracture_list[-1].effVisc is None:
            raise SystemExit(err_var_not_saved)
        for i in range(len(fracture_list)):
            if edge < 0 or edge > 4:
                raise ValueError('Edge can be an integer between and including 0 and 4.')
            if edge < 4:
                variable_list[i] = fracture_list[i].effVisc[edge]
                time_srs[i] = fracture_list[i].time
            elif i.effVisc is not None:
                variable_list[i] = np.mean(fracture_list[i].effVisc, axis=0)
                time_srs[i] = fracture_list[i].time
            else:
                if isinstance(i.mesh, int):
                    fr_mesh = fracture_list[i.mesh].mesh
                else:
                    fr_mesh = i.mesh
                variable_list.append(np.full((fr_mesh.NumberOfElts,), np.nan))

    elif variable == 'yielded' or variable == 'y':
        if fracture_list[-1].yieldRatio is None:
            raise SystemExit(err_var_not_saved)
        for i in range(len(fracture_list)):
            if edge < 0 or edge > 4:
                raise ValueError('Edge can be an integer between and including 0 and 4.')
            if edge < 4:
                variable_list[i] = fracture_list[i].yieldRatio[edge]
                time_srs[i] = fracture_list[i].time
            elif i.yieldRatio is not None:
                variable_list[i] = np.mean(fracture_list[i].yieldRatio, axis=0)
                time_srs[i] = fracture_list[i].time
            else:
                if isinstance(i.mesh, int):
                    fr_mesh = fracture_list[i.mesh].mesh
                else:
                    fr_mesh = i.mesh
                variable_list.append(np.full((fr_mesh.NumberOfElts,), np.nan))

    elif variable in ('front_dist_min', 'd_min', 'front_dist_max', 'd_max', 'front_dist_mean', 'd_mean'):
        for i in range(len(fracture_list)):
            if isinstance(fracture_list[i].mesh, int):
                fr_mesh = fracture_list[fracture_list[i].mesh].mesh
            else:
                fr_mesh = fracture_list[i].mesh
            if len(fracture_list[i].source) != 0:
                source_loc = fr_mesh.CenterCoor[fracture_list[i].source[0]]
            # coordinate of the zero vertex in the tip cells
            front_intersect_dist = np.sqrt((fracture_list[i].Ffront[::, [0, 2]].flatten() - source_loc[0]) ** 2
                                           + (fracture_list[i].Ffront[::, [1, 3]].flatten() - source_loc[1]) ** 2)
            if variable == 'front_dist_mean' or variable == 'd_mean':
                variable_list[i] = np.mean(front_intersect_dist)
            elif variable == 'front_dist_max' or variable == 'd_max':
                variable_list[i] = np.max(front_intersect_dist)
            elif variable == 'front_dist_min' or variable == 'd_min':
                variable_list[i] = np.min(front_intersect_dist)
            time_srs[i] = fracture_list[i].time
    elif variable == 'mesh':
        for i in range(len(fracture_list)):
            if isinstance(fracture_list[i].mesh, int):
                fr_mesh = fracture_list[fracture_list[i].mesh].mesh
            else:
                fr_mesh = fracture_list[i].mesh
            variable_list[i] = fr_mesh
            time_srs[i] = fracture_list[i].time

    elif variable == 'efficiency' or variable == 'ef':
        for i in range(len(fracture_list)):
            variable_list[i] = fracture_list[i].efficiency
            time_srs[i] = fracture_list[i].time
            
    elif variable == 'volume' or variable == 'V':
        for i in range(len(fracture_list)):
            variable_list[i] = fracture_list[i].FractureVolume
            time_srs[i] = fracture_list[i].time
            
    elif variable == 'leak off' or variable == 'lk':
        for i in range(len(fracture_list)):
            variable_list[i] = fracture_list[i].LkOff
            time_srs[i] = fracture_list[i].time
            
    elif variable == 'leaked off volume' or variable == 'lkv':
        for i in range(len(fracture_list)):
            variable_list[i] = sum(fracture_list[i].LkOffTotal[fracture_list[i].EltCrack])
            time_srs[i] = fracture_list[i].time

    elif variable == 'aspect ratio' or variable == 'ar':
        for i in range(len(fracture_list)):
            x_coords = np.hstack((fracture_list[i].Ffront[:, 0], fracture_list[i].Ffront[:, 2]))
            x_len = np.max(x_coords) - np.min(x_coords)
            y_coords = np.hstack((fracture_list[i].Ffront[:, 1], fracture_list[i].Ffront[:, 3]))
            y_len = np.max(y_coords) - np.min(y_coords)
            variable_list[i] = x_len / y_len
            time_srs[i] = fracture_list[i].time

    elif variable == 'chi':
        for i in fracture_list:
            if isinstance(i.mesh, int):
                fr_mesh = fracture_list[i.mesh].mesh
            else:
                fr_mesh = i.mesh
            vel = np.full((fr_mesh.NumberOfElts,), np.nan)
            vel[i.EltTip] = i.v
            variable_list.append(vel)
            time_srs.append(i.time)


    elif variable == 'regime':
        legend_coord = []
        if hasattr(fracture_list[0], 'regime_color'):
            for i in range(len(fracture_list)):
                variable_list[i] = fracture_list[i].regime_color
                time_srs[i] = fracture_list[i].time
        else:
            raise ValueError('The regime cannot be found. Saving of regime is most likely not enabled.\n'
                             ' See the saveRegime falg of SimulationProperties class.')

    elif variable == 'source elements' or variable == 'se':
        for i, fr in enumerate(fracture_list):
            variable_list[i] = fracture_list[i].source
            time_srs[i] = fracture_list[i].time

    elif variable == 'injection line pressure' or variable == 'ilp':
        for i, fr in enumerate(fracture_list):
            if fr.pInjLine is None:
                raise ValueError("It seems that injection line is not solved. Injection line pressure is not available")
            else:
                variable_list[i] = fr.pInjLine
            time_srs[i] = fr.time

    elif variable == 'injection rate' or variable == 'ir':
        for i, fr in enumerate(fracture_list):
            if fr.injectionRate is None:
                raise ValueError("It seems that injection line is not solved. Injection rate is not available")
            else:
                variable_list[i] = fr.injectionRate
            time_srs[i] = fr.time

    elif variable == 'total injection rate' or variable == 'tir':
        for i, fr in enumerate(fracture_list):
            if fr.injectionRate is None:
                raise ValueError("It seems that injection line is not solved. Injection rate is not available")
            else:
                variable_list[i] = (np.sum(fr.injectionRate))
            time_srs[i] = fr.time
    else:
        raise ValueError('The variable type is not correct.')

    if not return_time:
        return variable_list
    elif variable == 'regime':
        return variable_list, legend_coord, time_srs
    else:
        return variable_list, time_srs


#-----------------------------------------------------------------------------------------------------------------------

def get_fracture_variable_at_point(fracture_list, variable, point, edge=4, return_time=True):
    """ This function returns the required variable from a fracture list at the given point.

        Args:
            fracture_list (list):       -- the fracture list from which the variable is to be extracted.
            variable (string):          -- the variable to be extracted. See :py:data:`supported_variables` of the
                                            :py:mod:`Labels` module for a list of supported variables.
            point (list or ndarray):    -- the point at which the given variable is plotted against time [x, y].
            edge (int):                 -- the edge of the cell that will be plotted. This is for variables that
                                           are evaluated on the cell edges instead of cell center. It can have a
                                           value from 0 to 4 (0->left, 1->right, 2->bottom, 3->top, 4->average).
            return_time (bool):         -- if True, the times at which the fractures are stored will also be returned.

        Returns:
            - variable_list (list)      -- a list containing the extracted variable from each of the fracture. The \
                                           dimension and type of each member of the list depends upon the variable type.
            - time_srs (list)           -- a list of times at which the fractures are stored.

    """
    log = logging.getLogger('PyFrac.get_fracture_variable_at_point')
    if variable not in supported_variables:
        raise ValueError(err_msg_variable)

    return_list = init_list_of_objects(len(fracture_list))

    if variable in ['front intercepts', 'fi']:
        return_list = get_front_intercepts(fracture_list, point)
        if return_time:
            return return_list, get_fracture_variable(fracture_list, 't')
        else:
            return return_list
    else:
        var_values, time_list = get_fracture_variable(fracture_list,
                                                    variable,
                                                    edge=edge,
                                                    return_time=True)

    if variable in unidimensional_variables:
        return_list = var_values
    else:
        for i in range(len(fracture_list)):
            if variable in bidimensional_variables:
                if isinstance(fracture_list[i].mesh, int):
                    fr_mesh = fracture_list[fracture_list[i].mesh].mesh
                else:
                    fr_mesh = fracture_list[i].mesh
                try:
                    fr_mesh.locate_element(point[0][0], point[0][1])[0]
                except TypeError:
                    value_point = [np.nan]
                else:
                    ind = fr_mesh.locate_element(point[0][0], point[0][1])[0]
                    cellDiag = np.sqrt(fr_mesh.hx ** 2 + fr_mesh.hy ** 2)
                    xmin = (fr_mesh.CenterCoor[ind].flatten())[0] - 3 * cellDiag
                    xmax = (fr_mesh.CenterCoor[ind].flatten())[0] + 3 * cellDiag
                    ymin = (fr_mesh.CenterCoor[ind].flatten())[1] - 3 * cellDiag
                    ymax = (fr_mesh.CenterCoor[ind].flatten())[1] + 3 * cellDiag
                    indBox = fr_mesh.get_cells_inside_box(xmin, xmax, ymin, ymax)

                    value_point = griddata(fr_mesh.CenterCoor[indBox],
                                           var_values[i][indBox],
                                           point,
                                           method='linear',
                                           fill_value=np.nan)
                # interpolate the neighbours of the element only
                # value_point = griddata(fracture_list[i].mesh.CenterCoor,
                #                        var_values[i],
                #                        point,
                #                        method='linear',
                #                        fill_value=np.nan)
                if np.isnan(value_point):
                    log.warning('Point outside fracture.')

                return_list[i] = value_point[0]

    if return_time:
        return return_list, time_list
    else:
        return return_list


#-----------------------------------------------------------------------------------------------------------------------

def get_fracture_variable_slice_interpolated(var_value, mesh, point1=None, point2=None):
    """
    This function returns the given fracture variable on a given slice of the domain. Two points are to be given that
    will be joined to form the slice. The values on the slice are interpolated from the values available on the cell
    centers.

    Args:
        var_value (ndarray):        -- the value of the variable on each cell of the domain.
        mesh (CartesianMesh):       -- the CartesianMesh object describing the mesh.
        point1 (list or ndarray):   -- the left point from which the slice should pass [x, y].
        point2 (list or ndarray):   -- the right point from which the slice should pass [x, y].

    Returns:
        - value_samp_points (ndarray)   -- the values of the variable at the sampling points given by sampling_line \
                                           (see below).
        - sampling_line (ndarray)       -- the distance of the point where the value is provided from the center of\
                                           the slice.

    """
    if not isinstance(var_value, np.ndarray):
        raise ValueError("Variable value should be provided in the form of numpy array with the size equal to the "
                         "number of elements in the mesh!")
    elif var_value.size != mesh.NumberOfElts:
        raise ValueError("Given array is not equal to the number of elements in mesh!")

    if point1 is None:
        point1 = np.array([-mesh.Lx, 0.])
    if point2 is None:
        point2 = np.array([mesh.Lx, 0.])

    # the code below find the extreme points of the line joining the two given points with the current mesh
    if point2[0] == point1[0]:
        point1[1] = -mesh.Ly
        point2[1] = mesh.Ly
    elif point2[1] == point1[1]:
        point1[0] = -mesh.Lx
        point2[0] = mesh.Lx
    else:
        slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
        y_intrcpt_lft = slope * (-mesh.Lx - point1[0]) + point1[1]
        y_intrcpt_rgt = slope * (mesh.Lx - point1[0]) + point1[1]
        x_intrcpt_btm = (-mesh.Ly - point1[1]) / slope + point1[0]
        x_intrcpt_top = (mesh.Ly - point1[1]) / slope + point1[0]

        if abs(y_intrcpt_lft) < mesh.Ly:
            point1[0] = -mesh.Lx
            point1[1] = y_intrcpt_lft
        if y_intrcpt_lft > mesh.Ly:
            point1[0] = x_intrcpt_top
            point1[1] = mesh.Ly
        if y_intrcpt_lft < -mesh.Ly:
            point1[0] = x_intrcpt_btm
            point1[1] = -mesh.Ly

        if abs(y_intrcpt_rgt) < mesh.Ly:
            point2[0] = mesh.Lx
            point2[1] = y_intrcpt_rgt
        if y_intrcpt_rgt > mesh.Ly:
            point2[0] = x_intrcpt_top
            point2[1] = mesh.Ly
        if y_intrcpt_rgt < -mesh.Ly:
            point2[0] = x_intrcpt_btm
            point2[1] = -mesh.Ly

    sampling_points = np.hstack((np.linspace(point1[0], point2[0], 105).reshape((105, 1)),
                                 np.linspace(point1[1], point2[1], 105).reshape((105, 1))))

    value_samp_points = griddata(mesh.CenterCoor,
                                 var_value,
                                 sampling_points,
                                 method='linear',
                                 fill_value=np.nan)

    sampling_line_lft = ((sampling_points[:52, 0] - sampling_points[52, 0]) ** 2 +
                         (sampling_points[:52, 1] - sampling_points[52, 1]) ** 2) ** 0.5
    sampling_line_rgt = ((sampling_points[52:, 0] - sampling_points[52, 0]) ** 2 +
                         (sampling_points[52:, 1] - sampling_points[52, 1]) ** 2) ** 0.5
    sampling_line = np.concatenate((-sampling_line_lft, sampling_line_rgt))

    return value_samp_points, sampling_line


#-----------------------------------------------------------------------------------------------------------------------

def get_fracture_variable_slice_cell_center(var_value, mesh, point=None, orientation='horizontal'):
    """
    This function returns the given fracture variable on a given slice of the domain. Two slice is constructed from the
    given point and the orientation. The values on the slice are taken from the cell centers.

    Args:
        var_value (ndarray):        -- the value of the variable on each cell of the domain.
        mesh (CartesianMesh):       -- the CartesianMesh object describing the mesh.
        point (list or ndarray):    -- the point from which the slice should pass [x, y]. If it does not lie on a cell
                                       center, the closest cell center will be taken. By default, [0., 0.] will be
                                       taken.
        orientation (string):       -- the orientation according to which the slice is made in the case the
                                       plotted values are not interpolated and are taken at the cell centers.
                                       Any of the four ('vertical', 'horizontal', 'ascending' and 'descending')
                                       orientation can be used.

    Returns:
        - var_value (ndarray)       -- the values of the variable at the sampling points given by sampling_line \
                                       (see below).
        - sampling_line (ndarray)   -- the distance of the point where the value is provided from the center of\
                                       the slice.
        - sampling_cells (ndarray)  -- the cells on the mesh along with the slice is made.
    """

    if not isinstance(var_value, np.ndarray):
        raise ValueError("Variable value should be provided in the form of numpy array with the size equal to the "
                         "number of elements in the mesh!")
    elif var_value.size != mesh.NumberOfElts:
        raise ValueError("Given array is not equal to the number of elements in mesh!")

    if point is None:
        point = np.array([0., 0.])
    if orientation not in ('horizontal', 'vertical', 'increasing', 'decreasing'):
        raise ValueError("Given orientation is not supported. Possible options:\n 'horizontal', 'vertical',"
                         " 'increasing', 'decreasing'")

    zero_cell = mesh.locate_element(point[0], point[1])
    if zero_cell is np.nan:
        raise ValueError("The given point does not lie in the grid!")

    if orientation == 'vertical':
        sampling_cells = np.hstack((np.arange(zero_cell, 0, -mesh.nx)[::-1],
                                    np.arange(zero_cell, mesh.NumberOfElts, mesh.nx)))
    elif orientation == 'horizontal':
        sampling_cells = np.arange(zero_cell // mesh.nx * mesh.nx, (zero_cell // mesh.nx + 1) * mesh.nx)

    elif orientation == 'increasing':
        bottom_half = np.arange(zero_cell, 0, -mesh.nx - 1)
        bottom_half = np.delete(bottom_half, np.where(mesh.CenterCoor[bottom_half, 0] >
                                                      mesh.CenterCoor[zero_cell, 0])[0])
        top_half = np.arange(zero_cell, mesh.NumberOfElts, mesh.nx + 1)
        top_half = np.delete(top_half, np.where(mesh.CenterCoor[top_half, 0] <
                                                mesh.CenterCoor[zero_cell, 0])[0])
        sampling_cells = np.hstack((bottom_half[::-1], top_half))

    elif orientation == 'decreasing':
        bottom_half = np.arange(zero_cell, 0, -mesh.nx + 1)
        bottom_half = np.delete(bottom_half, np.where(mesh.CenterCoor[bottom_half, 0] <
                                                      mesh.CenterCoor[zero_cell, 0])[0])
        top_half = np.arange(zero_cell, mesh.NumberOfElts, mesh.nx - 1)
        top_half = np.delete(top_half, np.where(mesh.CenterCoor[top_half, 0] >
                                                      mesh.CenterCoor[zero_cell, 0])[0])
        sampling_cells = np.hstack((bottom_half[::-1], top_half))

    sampling_len = ((mesh.CenterCoor[sampling_cells[0], 0] - mesh.CenterCoor[sampling_cells[-1], 0]) ** 2 + \
                    (mesh.CenterCoor[sampling_cells[0], 1] - mesh.CenterCoor[sampling_cells[-1], 1]) ** 2) ** 0.5

    # making x-axis centered at zero for the 1D slice. Necessary to have same reference with different meshes and
    # analytical solution plots.
    sampling_line = np.linspace(0, sampling_len, len(sampling_cells)) - sampling_len / 2

    return var_value[sampling_cells], sampling_line, sampling_cells


#-----------------------------------------------------------------------------------------------------------------------

def get_HF_analytical_solution(regime, variable, mat_prop, inj_prop, mesh=None, fluid_prop=None,
                                time_srs=None, length_srs=None, h=None, samp_cell=None, gamma=None):

    if time_srs is None and length_srs is None:
        raise ValueError('Either time series or lengths series is to be provided.')

    if regime == 'E_K':
        Kc_1 = mat_prop.Kc1
    else:
        Kc_1 = None

    if regime == 'E_E':
        Cij = mat_prop.Cij
    else:
        Cij = None

    if regime == 'MDR':
        density = fluid_prop.density
    else:
        density = None

    if inj_prop.injectionRate.size > 2:
        V0 = inj_prop.injectionRate[0, 1] * inj_prop.injectionRate[1, 0]
    else:
        V0 = None

    if regime in ['M', 'MDR', 'Mt', 'PKN', 'Mp']:
        if fluid_prop is None:
            raise ValueError('Fluid properties required for ' + regime + ' type analytical solution')
        muPrime = fluid_prop.muPrime
    else:
        muPrime = None

    if samp_cell is None:
        samp_cell = int(len(mat_prop.Kprime) / 2)

    if time_srs is not None:
        srs_length = len(time_srs)
    else:
        srs_length = len(length_srs)

    mesh_list = []
    return_list = []

    for i in range(srs_length):

        if length_srs is not None:
            length = length_srs[i]
        else:
            length = None

        if time_srs is not None:
            time = time_srs[i]
        else:
            time = None

        if variable in ['time', 't', 'width', 'w', 'net pressure', 'pn', 'front velocity', 'v']:

            if mesh is None and variable in ['width', 'w', 'net pressure', 'pn']:
                x_len, y_len = get_fracture_dimensions_analytical_with_properties(regime,
                                                                                  time_srs[i],
                                                                                  mat_prop,
                                                                                  inj_prop,
                                                                                  fluid_prop=fluid_prop,
                                                                                  h=h,
                                                                                  samp_cell=samp_cell,
                                                                                  gamma=gamma)

                from mesh_obj.mesh import CartesianMesh
                mesh_i = CartesianMesh(x_len, y_len, 151, 151)
            else:
                mesh_i = mesh

            t, r, p, w, v, actvElts = HF_analytical_sol(regime,
                                                        mesh_i,
                                                        mat_prop.Eprime,
                                                        inj_prop.injectionRate[1, 0],
                                                        inj_point=inj_prop.sourceCoordinates,
                                                        muPrime=muPrime,
                                                        Kprime=mat_prop.Kprime[samp_cell],
                                                        Cprime=mat_prop.Cprime[samp_cell],
                                                        length=length,
                                                        t=time,
                                                        Kc_1=Kc_1,
                                                        h=h,
                                                        density=density,
                                                        Cij=Cij,
                                                        gamma=gamma,
                                                        required=required_string[variable],
                                                        Vinj=V0)
            mesh_list.append(mesh_i)

            if variable == 'time' or variable == 't':
                return_list.append(t)
            elif variable == 'width' or variable == 'w':
                return_list.append(w)
            elif variable == 'net pressure' or variable == 'pn':
                return_list.append(p)
            elif variable == 'front velocity' or variable == 'v':
                return_list.append(v)

        elif variable in ['front_dist_min', 'd_min', 'front_dist_max', 'd_max', 'front_dist_mean', 'd_mean',
                          'radius', 'r']:
            x_len, y_len = get_fracture_dimensions_analytical_with_properties(regime,
                                                                              time,
                                                                              mat_prop,
                                                                              inj_prop,
                                                                              fluid_prop=fluid_prop,
                                                                              h=h,
                                                                              samp_cell=samp_cell,
                                                                              gamma=gamma)
            if variable == 'radius' or variable == 'r':
                return_list.append(x_len)
            elif variable == 'front_dist_min' or variable == 'd_min':
                return_list.append(y_len)
            elif variable == 'front_dist_max' or variable == 'd_max':
                return_list.append(x_len)
            elif variable == 'front_dist_mean' or variable == 'd_mean':
                if regime in ('E_K', 'E_E'):
                    raise ValueError('Mean distance not available.')
                else:
                    return_list.append(x_len)
        else:
            raise ValueError('The variable type is not correct or the analytical solution not available. Select'
                             ' one of the following:\n'
                                '-- \'r\' or \'radius\'\n' 
                                '-- \'w\' or \'width\'\n' 
                                '-- \'pn\' or \'net pressure\'\n' 
                                '-- \'v\' or \'front velocity\'\n'
                                '-- \'d_min\' or \'front_dist_min\'\n'
                                '-- \'d_max\' or \'front_dist_max\'\n'
                                '-- \'d_mean\' or \'front_dist_mean\'\n' )

    return return_list, mesh_list


#-----------------------------------------------------------------------------------------------------------------------

def get_HF_analytical_solution_at_point(regime, variable, point, mat_prop, inj_prop, fluid_prop=None, time_srs=None,
                                        length_srs=None, h=None, samp_cell=None, gamma=None):

    values_point = []

    if time_srs is not None:
        srs_length = len(time_srs)
    else:
        srs_length = len(length_srs)

    from mesh_obj.mesh import CartesianMesh
    if point[0] == 0.:
        mesh_Lx = 1.
    else:
        mesh_Lx = 2 * abs(point[0])
    if point[1] == 0.:
        mesh_Ly = 1.
    else:
        mesh_Ly = 2 * abs(point[1])
    mesh = CartesianMesh(mesh_Lx, mesh_Ly, 5, 5)

    for i in range(srs_length):

        if time_srs is not None:
            time = [time_srs[i]]
        else:
            time = None

        if length_srs is not None:
            length = [length_srs[i]]
        else:
            length = None

        value_mesh, mesh_list = get_HF_analytical_solution(regime,
                                                        variable,
                                                        mat_prop,
                                                        inj_prop,
                                                        mesh=mesh,
                                                        fluid_prop=fluid_prop,
                                                        time_srs=time,
                                                        length_srs=length,
                                                        h=h,
                                                        samp_cell=samp_cell,
                                                        gamma=gamma)
        if variable in ['front_dist_min', 'd_min', 'front_dist_max', 'd_max', 'front_dist_mean', 'd_mean',
                          'radius', 'r', 't', 'time']:
            values_point.append(value_mesh[0])
        elif point == [0., 0.]:
            values_point.append(value_mesh[0][mesh_list[0].CenterElts])
        else:
            value_point = value_mesh[0][18]
            values_point.append(value_point)

    return values_point

#-----------------------------------------------------------------------------------------------------------------------

def get_fracture_dimensions_analytical_with_properties(regime, time_srs, mat_prop, inj_prop, fluid_prop=None,
                                              h=None, samp_cell=None, gamma=None):


    if regime == 'E_K':
        Kc_1 = mat_prop.Kc1
    else:
        Kc_1 = None

    if regime == 'MDR':
        density = fluid_prop.density
    else:
        density = None

    if regime in ('M', 'Mt', 'PKN', 'MDR', 'Mp', 'La'):
        if fluid_prop is None:
            raise ValueError('Fluid properties required to evaluate analytical solution')
        muPrime = fluid_prop.muPrime
    else:
        muPrime = None

    if samp_cell is None:
        samp_cell = int(len(mat_prop.Kprime) / 2)

    if inj_prop.injectionRate.size > 2:
        V0 = inj_prop.injectionRate[0, 1] * inj_prop.injectionRate[1, 0]
    else:
        V0=None

    Q0 = inj_prop.injectionRate[1, 0]

    x_len, y_len = get_fracture_dimensions_analytical(regime,
                                                      np.max(time_srs),
                                                      mat_prop.Eprime,
                                                      Q0,
                                                      muPrime,
                                                      Kprime=mat_prop.Kprime[samp_cell],
                                                      Cprime=mat_prop.Cprime[samp_cell],
                                                      Kc_1=Kc_1,
                                                      h=h,
                                                      density=density,
                                                      gamma=gamma,
                                                      Vinj=V0)

    return x_len, y_len


#-----------------------------------------------------------------------------------------------------------------------

def write_fracture_variable_csv_file(file_name, fracture_list, variable, point=None, edge=4):
    """ This function writes fracture variable from each fracture in the list as a csv file. The variable from each of
        the fracture in the list will saved in a row of the csv file. If a variable is bi-dimensional, a point can be
        given at which the variable is to be saved.

        Args:
            file_name (string):         -- the name of the file to be written.
            fracture_list (list):       -- the fracture list from which the variable is to be extracted.
            variable (string):          -- the variable to be saved. See :py:data:`supported_variables` of the
                                            :py:mod:`Labels` module for a list of supported variables.
            point (list or ndarray):    -- the point in the mesh at which the given variable is saved [x, y]. If the
                                           point is not given, the variable will be saved on the whole mesh.
            edge (int):                 -- the edge of the cell that will be saved. This is for variables that
                                           are evaluated on the cell edges instead of cell center. It can have a
                                           value from 0 to 4 (0->left, 1->right, 2->bottom, 3->top, 4->average).


    """
    log = logging.getLogger('PyFrac.write_fracture_variable_csv_file')
    if variable not in supported_variables:
        raise ValueError(err_msg_variable)

    return_list = []

    var_values, time_list = get_fracture_variable(fracture_list,
                                                    variable,
                                                    edge=edge,
                                                    return_time=True)

    if point == None:
        return_list = var_values
    else:
        for i in range(len(fracture_list)):
            if isinstance(fracture_list[i].mesh, int):
                fr_mesh = fracture_list[fracture_list[i].mesh].mesh
            else:
                fr_mesh = fracture_list[i].mesh
            value_point = griddata(fr_mesh.CenterCoor,
                                   var_values[i],
                                   point,
                                   method='linear',
                                   fill_value=np.nan)
            if np.isnan(value_point):
                log.warning('Point outside fracture.')
            return_list.append(value_point[0])

    if file_name[-4:] != '.csv':
        file_name = file_name + '.csv'

    return_list_np = np.asarray(return_list)
    # np.savetxt(file_name, return_list_np, delimiter=',')

    import csv
    # write EltCrack file
    file2 = open(file_name, 'w')
    writer2 = csv.writer(file2)
    writer2.writerows(return_list_np)
    file2.close()


#-----------------------------------------------------------------------------------------------------------------------

def read_fracture_variable_csv_file(file_name):
    """ This function returns the required variable from the csv file.

        Args:
            file_name (string):         -- the name of the file to be written.

        Returns:
            - variable_list (list)      -- a list containing the extracted variable from each of the fracture. The \
                                           dimension and type of each member of the list depends upon the variable type.

    """

    if file_name[-4:] != '.csv':
        file_name = file_name + '.csv'

    return np.genfromtxt(file_name, delimiter=',')

#-----------------------------------------------------------------------------------------------------------------------


def write_fracture_mesh_csv_file(file_name, mesh_list):
    """ This function writes important data of a mesh as a csv file. The csv contains (in a row vector) the number of
    elements, hx, hy, nx, ny, the flattened connectivity matrix and the flattened node coordinates. Each row of the csv
    corresponds to an entry in the mesh_list

        Args:
            file_name (string):         -- the name of the file to be written.
            mesh_list (list):       -- the fracture list from which the variable is to be extracted.

    """

    return_list = []

    for i in mesh_list:
        export_mesh = np.array([i.NumberOfElts])
        export_mesh = np.append(export_mesh, [i.hx, i.hy, i.nx, i.ny])
        export_mesh = np.append(export_mesh, i.Connectivity.flatten())
        export_mesh = np.append(export_mesh, i.VertexCoor.flatten())
        return_list.append(export_mesh)

    if file_name[-4:] != '.csv':
        file_name = file_name + '.csv'

    return_list_np = np.asarray(return_list)
    # np.savetxt(file_name, return_list_np, delimiter=',')

    import csv
    # write EltCrack file
    file2 = open(file_name, 'w')
    writer2 = csv.writer(file2)
    writer2.writerows(return_list_np)
    file2.close()

#-----------------------------------------------------------------------------------------------------------------------

def append_to_json_file(file_name, content, action, key=None, delete_existing_filename=False):
    """ This function appends data of a mesh as a json file.

        Args:
            file_name (string):     -- the name of the file to be written.
            key (string):           -- a string that describes the information you are passing.
            content (list):         -- a list of some informations
            action (string):        -- action to take. Current options are:
                                       'append2keyASnewlist'
                                       This option means that you load the json file, you take the content of the key
                                       and then you append the new content as a new list in a list of lists
                                       if the existing content was not a list it will be put in a list

                                       'append2keyAND2list'
                                       This option means that you load the json file, you take the content of the key
                                       and, supposing that it is a key, you append the new content to it
                                       if the existing content was not a list it will be put in a list and the new content
                                       will be appended to it

                                       'dump_this_dictionary'
                                       You will dump only the content of the dictionary

    """
    log = logging.getLogger('PyFrac.append_to_json_file')

    # 0)transform np.ndarray to list before output
    if isinstance(content, np.ndarray):
        content = np.ndarray.tolist(content)

    # 1)check if the file_name is a Json file
    if file_name[-5:] != '.json':
        file_name = file_name + '.json'

    # 3)check if the file already exist
    if os.path.isfile(file_name) and delete_existing_filename:
        os.remove(file_name)
        log.warning("File " +file_name +"  existed and it will be Removed!")

    # 4)check if the file already exist
    if os.path.isfile(file_name):
        # The file exist

        with open(file_name, "r+") as json_file:
            data = json.load(json_file) # get the data
            if action in ['append2keyASnewlist', 'append2keyAND2list'] and not key == None:
                if key in data: # the key exist and we need just to add the value
                    if isinstance(data[key], list): # the data that is already there is a list and a key is provided
                        data[key].append(content)
                    elif action == 'append2keyAND2list':
                        data[key] = [data[key], content]
                    elif action == 'append2keyASnewlist':
                        data[key] = [[data[key]], [content]]
                else:
                    if action == 'append2keyAND2list':
                        to_write = {key: content,}
                    elif action == 'append2keyASnewlist':
                        to_write = {key: [content],}
                    data.update(to_write)
                    json_file.seek(0)
                    return json.dump(data, json_file) # dump directly the content referenced by the key to the file
            elif action == 'dump_this_dictionary':
                return json.dump(content, json_file) # dump directly the dictionary to the file
            elif action == 'extend_dictionary':
                if isinstance(content, dict):
                    data.update(content)
                    json_file.seek(0)  # return to the beginning of the file
                    return json.dump(data, json_file) # dump directly the dictionary to the file
                else: raise SystemExit('DUMP TO JSON ERROR: You should provide a dictionary')
            else: raise SystemExit('DUMP TO JSON ERROR: action not supported OR key not provided')
            json_file.seek(0)           # return to the beginning of the file
            return json.dump(data, json_file)  # dump the updated data
    else:
        # The file do not exist, create a new one
        with open(file_name, "w") as json_file:
            if action == 'append2keyAND2list' or action == 'append2keyASnewlist':
                to_write = {key: content,}
                return json.dump(to_write, json_file) # dump directly the content referenced by the key to the file
            elif action == 'dump_this_dictionary':
                return json.dump(content, json_file) # dump directly the dictionary to the file
            else: raise SystemExit('DUMP TO JSON ERROR: action not supported')

#-----------------------------------------------------------------------------------------------------------------------

def get_extremities_cells(Fr_list):
    """
    This function returns the extreme points for each of the fracture in the list.

    Args:
        Fr_list (list):         -- the fracture list

    Returns:
        extremeties             -- the [left, rigth, bottom, top] extremeties of the each of the fracture in the list.

    """
    extremities = np.zeros((len(Fr_list), 4), dtype=int)

    for indx, fracture in enumerate(Fr_list):
        max_intrsct1_x = np.argmax(fracture.Ffront[:, 0])
        max_intrsct2_x = np.argmax(fracture.Ffront[:, 2])
        if fracture.Ffront[max_intrsct1_x, 0] > fracture.Ffront[max_intrsct2_x, 2]:
            extremities[indx, 1] = fracture.EltTip[max_intrsct1_x]
        else:
            extremities[indx, 1] = fracture.EltTip[max_intrsct2_x]

        min_intrsct1_x = np.argmin(fracture.Ffront[:, 0])
        min_intrsct2_x = np.argmin(fracture.Ffront[:, 2])
        if fracture.Ffront[min_intrsct1_x, 0] < fracture.Ffront[min_intrsct2_x, 2]:
            extremities[indx, 0] = fracture.EltTip[min_intrsct1_x]
        else:
            extremities[indx, 0] = fracture.EltTip[min_intrsct2_x]

        max_intrsct1_y = np.argmax(fracture.Ffront[:, 1])
        max_intrsct2_y = np.argmax(fracture.Ffront[:, 3])
        if fracture.Ffront[max_intrsct1_y, 1] > fracture.Ffront[max_intrsct2_y, 3]:
            extremities[indx, 3] = fracture.EltTip[max_intrsct1_y]
        else:
            extremities[indx, 3] = fracture.EltTip[max_intrsct2_y]

        min_intrsct1_y = np.argmin(fracture.Ffront[:, 1])
        min_intrsct2_y = np.argmin(fracture.Ffront[:, 3])
        if fracture.Ffront[min_intrsct1_x, 1] < fracture.Ffront[min_intrsct2_x, 3]:
            extremities[indx, 2] = fracture.EltTip[min_intrsct1_y]
        else:
            extremities[indx, 2] = fracture.EltTip[min_intrsct2_y]

    return extremities


#-----------------------------------------------------------------------------------------------------------------------

def get_front_intercepts(fr_list, point):
    """
    This function returns the top, bottom, left and right intercepts on the front of the horizontal and vertical lines
    drawn from the given point.

    Arguments:
         fr_list (list):            -- the given fracture list.
         point (list or ndarray)    -- the point from the horizontal and vertical lines are drawn.

    Returns:
          intercepts (list):        -- list of top, bottom, left and right intercepts for each fracture in the list

    """
    log = logging.getLogger('PyFrac.get_front_intercepts')
    intercepts = []

    for fr in fr_list:
        if isinstance(fr.mesh, int):
            fr_mesh = fr_list[fr.mesh].mesh
        else:
            fr_mesh = fr.mesh
        intrcp_top = intrcp_btm = intrcp_lft = intrcp_rgt = [np.nan]  # set to nan if not available
        try:
            pnt_cell = fr_mesh.locate_element(point[0], point[1])   # the cell in which the given point lie
        except TypeError:
            log.warning("Point is not inside Domain!")
            intercepts.append([0, 0, 0, 0])
        else:
            if pnt_cell not in fr.EltChannel:
                log.warning("Point is not inside fracture!")
                intercepts.append([0, 0, 0, 0])
            else:
                pnt_cell_y = fr_mesh.CenterCoor[pnt_cell, 1]            # the y coordinate of the cell
                cells_x_axis = np.where(fr_mesh.CenterCoor[:, 1] == pnt_cell_y)[0]    # all the cells with the same y coord
                tipCells_x_axis = np.intersect1d(fr.EltTip, cells_x_axis)             # the tip cells with the same y coord

                # the code bellow remove the tip cells which are directly at right and left of the cell containing the point
                # but have the front line partially passing through them. For them, the horizontal line drawn from the given
                # point will pass through the cell but not from the front line.
                if len(tipCells_x_axis) > 2:
                    invalid_cell = np.full(len(tipCells_x_axis), True, dtype=bool)
                    for indx, cell in enumerate(tipCells_x_axis):
                        in_tip_cells = np.where(fr.EltTip == cell)[0]
                        if (point[1] > fr.Ffront[in_tip_cells, 1] and point[1] <= fr.Ffront[in_tip_cells, 3]) or (
                                point[1] < fr.Ffront[in_tip_cells, 1] and point[1] >= fr.Ffront[in_tip_cells, 3]):
                            invalid_cell[indx] = False
                    tipCells_x_axis = np.delete(tipCells_x_axis, np.where(invalid_cell)[0])

                # find out the left and right cells
                if len(tipCells_x_axis) == 2:
                    if fr_mesh.CenterCoor[tipCells_x_axis[0], 0] < point[0]:
                        lft_cell = tipCells_x_axis[0]
                        rgt_cell = tipCells_x_axis[1]
                    else:
                        lft_cell = tipCells_x_axis[1]
                        rgt_cell = tipCells_x_axis[0]
                else:
                    lft_cell = np.nan
                    rgt_cell = np.nan

                pnt_cell_x = fr_mesh.CenterCoor[pnt_cell, 0]
                cells_y_axis = np.where(fr_mesh.CenterCoor[:, 0] == pnt_cell_x)[0]
                tipCells_y_axis = np.intersect1d(fr.EltTip, cells_y_axis)

                # the code bellow remove the tip cells which are directly at top and bottom of the cell containing the point
                # but have the front line partially passing through them. For them, the vertical line drawn from the given
                # point will pass through the cell but not from the front line.
                if len(tipCells_y_axis) > 2:
                    invalid_cell = np.full(len(tipCells_y_axis), True, dtype=bool)
                    for indx, cell in enumerate(tipCells_y_axis):
                        in_tip_cells = np.where(fr.EltTip == cell)[0]
                        if (point[0] > fr.Ffront[in_tip_cells, 0] and point[0] <= fr.Ffront[in_tip_cells, 2]) or (
                                point[0] < fr.Ffront[in_tip_cells, 0] and point[0] >= fr.Ffront[in_tip_cells, 2]):
                            invalid_cell[indx] = False
                    tipCells_y_axis = np.delete(tipCells_y_axis, np.where(invalid_cell)[0])

                if len(tipCells_y_axis) == 2:
                    if fr_mesh.CenterCoor[tipCells_y_axis[0], 1] < point[1]:
                        btm_cell = tipCells_y_axis[0]
                        top_cell = tipCells_y_axis[1]
                    else:
                        btm_cell = tipCells_y_axis[1]
                        top_cell = tipCells_y_axis[0]
                else:
                    btm_cell = np.nan
                    top_cell = np.nan

                top_in_tip = np.where(fr.EltTip == top_cell)[0]
                btm_in_tip = np.where(fr.EltTip == btm_cell)[0]
                lft_in_tip = np.where(fr.EltTip == lft_cell)[0]
                rgt_in_tip = np.where(fr.EltTip == rgt_cell)[0]

                # find the intersection using the equations of the front lines in the tip cells
                if top_in_tip.size > 0:
                    intrcp_top = fr.Ffront[top_in_tip, 3] + \
                             (fr.Ffront[top_in_tip, 3] - fr.Ffront[top_in_tip, 1]) / (
                                         fr.Ffront[top_in_tip, 2] - fr.Ffront[top_in_tip, 0]) * \
                             (point[0] - fr.Ffront[top_in_tip, 2])

                if btm_in_tip.size > 0:
                    intrcp_btm = fr.Ffront[btm_in_tip, 3] + \
                             (fr.Ffront[btm_in_tip, 3] - fr.Ffront[btm_in_tip, 1]) / (
                                         fr.Ffront[btm_in_tip, 2] - fr.Ffront[btm_in_tip, 0]) * \
                             (point[0] - fr.Ffront[btm_in_tip, 2])

                if lft_in_tip.size > 0:
                    intrcp_lft = (point[1] - fr.Ffront[lft_in_tip, 3]) / \
                             (fr.Ffront[lft_in_tip, 3] - fr.Ffront[lft_in_tip, 1]) * (
                                         fr.Ffront[lft_in_tip, 2] - fr.Ffront[lft_in_tip, 0]) + \
                             fr.Ffront[lft_in_tip, 2]

                if rgt_in_tip.size > 0:
                    intrcp_rgt = (point[1] - fr.Ffront[rgt_in_tip, 3]) / \
                             (fr.Ffront[rgt_in_tip, 3] - fr.Ffront[rgt_in_tip, 1]) * (
                                         fr.Ffront[rgt_in_tip, 2] - fr.Ffront[rgt_in_tip, 0]) + \
                             fr.Ffront[rgt_in_tip, 2]

            intercepts.append([intrcp_top[0], intrcp_btm[0], intrcp_lft[0], intrcp_rgt[0]])

    return intercepts


#-----------------------------------------------------------------------------------------------------------------------

def write_properties_csv_file(file_name, properties):
    """ This function writes the properties of a simulatio as a csv file. The csv contains (in a row vector) Eprime, K1c
    , Cl, mu, rho_f, Q and t_inj

        Args:
            file_name (string):         -- the name of the file to be written.
            properties (tuple):         -- the properties of the fracture loaded

    """

    if len(properties[2].injectionRate[0]) > 1:
        output_list = [None] * 7
    else:
        output_list = [None] * 6

    output_list[0] = properties[0].Eprime
    output_list[1] = properties[0].K1c[0]
    output_list[2] = properties[0].Cl
    output_list[3] = properties[1].viscosity
    output_list[4] = properties[1].density

    if len(properties[2].injectionRate[0]) > 1:
        output_list[5] = properties[2].injectionRate[1][0]
        output_list[6] = properties[2].injectionRate[0][1]
    else:
        output_list[5] = properties[2].injectionRate[1][0]


    if file_name[-4:] != '.csv':
        file_name = file_name + '.csv'

    output_list_np = np.asarray(output_list)
    np.savetxt(file_name, output_list_np, delimiter=',')

#-----------------------------------------------------------------------------------------------------------------------

def get_fracture_geometric_parameters(fr_list, head=True, lateral_diking=False):
    # --- Initializing all the solution vectors --- #
    max_breadth = np.full((len(fr_list), 1), np.nan)
    avg_breadth = np.full((len(fr_list), 1), np.nan)
    var_breadth = np.full((len(fr_list), 1), np.nan)
    height = np.full((len(fr_list), 1), np.nan)
    dist_lower_end = np.full((len(fr_list), 1), np.nan)
    dist_max_breadth = np.full((len(fr_list), 1), np.nan)

    if head:
        behind_head_breadth = np.full((len(fr_list), 1), np.nan)
        # dbdz_tail = np.full((len(fr_list), 1), np.nan)
        l_head = np.full((len(fr_list), 1), np.nan)
        wmaxh = np.full((len(fr_list), 1), np.nan)
        pmaxh = np.full((len(fr_list), 1), np.nan)

    if lateral_diking:
        coord_max_w = np.full((len(fr_list), 2), np.nan)
        max_w = np.full((len(fr_list), 1), np.nan)
        p_at_max_w = np.full((len(fr_list), 1), np.nan)
        b_at_max_w = np.full((len(fr_list), 1), np.nan)
        a_plus = np.full((len(fr_list), 1), np.nan)
        a_minus = np.full((len(fr_list), 1), np.nan)
        adjusted_V = np.full((len(fr_list), 1), np.nan)
        l_head = np.full((len(fr_list), 1), np.nan)
        wmaxh = np.full((len(fr_list), 1), np.nan)
        pmaxh = np.full((len(fr_list), 1), np.nan)

    iter = 0

    for jk in fr_list:
        # get the mesh (either stored there or retrieve from location)
        if isinstance(jk.mesh, int):
            fr_mesh = fr_list[jk.mesh].mesh
        else:
            fr_mesh = jk.mesh

        if len(jk.source) != 0:
            left, right = get_Ffront_as_vector(jk, fr_mesh.CenterCoor[jk.source[0], ::])[1:]
        else:
            left, right = get_Ffront_as_vector(jk, [0., 0.])[1:]

        if left.shape[0] == right.shape[0]:
            breadth = np.vstack((np.abs(left - right)[::, 0], left[::, 1]))
        else:
            breadth = np.vstack((np.abs(left[:np.min((left.shape[0], right.shape[0])), :] -
                                        right[:np.min((left.shape[0], right.shape[0])), :])[:, 0],
                                 np.vstack((left[:np.min((left.shape[0], right.shape[0])), 1],
                                            right[:np.min((left.shape[0], right.shape[0])), 1]))
                                 [np.argmin((left.shape[0], right.shape[0])), :]))

        max_breadth[iter] = np.max(breadth[0, ::])

        dist_max_breadth[iter] = np.max(np.asarray([np.min(breadth[1, breadth[0, ::] >= 0.975
                                                                          * max_breadth[iter]]), 0.])) # we account for
        # a 2.5% error on the breadth

        avg_breadth[iter] = np.mean(breadth[0, ::])
        var_breadth[iter] = np.var(breadth[0, ::])

        height[iter] = np.abs(np.max(np.hstack((jk.Ffront[::, 1], jk.Ffront[::, 3]))) -
                              np.min(np.hstack((jk.Ffront[::, 1], jk.Ffront[::, 3]))))
        dist_lower_end[iter] = np.abs(np.min(np.hstack((jk.Ffront[::, 1], jk.Ffront[::, 3]))))

        if head:
            pressure, line, cells = get_fracture_variable_slice_cell_center(jk.pNet, jk.mesh, orientation='vertical')
            opening, line, cells = get_fracture_variable_slice_cell_center(jk.w, jk.mesh, orientation='vertical')
            z_coord = jk.mesh.CenterCoor[cells][:, 1]
            ind_zero = np.argmin(np.abs(z_coord))
            ind_max_w = np.argmax(opening)
            ind_tip = np.argwhere((np.diff(np.sign(np.diff(opening))) != 0) * 1)[-1][0] + 1
            if ind_max_w not in set(np.arange(ind_zero - 4, ind_zero + 4)):  # the max is not at the origin so it must be in the head
                # then we can check if in between the injection point and the max opening (in the head) we have a sign
                # sign change.
                if ((np.diff(np.sign(np.diff(opening[ind_zero + 1:ind_max_w - 1]))) != 0) * 1).any():
                    # If we have such a sign change we have a part where the opening reduces and then restarts to grow.
                    # So we search for the sign change closest to the head as the beginning of the head
                    l_head[iter] = np.max(np.hstack((jk.Ffront[::, 1], jk.Ffront[::, 3]))) - \
                                   z_coord[ind_zero + 1:ind_max_w - 1][np.where(((np.diff(np.sign(
                                       np.diff(opening[ind_zero + 1:ind_max_w - 1]))) != 0) * 1) == 1)[0][-1] + 1]
                else:
                    # If we don't have a sign change we need to search for the inflexion point second closest to the maximum
                    # opening. The closest one is where the opening starts to reduce again towards the max
                    try:
                        secDer = np.gradient(np.gradient(opening[ind_zero + 1:ind_max_w - 1],
                                                         z_coord[ind_zero + 1:ind_max_w - 1]),
                                             z_coord[ind_zero + 1:ind_max_w - 1])
                    except:
                        secDer = np.gradient(np.gradient(opening[ind_max_w + 1:ind_zero - 1],
                                                         z_coord[ind_max_w + 1:ind_zero - 1]),
                                             z_coord[ind_max_w + 1:ind_zero - 1])
                    if len(np.argwhere((np.diff(np.sign(secDer)) != 0) * 1)) < 2:
                        l_head[iter] = np.max(np.hstack((jk.Ffront[::, 1], jk.Ffront[::, 3])))
                    else:
                        try:
                            l_head[iter] = np.max(np.hstack((jk.Ffront[::, 1], jk.Ffront[::, 3]))) - \
                                           z_coord[ind_zero + 1:ind_max_w - 1][
                                               np.argwhere((np.diff(np.sign(secDer)) != 0) * 1)[-2][0] + 1]
                        except:
                            l_head[iter] = np.max(np.hstack((jk.Ffront[::, 1], jk.Ffront[::, 3]))) - \
                                           z_coord[ind_max_w + 1:ind_zero - 1][
                                               np.argwhere((np.diff(np.sign(secDer)) != 0) * 1)[-2][0] + 1]
            elif not (np.sign(
                    np.diff(opening[ind_zero + 1:ind_tip - 2])) == 1.0)[4:].any():  # the max is at the origin: and there
                # is no sign change so we should still be radial
                # Note: sometimes we get a numerical non-linearity at the tip so that is what we want to exclude here.
                # The -2 is thus used to exclude the elements closest to the tip.
                l_head[iter] = np.max(np.hstack((jk.Ffront[::, 1], jk.Ffront[::, 3])))
            else:
                # So we have the max at the origin but we have a head. We need to get where the max of the head
                # is located as to solve again for the inflection point from there on
                ind_inc = np.argwhere((np.sign(np.diff(opening[ind_zero + 1:])) == 1) * 1)[0][0] - 1 # Index where we
                                                                                        # get out of the source influence
                ind_w_max_head = np.argmax(opening[ind_zero + 1:][ind_inc:]) # index of the maximum opening in the head
                try:
                    l_head[iter] = np.max(np.hstack((jk.Ffront[::, 1], jk.Ffront[::, 3]))) - \
                                       z_coord[ind_zero + 1:][ind_inc:][:ind_w_max_head][np.where(((np.diff(np.sign(
                                           np.diff(opening[ind_zero + 1:][ind_inc:][:ind_w_max_head]))) != 0) * 1) == 1)
                                                                                         [0][-1]+ 1]
                except:
                    l_head[iter] = np.max(np.hstack((jk.Ffront[::, 1], jk.Ffront[::, 3])))

            # --- This works fine so far but now we use it as an estimate and get the real length from the pressure
            # Note: but let's only do this if we are no longer radial!
            if np.max(np.hstack((jk.Ffront[::, 1], jk.Ffront[::, 3]))) - l_head[iter] <= dist_max_breadth[iter]:
                behind_head_breadth[iter] = max_breadth[iter]
                wmaxh[iter] = np.max(opening)
                pmaxh[iter] = np.max(pressure)
            else:
                indEndhead = np.abs(z_coord - (np.max(np.hstack((jk.Ffront[::, 1], jk.Ffront[::, 3]))) -
                                               l_head[iter])).argmin()
                indMinP = np.argmin(pressure[indEndhead:ind_tip])
                if indMinP != 0 and indMinP != len(pressure[indEndhead:ind_tip]) - 1:
                    l_head[iter] = np.max(np.hstack((jk.Ffront[::, 1], jk.Ffront[::, 3]))) - \
                                   z_coord[indEndhead:ind_tip][indMinP]

                behind_head_breadth[iter] = breadth[0, np.abs(breadth[1, ::] -
                                                              (np.max(np.hstack((jk.Ffront[::, 1], jk.Ffront[::, 3]))) -
                                                               l_head[iter])).argmin()]

                wmaxh[iter] = np.max(opening[indEndhead:])

                pmaxh[iter] = np.max(pressure[indEndhead:])
            # # --- We also need to get the gradient of the breadth
            # # Note: we only want it between the max breadth and the head
            # ind_start = np.argwhere(breadth[1, ::] == np.min(breadth[1, breadth[0, ::] == max_breadth[iter]])).flatten()[0]
            # ind_end = np.max([np.abs(breadth[1, ::] - (np.max(np.hstack((jk.Ffront[::, 1], jk.Ffront[::, 3])))
            #                                    - l_head[iter])).argmin(), ind_start + 1])
            # dbdz = np.gradient(breadth[0, ind_start:ind_end + 1], breadth[1, ind_start:ind_end + 1])
            # if ind_end == ind_start + 1:
            #     dbdz_tail[iter] = 0.
            # else:
            #     dbdz_tail[iter] = np.mean(dbdz)

        if lateral_diking:
            ind_max_w = np.argmax(jk.w)
            coord_max_w[iter] = jk.mesh.CenterCoor[ind_max_w]
            max_w[iter] = jk.w[ind_max_w]
            p_at_max_w[iter] = jk.pNet[ind_max_w]
            idx = (np.abs(breadth[1] - coord_max_w[iter][1])).argmin()
            b_at_max_w[iter] = breadth[0][idx]

            el_trapped_tip = np.asarray(np.where(jk.mesh.CenterCoor[jk.EltTip, 1] <= a_minus[iter])).flatten()
            el_trapped_channel = np.asarray(np.where(jk.mesh.CenterCoor[jk.EltChannel, 1] <= a_minus[iter])).flatten()
            v_trapped = (np.sum(jk.w[jk.EltChannel[el_trapped_channel]])
                                   + np.sum(jk.w[jk.EltTip[el_trapped_tip]] *
                                            jk.FillF[el_trapped_tip])) * jk.mesh.EltArea
            v_total = jk.mesh.EltArea * (np.sum(jk.w[jk.EltTip]*jk.FillF) + np.sum(jk.w[jk.EltChannel]))
            adjusted_V[iter] = v_total - v_trapped

        iter = iter + 1

    if head and lateral_diking:
        out_dict = {
          'l': height.flatten().flatten(),
          'bmax': max_breadth.flatten().flatten(),
          'bavg': avg_breadth.flatten().flatten(),
          'bvar': var_breadth.flatten(),
          'bhead': behind_head_breadth.flatten(),
          'dle': dist_lower_end.flatten().flatten(),
          'dbmax': dist_max_breadth.flatten().flatten(),
          'lhead': l_head.flatten().flatten(),
          'whmax': wmaxh.flatten().flatten(),
          'phmax': pmaxh.flatten().flatten(),
          'coord max w': coord_max_w.flatten().flatten(),
          'max w': max_w.flatten().flatten(),
          'p at max w': p_at_max_w.flatten().flatten(),
          'b at max w': b_at_max_w.flatten().flatten(),
          'V_adjusted': adjusted_V
        }
    elif head:
        out_dict = {
          'l': height.flatten().flatten(),
          'bmax': max_breadth.flatten().flatten(),
          'bavg': avg_breadth.flatten().flatten(),
          'bvar': var_breadth.flatten(),
          'bhead': behind_head_breadth.flatten(),
          'dle': dist_lower_end.flatten().flatten(),
          'dbmax': dist_max_breadth.flatten().flatten(),
          'lhead': l_head.flatten().flatten(),
          'whmax': wmaxh.flatten().flatten(),
          'phmax': pmaxh.flatten().flatten()
        }
    elif lateral_diking:
        out_dict = {
          'l': height.flatten().flatten(),
          'bmax': max_breadth.flatten().flatten(),
          'bavg': avg_breadth.flatten().flatten(),
          'bvar': var_breadth.flatten(),
          'dle': dist_lower_end.flatten().flatten(),
          'dbmax': dist_max_breadth.flatten().flatten(),
          'coord max w': coord_max_w.flatten().flatten(),
          'max w': max_w.flatten().flatten(),
          'p at max w': p_at_max_w.flatten().flatten(),
          'b at max w': b_at_max_w.flatten().flatten(),
          'V_adjusted': adjusted_V
        }
    else:
        out_dict = {
          'l': height.flatten().flatten(),
          'bmax': max_breadth.flatten().flatten(),
          'bavg': avg_breadth.flatten().flatten(),
          'bvar': var_breadth.flatten(),
          'dle': dist_lower_end.flatten().flatten(),
          'dbmax': dist_max_breadth.flatten().flatten()
        }

    return out_dict

#-----------------------------------------------------------------------------------------------------------------------

# This is the function to get the breadth
def get_local_geometry(Fr_list, points):

    breadth = np.empty((len(Fr_list),len(points)))
    breadth[:] = np.nan
    height = np.empty((len(Fr_list),len(points)))
    height[:] = np.nan
    for p in range(len(points)):
        for Fr in range(len(Fr_list)):
            # get the mesh (either stored there or retrieve from location)
            if isinstance(Fr_list[Fr].mesh, int):
                fr_mesh = Fr_list[Fr_list[Fr].mesh].mesh
            else:
                fr_mesh = Fr_list[Fr].mesh
            fr = Fr_list[Fr]
            intrcp_lft = intrcp_rgt = [np.nan]  # set to nan if not available
            pnt_cell = fr_mesh.locate_element(points[p][0], points[p][1])  # the cell in which the given point lie
            if pnt_cell not in fr.EltChannel:
                print("Point is not inside fracture!")
            else:
                pnt_cell_y = fr_mesh.CenterCoor[pnt_cell, 1]  # the y coordinate of the cell
                cells_x_axis = np.where(fr_mesh.CenterCoor[:, 1] == pnt_cell_y)[0]  # all the cells with the same y coord
                tipCells_x_axis = np.intersect1d(fr.EltTip, cells_x_axis)  # the tip cells with the same y coord

                pnt_cell_x = fr_mesh.CenterCoor[pnt_cell, 0]  # the x coordinate of the cell
                cells_y_axis = np.where(fr_mesh.CenterCoor[:, 0] == pnt_cell_x)[0]  # all the cells with the same y coord
                tipCells_y_axis = np.intersect1d(fr.EltTip, cells_y_axis)  # the tip cells with the same y coord

                # find out the left and right cells
                cells_left = tipCells_x_axis[fr_mesh.CenterCoor[tipCells_x_axis, 0] < fr_mesh.CenterCoor[pnt_cell, 0]]
                cells_right = tipCells_x_axis[fr_mesh.CenterCoor[tipCells_x_axis, 0] > fr_mesh.CenterCoor[pnt_cell, 0]]
                lft_cell = cells_left[abs(fr_mesh.CenterCoor[cells_left, 0] - fr_mesh.CenterCoor[pnt_cell, 0]) ==
                                      min(abs(fr_mesh.CenterCoor[cells_left, 0] - fr_mesh.CenterCoor[pnt_cell, 0]))]
                rgt_cell = cells_right[abs(fr_mesh.CenterCoor[pnt_cell, 0] - fr_mesh.CenterCoor[cells_right, 0]) ==
                                      min(abs(fr_mesh.CenterCoor[pnt_cell, 0] - fr_mesh.CenterCoor[cells_right, 0]))]

                lft_in_tip = np.where(fr.EltTip == lft_cell)[0]
                rgt_in_tip = np.where(fr.EltTip == rgt_cell)[0]

                # find out the top and bottom cells
                cells_top = tipCells_y_axis[fr_mesh.CenterCoor[tipCells_y_axis, 1] < fr_mesh.CenterCoor[pnt_cell, 1]]
                cells_bottom = tipCells_y_axis[fr_mesh.CenterCoor[tipCells_y_axis, 1] > fr_mesh.CenterCoor[pnt_cell, 1]]
                top_cell = cells_top[abs(fr_mesh.CenterCoor[cells_top, 1] - fr_mesh.CenterCoor[pnt_cell, 1]) ==
                                      min(abs(fr_mesh.CenterCoor[cells_top, 1] - fr_mesh.CenterCoor[pnt_cell, 1]))]
                bot_cell = cells_bottom[abs(fr_mesh.CenterCoor[pnt_cell, 1] - fr_mesh.CenterCoor[cells_bottom, 1]) ==
                                      min(abs(fr_mesh.CenterCoor[pnt_cell, 1] - fr_mesh.CenterCoor[cells_bottom, 1]))]

                top_in_tip = np.where(fr.EltTip == top_cell)[0]
                bot_in_tip = np.where(fr.EltTip == bot_cell)[0]

                # find the intersection using the equations of the front lines in the tip cells
                if lft_in_tip.size > 0:
                    intrcp_lft = (points[p][1] - fr.Ffront[lft_in_tip, 3]) / \
                                 (fr.Ffront[lft_in_tip, 3] - fr.Ffront[lft_in_tip, 1]) * (
                                         fr.Ffront[lft_in_tip, 2] - fr.Ffront[lft_in_tip, 0]) + \
                                 fr.Ffront[lft_in_tip, 2]

                if rgt_in_tip.size > 0:
                    intrcp_rgt = (points[p][1] - fr.Ffront[rgt_in_tip, 3]) / \
                                 (fr.Ffront[rgt_in_tip, 3] - fr.Ffront[rgt_in_tip, 1]) * (
                                         fr.Ffront[rgt_in_tip, 2] - fr.Ffront[rgt_in_tip, 0]) + \
                                 fr.Ffront[rgt_in_tip, 2]

                # find the intersection using the equations of the front lines in the tip cells
                if top_in_tip.size > 0:
                    intrcp_top = (points[p][0] - fr.Ffront[top_in_tip, 2]) / \
                                 (fr.Ffront[top_in_tip, 2] - fr.Ffront[top_in_tip, 0]) * (
                                         fr.Ffront[top_in_tip, 3] - fr.Ffront[top_in_tip, 1]) + \
                                 fr.Ffront[top_in_tip, 3]

                if bot_in_tip.size > 0:
                    intrcp_bot = (points[p][0] - fr.Ffront[top_in_tip, 2]) / \
                                 (fr.Ffront[top_in_tip, 2] - fr.Ffront[top_in_tip, 0]) * (
                                         fr.Ffront[top_in_tip, 3] - fr.Ffront[top_in_tip, 1]) + \
                                 fr.Ffront[top_in_tip, 3]

                breadth[Fr, p] = intrcp_rgt[0]-intrcp_lft[0]
                height[Fr, p] = intrcp_top[0]-intrcp_bot[0]

    return breadth, height

#-----------------------------------------------------------------------------------------------------------------------

def get_fracture_head_volume(fr_list, geometric_data=None):
    # get the geometric data if necessary
    if geometric_data == None:
        geometric_data = get_fracture_geometric_parameters(fr_list)
    # instantiate head and total volume
    v_head = np.full((len(fr_list), 1), np.nan)
    v_tot = np.full((len(fr_list), 1), np.nan)
    #loop over time steps to get the volume
    iter = 0
    for jk in fr_list:
        # get the mesh (either stored there or retrieve from location)
        if isinstance(jk.mesh, int):
            fr_mesh = fr_list[jk.mesh].mesh
        else:
            fr_mesh = jk.mesh
        z_head = np.max(np.hstack((jk.Ffront[::, 1], jk.Ffront[::, 3])))-geometric_data['lhead'][iter]
        el_head_tip = np.asarray(np.where(fr_mesh.CenterCoor[jk.EltTip, 1] >= z_head)).flatten()
        el_head_channel = np.asarray(np.where(fr_mesh.CenterCoor[jk.EltChannel, 1] >= z_head)).flatten()
        v_head[iter] = (np.sum(jk.w[jk.EltChannel[el_head_channel]]) + np.sum(jk.w[jk.EltTip[el_head_tip]] *
                                                                              jk.FillF[el_head_tip])) * fr_mesh.EltArea
        v_tot[iter] = fr_mesh.EltArea * (np.sum(jk.w[jk.EltTip]*jk.FillF) + np.sum(jk.w[jk.EltChannel]))
        iter += 1
    return v_tot.flatten(), v_head.flatten()

#-----------------------------------------------------------------------------------------------------------------------

def init_list_of_objects(size):
    list_of_objects = list()
    for i in range(0, size):
        list_of_objects.append(list()) #different object reference each time
    return list_of_objects

#-----------------------------------------------------------------------------------------------------------------------
def get_breadth_at_point(Fr_list, points):

    log = logging.getLogger('PyFrac.get_front_intercepts')
    breadth = init_list_of_objects(len(Fr_list))
    Az = init_list_of_objects(len(Fr_list))
    for p in range(len(points)):
        for Fr in range(len(Fr_list)):
            if isinstance(fr.mesh, int):
                fr_mesh = Fr_list[fr.mesh].mesh
            else:
                fr_mesh = fr.mesh
            if p == 0:
                breadth[Fr] = len(points) * [0.]
                Az[Fr] = len(points) * [0.]
            fr = Fr_list[Fr]
            intrcp_lft = intrcp_rgt = [np.nan]  # set to nan if not available
            pnt_cell = fr_mesh.locate_element(points[p][0], points[p][1])  # the cell in which the given point lie
            if pnt_cell not in fr.EltChannel:
                log.warning("Point is not inside fracture!")
            else:
                pnt_cell_y = fr_mesh.CenterCoor[pnt_cell, 1]  # the y coordinate of the cell
                cells_x_axis = np.where(fr_mesh.CenterCoor[:, 1] == pnt_cell_y)[0] # all the cells with the same y coord
                tipCells_x_axis = np.intersect1d(fr.EltTip, cells_x_axis)  # the tip cells with the same y coord

                # find out the left and right cells
                cells_left = tipCells_x_axis[fr_mesh.CenterCoor[tipCells_x_axis, 0] < fr_mesh.CenterCoor[pnt_cell, 0]]
                cells_right = tipCells_x_axis[fr_mesh.CenterCoor[tipCells_x_axis, 0] > fr_mesh.CenterCoor[pnt_cell, 0]]
                lft_cell = cells_left[abs(fr_mesh.CenterCoor[cells_left, 0] - fr_mesh.CenterCoor[pnt_cell, 0]) ==
                                      min(abs(fr_mesh.CenterCoor[cells_left, 0] - fr_mesh.CenterCoor[pnt_cell, 0]))]
                rgt_cell = cells_right[abs(fr_mesh.CenterCoor[pnt_cell, 0] - fr_mesh.CenterCoor[cells_right, 0]) ==
                                      min(abs(fr_mesh.CenterCoor[pnt_cell, 0] - fr_mesh.CenterCoor[cells_right, 0]))]

                lft_in_tip = np.where(fr.EltTip == lft_cell)[0]
                rgt_in_tip = np.where(fr.EltTip == rgt_cell)[0]

                # find the intersection using the equations of the front lines in the tip cells
                if lft_in_tip.size > 0:
                    intrcp_lft = (points[p][1] - fr.Ffront[lft_in_tip, 3]) / \
                                 (fr.Ffront[lft_in_tip, 3] - fr.Ffront[lft_in_tip, 1]) * (
                                         fr.Ffront[lft_in_tip, 2] - fr.Ffront[lft_in_tip, 0]) + \
                                 fr.Ffront[lft_in_tip, 2]

                if rgt_in_tip.size > 0:
                    intrcp_rgt = (points[p][1] - fr.Ffront[rgt_in_tip, 3]) / \
                                 (fr.Ffront[rgt_in_tip, 3] - fr.Ffront[rgt_in_tip, 1]) * (
                                         fr.Ffront[rgt_in_tip, 2] - fr.Ffront[rgt_in_tip, 0]) + \
                                 fr.Ffront[rgt_in_tip, 2]

                breadth[Fr][p] = intrcp_rgt[0]-intrcp_lft[0]

                Az[Fr][p] = np.sum(fr.w[cells_x_axis]) * fr_mesh.hx

    return breadth, Az

#-----------------------------------------------------------------------------------------------------------------------

def get_Ffront_as_vector(frac, inj_p):

    mask13 = np.all(np.asarray([frac.Ffront[::, 0] <= inj_p[0], frac.Ffront[::, 1] <= inj_p[1]]), axis=0)
    mask24 = np.all(np.asarray([frac.Ffront[::, 2] <= inj_p[0], frac.Ffront[::, 3] <= inj_p[1]]), axis=0)
    lowLef = np.concatenate((frac.Ffront[mask13, :2:], frac.Ffront[mask24, 2::]), axis=0)
    lowLef = lowLef[np.flip(lowLef[:, 1].argsort()), ::]

    mask13 = np.all(np.asarray([frac.Ffront[::, 0] >= inj_p[0], frac.Ffront[::, 1] <= inj_p[1]]), axis=0)
    mask24 = np.all(np.asarray([frac.Ffront[::, 2] >= inj_p[0], frac.Ffront[::, 3] <= inj_p[1]]), axis=0)
    lowRig = np.concatenate((frac.Ffront[mask13, :2:], frac.Ffront[mask24, 2::]), axis=0)
    lowRig = lowRig[lowRig[:, 1].argsort(), ::]

    mask13 = np.all(np.asarray([frac.Ffront[::, 0] <= inj_p[0], frac.Ffront[::, 1] >= inj_p[1]]), axis=0)
    mask24 = np.all(np.asarray([frac.Ffront[::, 2] <= inj_p[0], frac.Ffront[::, 3] >= inj_p[1]]), axis=0)
    upLef = np.concatenate((frac.Ffront[mask13, :2:], frac.Ffront[mask24, 2::]), axis=0)
    upLef = upLef[np.flip(upLef[:, 1].argsort()), ::]

    mask13 = np.all(np.asarray([frac.Ffront[::, 0] >= inj_p[0], frac.Ffront[::, 1] >= inj_p[1]]), axis=0)
    mask24 = np.all(np.asarray([frac.Ffront[::, 2] >= inj_p[0], frac.Ffront[::, 3] >= inj_p[1]]), axis=0)
    upRig = np.concatenate((frac.Ffront[mask13, :2:], frac.Ffront[mask24, 2::]), axis=0)
    upRig = upRig[upRig[:, 1].argsort(), ::]

    if len(lowLef) != 0:
        Ffront = np.concatenate((lowLef, lowRig, upRig, upLef, np.asarray([lowLef[0, :]])), axis=0)
    else:
        Ffront = np.concatenate((upRig, upLef, np.asarray([upRig[0, :]])), axis=0)

    left = np.concatenate((lowLef, upLef), axis=0)
    left = np.unique(left, axis=0)[np.unique(left, axis=0)[::, 1].argsort(), ::]

    right = np.concatenate((lowRig, upRig), axis=0)
    right = np.unique(right, axis=0)[np.unique(right, axis=0)[::, 1].argsort(), ::]

    return Ffront, left, right


#-----------------------------------------------------------------------------------------------------------------------

def get_fracture_fp(fr_list):
    fp_list = []
    iter = 0

    for jk in fr_list:
        fr = copy.deepcopy(jk)
        if isinstance(jk.mesh, int):
            fr.mesh = fr_list[jk.mesh].mesh
        else:
            fr.mesh = jk.mesh
        if len(jk.source) != 0:
            fp_list.append(get_Ffront_as_vector(fr, fr.mesh.CenterCoor[jk.source[0], ::])[0])
        else:
            fp_list.append(get_Ffront_as_vector(fr, [0., 0])[0])
        iter = iter + 1

    return fp_list

#-----------------------------------------------------------------------------------------------------------------------

def get_velocity_as_vector(Solid, Fluid, Fr_list, SimProp): #CP 2020
    """This function gets the velocity components of the fluid flux for a given list of fractures

    :param Solid: Instance of the class MaterialProperties - see related documentation
    :param Fluid: Instance of the class FluidProperties - see related documentation
    :param Fr_list: List of Instances of the class Fracture - see related documentation
    :return: List containing a matrix with the information about the fluid velocity for each of the edges of any mesh element,
             List of time stations
    """
    fluid_vel_list = []
    time_srs = []
    for i in Fr_list:
        if isinstance(i.mesh, int):
            fr_mesh = Fr_list[i.mesh].mesh
        else:
            fr_mesh = i.mesh

        fluid_flux, \
        fluid_vel, \
        Rey_num, \
        fluid_flux_components, \
        fluid_vel_components = calculate_fluid_flow_characteristics_laminar(i.w,
                                                                            i.pFluid,
                                                                            Solid.SigmaO,
                                                                            fr_mesh,
                                                                            i.EltCrack,
                                                                            i.InCrack,
                                                                            Fluid.muPrime,
                                                                            Fluid.density,
                                                                            SimProp,
                                                                            Solid)
        # fluid_vel_components_for_one_elem = [fx left edge, fy left edge, fx right edge, fy right edge, fx bottom edge, fy bottom edge, fx top edge, fy top edge]
        #
        #                 6  7
        #               (ux,uy)
        #           o---top edge---o
        #     0  1  |              |    2  3
        #   (ux,uy)left          right(ux,uy)
        #           |              |
        #           o-bottom edge--o
        #               (ux,uy)
        #                 4  5
        #
        fluid_vel_list.append(fluid_vel_components)
        time_srs.append(i.time)

    return fluid_vel_list, time_srs

#-----------------------------------------------------------------------------------------------------------------------
def get_velocity_slice(Solid, Fluid, Fr_list, initial_point, simProp, vel_direction = 'ux',orientation='horizontal'): #CP 2020
    """
    This function returns, at each time station, the velocity component in x or y direction along a horizontal or vertical section passing
    through a given point.

    WARNING: ASSUMING NO MESH COARSENING OR REMESHING WITH DOMAIN COMPRESSION

    :param Solid: Instance of the class MaterialProperties - see related documentation
    :param Fluid: Instance of the class FluidProperties - see related documentation
    :param Fr_list: List of Instances of the class Fracture - see related documentation
    :param initial_point: coordinates of the point where to draw the slice
    :param vel_direction: component of the velocity vector, it can be 'ux' or 'uy'
    :param orientation: it can be 'horizontal' or 'vertical'
    :return: set of velocities
             set of times
             set of points along the slice, where the velocity is given
    """

    # initial_point - of the slice
    fluid_vel_list, time_srs = get_velocity_as_vector(Solid, Fluid, Fr_list, simProp)
    nOFtimes = len(time_srs)
    # fluid_vel_list is a list and each entry contains a matrix with the information about the fluid velocity for each of the edges of any mesh element
    # fluid_vel_components_for_one_elem = [fx left edge, fy left edge, fx right edge, fy right edge, fx bottom edge, fy bottom edge, fx top edge, fy top edge]
    #
    #                 6  7
    #               (ux,uy)
    #           o---top edge---o
    #     0  1  |              |    2  3
    #   (ux,uy)left          right(ux,uy)
    #           |              |
    #           o-bottom edge--o
    #               (ux,uy)
    #                 4  5
    #

    list_of_sampling_lines = []
    list_of_fluid_vel_lists = []

    for i in range(nOFtimes): #each fr has its own mesh
        if isinstance(Fr_list[i], int):
            fr_mesh = Fr_list[Fr_list[i]].mesh
        else:
            fr_mesh = Fr_list[i].mesh
        # 1) get the coordinates of the points in the slices
        vector_to_be_lost = np.zeros(fr_mesh.NumberOfElts,dtype=int)
        NotUsd_var_values, sampling_line_center, sampling_cells = get_fracture_variable_slice_cell_center(vector_to_be_lost,
                                                                                                            fr_mesh,
                                                                                                            point = initial_point,
                                                                                                            orientation = orientation)
        hx = fr_mesh.hx # element horizontal size
        hy = fr_mesh.hy # element vertical size
        # get the coordinates along the slice where you are getting the values
        if vel_direction ==  'ux' and orientation == 'horizontal': # take ux on the vertical edges
            indx1 = 0 #left
            indx2 = 2 #right
            sampling_line_center1=sampling_line_center-hx*.5
            sampling_line_center2=sampling_line_center+hx*.5
        elif vel_direction ==  'ux' and orientation == 'vertical': # take ux on the horizontal edges
            indx1 = 4 #bottom
            indx2 = 6 #top
            sampling_line_center1=sampling_line_center-hy*.5
            sampling_line_center2=sampling_line_center+hy*.5
        elif vel_direction == 'uy' and orientation == 'horizontal': # take uy on the vertical edges
            indx1 = 1 #left
            indx2 = 3 #rigt
            sampling_line_center1=sampling_line_center-hx*.5
            sampling_line_center2=sampling_line_center+hx*.5
        elif vel_direction == 'uy' and orientation == 'vertical': # take uy on the horizontal edges
            indx1 = 5 #bottom
            indx2 = 7 #top
            sampling_line_center1=sampling_line_center-hy*.5
            sampling_line_center2=sampling_line_center+hy*.5

        #combining the two list of locations where I get the velocity
        sampling_line = [None] * (len(sampling_line_center1) + len(sampling_line_center2))
        sampling_line[::2] = sampling_line_center1
        sampling_line[1::2] = sampling_line_center2
        list_of_sampling_lines.append(sampling_line)


        # 2) get the velocity values
        EltCrack_i = Fr_list[i].EltCrack
        fluid_vel_list_i = fluid_vel_list[i]

        vector_to_be_lost1 = np.zeros(fr_mesh.NumberOfElts, dtype=float)
        vector_to_be_lost1[EltCrack_i] = fluid_vel_list_i[indx1,:]
        vector_to_be_lost2 = np.zeros(fr_mesh.NumberOfElts, dtype=float)
        vector_to_be_lost2[EltCrack_i] = fluid_vel_list_i[indx2,:]

        fluid_vel_list_final_i = [None] * (len(vector_to_be_lost1[sampling_cells]) + len(vector_to_be_lost2[sampling_cells]))
        fluid_vel_list_final_i[::2] = vector_to_be_lost1[sampling_cells]
        fluid_vel_list_final_i[1::2] = vector_to_be_lost2[sampling_cells]
        list_of_fluid_vel_lists.append(fluid_vel_list_final_i)

    return list_of_fluid_vel_lists, time_srs, list_of_sampling_lines

#-----------------------------------------------------------------------------------------------------------------------

def get_power_split(Solid, Fluid, SimProp, Fr_list, head_split=None): #AM 2022, based on CP routines
    """This function returns the powers for a given series of fractures

    :param Solid: Instance of the class MaterialProperties - see related documentation
    :param Fluid: Instance of the class FluidProperties - see related documentation
    :param SimProp: Instance of the class SimulationProperties - see related documentation
    :param Fr_list: List of Instances of the class Fracture - see related documentation
    :return: A set of lists containing respectively: The total External, total internal, viscous, fracture, elastic,
             leak-off, elastic stress (from background stress) powers and a list of the corresponding times.
    """

    # * -- For back compatibility, we check if the fracture properties have a stored gravity Value -- * #
    if not hasattr(Solid, 'gravityValue'):
        Solid.gravityValue = np.zeros((2 * len(Solid.SigmaO),), float)
        Solid.gravityValue[1:-1:2] = -9.81

    # * -- For back compatibility, we check if the fracture properties have a stored density -- * #
    hasDensity = True
    if not hasattr(Solid, 'density'):
        hasDensity = False

    # * -- We check if we get a negative background stress, not physical, we correct it by adding a constant -- * #
    if min(Fr_list[-1].pFluid - Fr_list[-1].pNet) < 0.:
        offset = -10 * min(Fr_list[-1].pFluid - Fr_list[-1].pNet)
        for iter in range(len(Fr_list)):
            Fr_list[iter].pFluid[Fr_list[iter].EltCrack] += offset
        Solid.SigmaO += offset

    # * -- Preparing the output lists -- * #
    Viscous_P = init_list_of_objects(len(Fr_list)-1)
    Fracture_P = init_list_of_objects(len(Fr_list)-1)
    Elastic_P = init_list_of_objects(len(Fr_list)-1)
    LeakOff_P = init_list_of_objects(len(Fr_list) - 1)

    Injection_P = init_list_of_objects(len(Fr_list) - 1)
    Gravity_P = init_list_of_objects(len(Fr_list) - 1)

    power_time_steps = init_list_of_objects(len(Fr_list)-1)

    # * -- Loop on the fractures -- * #
    iter = 0 # We initiate a iterator to correctly assemble the elements
    for i in range(1, len(Fr_list)):
        fr_i = Fr_list[i]       # fracture of the considered time step
        fr_im1 = Fr_list[i-1]   # fracture of the previous time step

        # - Export the mesh for both fractures - #
        if isinstance(fr_i.mesh, int):
            fr_i_mesh = Fr_list[fr_i.mesh]
        else:
            fr_i_mesh = fr_i.mesh

        if isinstance(fr_im1.mesh, int):
            fr_im1_mesh = Fr_list[fr_im1.mesh]
        else:
            fr_im1_mesh = fr_im1.mesh

        # * -- check if fr_i and fr_im1 have the same mesh -- * #
        # Note: The routine only works if we do not have re-meshing in between
        if (fr_i_mesh.hx == fr_im1_mesh.hx and
            fr_i_mesh.hy == fr_im1_mesh.hy and
            fr_i_mesh.nx == fr_im1_mesh.nx and
            fr_i_mesh.ny == fr_im1_mesh.ny and
            fr_i_mesh.domainLimits[0] == fr_im1_mesh.domainLimits[0] and
            fr_i_mesh.domainLimits[1] == fr_im1_mesh.domainLimits[1] and
            fr_i_mesh.domainLimits[2] == fr_im1_mesh.domainLimits[2] and
            fr_i_mesh.domainLimits[3] == fr_im1_mesh.domainLimits[3]):

            # - Do we have a density (back compatibility) - #
            if hasDensity:
                Density_function = True
                # - Check if a density function exists - #
                try:
                    Solid.DensityFunc[0., 0.]
                except:
                    Density_function = False

                # - We have a density function - #
                if Density_function:
                    Solid.density = np.empty((fr_i_mesh.NumberOfElts,), dtype=np.float64)
                    for i in range(fr_i_mesh.NumberOfElts):
                        Solid.density[i] = Solid.DensityFunc(fr_i_mesh.CenterCoor[i, 0], fr_i_mesh.CenterCoor[i, 1])
                else:
                    Solid.density = Solid.density[0] * np.ones((fr_i_mesh.NumberOfElts,), float)
            else:
                # - Generally a density of 2700 was used for relevant simulations before the density was added - #
                Solid.density = 2700 * np.ones((fr_i_mesh.NumberOfElts,), float)

            # - Get the length and location of the front segments and their intersection with the grid - #
            l, x_m, y_m, x, y = get_l_Ffront_ordered_as_v(fr_i.Ffront, fr_i.EltTip, fr_i_mesh)

            if head_split == None:
                cells = [np.arange(fr_i_mesh.NumberOfElts)]
            elif np.max(np.hstack((fr_i.Ffront[::, 1], fr_i.Ffront[::, 3]))) != head_split['lhead'][iter]:
                head_cells = np.where(fr_i_mesh.CenterCoor[:, 1] >= np.max(y) - head_split['lhead'][iter])[0]
                cells = [np.arange(fr_i_mesh.NumberOfElts), head_cells,
                         np.setdiff1d(np.arange(fr_i_mesh.NumberOfElts), head_cells)]
            else:
                cells = [np.arange(fr_i_mesh.NumberOfElts)]

            # - Calculate the various powers - #
            Viscous_P[iter] = get_Viscous_P(fr_i, Fluid, Solid, SimProp, fr_i_mesh, x, y, cells)
            # dissipation by shear of Peruzzo et al.

            Fracture_P[iter] = get_Fracture_P(fr_i, Solid, l, x_m, y_m, x, y, cells)
            # dissipation by fracture of Peruzzo et al.

            Elastic_P[iter] = get_Elastic_P(fr_im1, fr_i, fr_i_mesh, fr_i.pNet, fr_im1.pNet, cells)
            # effective rate of elastic energy by Peruzzo et al.



            Injection_P[iter] = get_External_injection(fr_im1, fr_i, cells)
            # power by injection

            LeakOff_P[iter] = get_leakOff_P(fr_im1, fr_i, fr_i_mesh, Solid, cells)
            # power loss  by leak-off by Peruzzo et al.

            Gravity_P[iter] = get_External_gravity(fr_i, Fluid, Solid, SimProp, fr_i_mesh, x, y, cells)
            # power provided to the fluid by gravity

            # - Store the time and mark the next iteration - #
            power_time_steps[iter] = fr_i.time
            iter = iter + 1

    # * -- Return only the non-zero values -- * #
    return Viscous_P[:iter], Fracture_P[:iter], Elastic_P[:iter], Injection_P[:iter], LeakOff_P[:iter],\
           Gravity_P[:iter], power_time_steps[:iter]

#-----------------------------------------------------------------------------------------------------------------------

def get_Viscous_P(fr_i, Fluid, Solid, SimProp, fr_i_mesh, x, y, cells):
    """This function calculates the power dissipated by viscous flow in the fracture

    :param fr_i: Fracture object of the current time step - see related documentation
    :param Fluid: Instance of the class FluidProperties - see related documentation
    :param Solid: Instance of the class MaterialProperties - see related documentation
    :param SimProp: Instance of the class SimulationProperties - see related documentation
    :param fr_i_mesh: Mesh corresponding to the current fracture
    :param cells: an array containing either all elements in one entry or two entries with the elements of the tail and
                  head of buoyant fractures respectively
    :return: list with one (all elements) or two (separation between head and tail) float value(s) of the energy
             dissipated by viscous flow for this time-step
    """

    from level_set.continuous_front_reconstruction import findangle

    # * -- Export some required values -- * #
    viscosity = Fluid.viscosity             # the viscosity of the fluid
    cell_area = fr_i_mesh.hx * fr_i_mesh.hy # the surface of one cell (regular grid)
    w = fr_i.w[fr_i.EltCrack]               # the opening of all cells in the crack

    # * -- Get the fluid velocity everywhere in the fracture -- * #
    fluid_vel_list, waste = get_velocity_as_vector(Solid, Fluid, [fr_i], SimProp) # the fluid flow velocity of the cells
    fluid_vel = fluid_vel_list[0]

    output = [0.] * len(cells)
    # * -- We extract the averaged square of the velocity and calculate the local component of the dissipation -- * #
    for split in range(len(cells)):
        # - Initiate the required information - #
        common_cells = np.intersect1d(fr_i.EltCrack, cells[split])
        nEltCrack = common_cells.size
        sqVx = np.zeros(nEltCrack)
        sqVy = np.zeros(nEltCrack)
        sqV = np.zeros(nEltCrack)
        Viscous_P_vec = np.zeros(nEltCrack)
        # - Loop over all the elements - #
        for i in range(nEltCrack):
            ID = common_cells[i]
            crack_ind = np.where(fr_i.EltCrack == ID)[0][0]
            if w[crack_ind] == 0.:
                Viscous_P_vec[i] = 0.
            elif not ID in fr_i.EltTip:
                for nei, neiID in enumerate(fr_i_mesh.NeiElements[ID]):
                    if neiID not in fr_i.EltTip:
                        sqVx[i] += fluid_vel[2 * nei, crack_ind]**2
                        sqVy[i] += fluid_vel[2 * nei + 1, crack_ind]**2
                    else:
                        tip_ind = np.where(fr_i.EltTip == neiID)[0][0]  # tip index
                        coord_zero_vertex = fr_i_mesh.VertexCoor[fr_i_mesh.Connectivity[neiID][fr_i.ZeroVertex[tip_ind]]]
                        [alpha, xint, yint] = findangle(x[tip_ind][0], y[tip_ind][0], x[tip_ind][1], y[tip_ind][1],
                                                        coord_zero_vertex[0], coord_zero_vertex[1],
                                                        100 * np.sqrt(np.finfo(float).eps))
                        normal_x = xint - coord_zero_vertex[0]
                        normal_y = yint - coord_zero_vertex[1]
                        normal_x = normal_x / np.sqrt(normal_x ** 2 + normal_y ** 2)
                        normal_y = normal_y / np.sqrt(normal_x ** 2 + normal_y ** 2)
                        if normal_y == 0:
                            angle = np.pi / 2
                        else:
                            angle = np.tan(normal_x / normal_y)
                        sqVx[i] += (np.cos(angle) * fr_i.v[tip_ind]) ** 2
                        sqVy[i] += (np.sin(angle) * fr_i.v[tip_ind]) ** 2

                sqV[i] = sqVx[i]/4 + sqVy[i]/4
                Viscous_P_vec[i] = sqV[i]/w[crack_ind]
            else:
                tip_ind = np.where(fr_i.EltTip == ID)[0][0]
                Viscous_P_vec[i] = fr_i.FillF[tip_ind]*fr_i.v[tip_ind]**2 / w[crack_ind]

        # * -- Calculate the final viscous dissipation -- * #
        output[split] = cell_area * 12 * viscosity * np.sum(Viscous_P_vec)

        # from utilities.utility import plot_as_matrix
        # K = np.ones((fr_i_mesh.NumberOfElts,), ) * 1e2
        # K[fr_i.EltCrack] = Viscous_P_vec * cell_area * 12 * viscosity
        # # K[fr_i.EltTip] = 50
        # plot_as_matrix(K, fr_i_mesh)

        # tipsum = 0.
        # restsum = 0.
        # for i in range(nEltCrack):
        #     if common_cells[i] in fr_i.EltTip:
        #         tipsum += cell_area * 12 * viscosity * Viscous_P_vec[i]
        #     else:
        #         restsum += cell_area * 12 * viscosity * Viscous_P_vec[i]
        # print(tipsum)
        # print(restsum)

    return output

#-----------------------------------------------------------------------------------------------------------------------

def get_l_Ffront_ordered_as_v(Ffront, EltTip, mesh):
    """This function calculates the circumference of the fracture. This is necessary to estimate the fracture energy.

    :param Ffront: numpy array of the intersection points of the fracture front with the grid.
    :param EltTip: numpy array of the indices of the elements where the front passes through.
    :param mesh: mesh object of the evaluated fracture footprint - see related documentation
    :return: three numpy arrays containing the length of the front segment, the x-coordinate of its center, and the
                y-coordinate of its center
    """
    # * -- Initiate the required values -- * #
    l_collection = np.zeros(len(Ffront))
    center_collection_x = np.zeros(len(Ffront))         # x coordinate of mid point of every front segment
    center_collection_y = np.zeros(len(Ffront))         # y coordinate of mid point of every front segment
    collection_x = init_list_of_objects(len(Ffront))    # x coordinates of the intersection points of every segment
    collection_y = init_list_of_objects(len(Ffront))    # y coordinates of the intersection points of every segment
    frontIDlist = np.zeros(len(Ffront), dtype=int)

    # * -- Loop over the front segments -- * #
    for point_ID, point in enumerate(Ffront):
        # - Extracting the coordinates of the points - #
        [x1, y1] = [point[0], point[1]]
        [x2, y2] = [point[2], point[3]]
        collection_x[point_ID] = [x1, x2]
        collection_y[point_ID] = [y1, y2]

        # - Calculate the length of the segment - #
        l_collection[point_ID] = np.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))

        # - Calculate the length of the segment - #
        center_collection_x[point_ID] = 0.5 * (x1 + x2)
        center_collection_y[point_ID] = 0.5 * (y1 + y2)

        # - Get the element index - #
        frontIDlist[point_ID] = mesh.locate_element(center_collection_x[point_ID], center_collection_y[point_ID])[0]
        if frontIDlist[point_ID] not in EltTip:
            if center_collection_x[point_ID] >= mesh.domainLimits[3] + mesh.hx / 2. or \
                    center_collection_y[point_ID] >= mesh.domainLimits[1] + mesh.hy / 2. \
                    or center_collection_x[point_ID] <= mesh.domainLimits[2] - mesh.hx / 2. or \
                    center_collection_y[point_ID] <= mesh.domainLimits[0] - mesh.hy / 2.:
                print("ERROR Point outside domain")
                frontIDlist[point_ID] = np.nan

            precision = 0.1 * np.sqrt(np.finfo(float).eps)

            cellIDs = np.intersect1d(np.where(abs(mesh.CenterCoor[:, 0] - center_collection_x[point_ID]) <
                                              mesh.hx / 2. + precision),
                                     np.where(abs(mesh.CenterCoor[:, 1] - center_collection_y[point_ID]) <
                                              mesh.hy / 2. + precision)).flatten()

            common = np.intersect1d(cellIDs, EltTip)
            if len(common) == 1:
                frontIDlist[point_ID] = common[0]
            elif len(common) >= 1:
                deltaXi = mesh.CenterCoor[common, 0] - center_collection_x[point_ID]
                deltaXi = deltaXi * deltaXi
                deltaYi = mesh.CenterCoor[common, 1] - center_collection_y[point_ID]
                deltaYi = deltaYi * deltaYi
                dist = deltaXi + deltaYi
                closest = np.where(dist == dist.min())[0]
                if len(closest) > 1:
                    print(
                        "Can't find the closest among " + str(len(closest)) + " cells --> returning the first of them")
                    frontIDlist[point_ID] = np.asarray([common[0]]).flatten()[0]
                else:
                    frontIDlist[point_ID] = np.asarray([common[closest]]).flatten()[0]
            else:
                print("ERROR Not able to identify the front element, searching from neighbours, to be implemented")
                SystemExit()
                # HERE I NEED TO SOLVE IT!

    # * -- We check if we have duplicate front segments for one tip cell -- * #
    frontIDlist_new = np.unique(frontIDlist)

    # * -- A non unique front cell has been found for one tip segment -- * #
    if len(frontIDlist) != len(frontIDlist_new):
        print("ERROR locating the front cell from the front edge coordinates")
        SystemExit()

    # * -- We found two tip elements for one tip segment -- * #
    if len(EltTip) != len(frontIDlist_new):
        print("ERROR the number of tip elements is not the same as the one derived by the front segments")
        SystemExit()

    # * -- The tip elements of the fracture do not correspond to the ones we've derived -- * #
    if len(np.setdiff1d(EltTip, frontIDlist_new)) != 0:
        print("ERROR Fr.EltTip does not contain the same elements as in 'frontIDlist_new' ")
        SystemExit()

    # * -- If the tip elements are not naturally sorted correctly, we sort them here -- * #
    if np.sum(EltTip - frontIDlist_new) != 0:
    # - strategy - #
    #   v1 = [1 3 5 2]      -> [1 2 3 5]  and via [0 3 1 2]
    #   v2 = [5 2 3 1]      -> [2 3 5 1]  and via [3 1 2 0]
    #      so element in pos 0 correspond to elem in pos 3
    #      so element in pos 3 correspond to elem in pos 1
    #      so element in pos 1 correspond to elem in pos 2
    #      so element in pos 2 correspond to elem in pos 0

        # - Initialize the required vectors - #
        indSort1 = np.argsort(EltTip)
        indSort2 = np.argsort(frontIDlist_new)
        sorted_l = np.zeros(len(EltTip))
        sorted_center_collection_x = np.zeros(len(EltTip))
        sorted_center_collection_y = np.zeros(len(EltTip))
        sorted_collection_x = init_list_of_objects(len(EltTip))
        sorted_collection_y = init_list_of_objects(len(EltTip))
        check_ = np.zeros(len(EltTip))

        # - Sort frontIDlist_new, length, center coordinates, and intersection coordinates as Fr.EltTip - #
        for ii in range(len(indSort1)):
            sorted_l[indSort1[ii]] = l_collection[indSort2[ii]]
            sorted_center_collection_x[indSort1[ii]] = center_collection_x[indSort2[ii]]
            sorted_center_collection_y[indSort1[ii]] = center_collection_y[indSort2[ii]]
            sorted_collection_x[indSort1[ii]] = collection_x[indSort2[ii]]
            sorted_collection_y[indSort1[ii]] = collection_y[indSort2[ii]]
            check_[indSort1[ii]] = frontIDlist_new[indSort2[ii]]

        # - The elements are not correctly sorted - #
        if np.sum(check_ - EltTip) != 0:
            print("ERROR sorting the arrays has a bug ")
            SystemExit()
        else:
            # - Re-assigne the sorted collections - #
            l_collection = sorted_l
            center_collection_x = sorted_center_collection_x
            center_collection_y = sorted_center_collection_y
            collection_x = sorted_collection_x
            collection_y = sorted_collection_y

    # - Export the sorted results - #
    return l_collection, center_collection_x, center_collection_y, collection_x, collection_y

#-----------------------------------------------------------------------------------------------------------------------

def get_Fracture_P(fr_i, Solid, l, x_m, y_m, x, y, cells):
    """This function calculates the power dissipated to create new fractures for a current time-step.

    :param fr_i: fracture object of the current time step - see related documentation
    :param Solid: A material properties object - see related documentation
    :param l: a list containing the length of the different front segments
    :param x_m: a list containing x coordinate of the center of the front segment
    :param y_m: a list containing y coordinate of the center of the front segment
    :param x: a list containing x coordinates of the intersections of the tip segments with the grid
    :param y: a list containing y coordinates of the intersections of the tip segments with the grid
    :return: the value of the dissipated fracturing power
    """
    # Note: 1) Not yet for TI material.
    #       2) could be improved by having "get_l_Ffront_ordered_as_v" faster

    from level_set.continuous_front_reconstruction import findangle

    # * -- Initialize the solution and administrative parameters -- * #
    Toughness_function = True

    # * -- Check if a toughness function exists -- * #
    try:
        Solid.K1cFunc[0., 0., 0.]
    except:
        Toughness_function = False

    output = [0.] * len(cells)
    # * -- We extract the averaged square of the velocity and calculate the local component of the dissipation -- * #
    for split in range(len(cells)):
        common_cells = np.intersect1d(fr_i.EltTip, cells[split])
        NofFfrontSegments = common_cells.size
        Fracture_P_vec = np.zeros(NofFfrontSegments)
        if Toughness_function:
            # * -- Evaluate the power using the function for fracturing toughness -- * #
            for i in range(NofFfrontSegments):
                # - The following is to calculate the normal - #
                ID = common_cells[i]                           # global index
                tip_ind = np.where(fr_i.EltTip == ID)[0][0]     # tip index
                coord_zero_vertex = fr_i.Mesh.VertexCoor[fr_i.Mesh.Connectivity[ID][fr_i.ZeroVertex[tip_ind]]]
                [alpha, xint, yint] = findangle(x[tip_ind][0], y[tip_ind][0], x[tip_ind][1], y[tip_ind][1],
                                                coord_zero_vertex[0], coord_zero_vertex[1],
                                                100*np.sqrt(np.finfo(float).eps))
                normal_x = xint - coord_zero_vertex[0]
                normal_y = yint - coord_zero_vertex[1]
                normal_x = normal_x / np.sqrt(normal_x ** 2 + normal_y ** 2)
                normal_y = normal_y / np.sqrt(normal_x ** 2 + normal_y ** 2)
                angle = np.tan(normal_x / normal_y)
                # - Evaluate the power as l * v * KIc^2/E' - #
                # Note: The function assumes a homogeneous and constant E'
                Fracture_P_vec[i] = l[tip_ind] * fr_i.v[tip_ind] * \
                                    Solid.K1cFunc[x_m[tip_ind], y_m[tip_ind], angle]**2 / Solid.Eprime
        else:
            # * -- Evaluate the power using a homogeneous fracturing toughness or arrest the simulation -- * #
            homogeneous = False
            if Solid.K1c.max() == Solid.K1c.min():
                homogeneous = True
            if homogeneous:
                KIc = Solid.K1c[0]
                for i in range(NofFfrontSegments):
                    ID = common_cells[i]  # global index
                    tip_ind = np.where(fr_i.EltTip == ID)[0][0]  # tip index
                    Fracture_P_vec[i] = l[tip_ind] * fr_i.v[tip_ind] * KIc**2/Solid.Eprime
            else:
                print("Not implemented for heterogenous toughness defined without a function")
        output[split] = np.sum(Fracture_P_vec)

    # * -- Export the total dissipated power -- * #
    return output

#-----------------------------------------------------------------------------------------------------------------------

def get_Elastic_P(fr_im1, fr_i, fr_i_mesh, traction_i, traction_im1, cells):
    """This function calculates the Elastic power stored in the system.

    :param fr_im1: fracture object of the previous time step - see related documentation
    :param fr_i: fracture object of the current time step - see related documentation
    :param fr_i_mesh: mesh object corresponding to the fracture object of the current time step -
                        see related documentation
    :param traction_i: the tractions (pressure) on the fracture surface at the current time step
    :param traction_im1: the tractions (pressure)  on the fracture surface at the previous time step
    :return: the value of the elastically stored power
    """

    # * -- Extract some base parameters -- * #
    dt = np.abs(fr_i.time - fr_im1.time)        # the time step
    cell_area = fr_i_mesh.hx * fr_i_mesh.hy     # the cell area (constant cell grid)
    w = fr_i.w[fr_i.EltCrack]                   # the opening of the cells in the crack

    output = [0.] * len(cells)
    # * -- We extract the averaged square of the velocity and calculate the local component of the dissipation -- * #
    for split in range(len(cells)):
        common_cells = np.intersect1d(fr_i.EltCrack, cells[split])
        nEltCrack = common_cells.size
        Elastic_P_vec = np.zeros(nEltCrack)
        if dt != 0:
            # * -- Loop over all the elements in the crack -- * #
            for i in range(nEltCrack):
                ID = common_cells[i]   # The index of the cell in the global context
                crack_ind = np.where(fr_i.EltCrack == ID)[0][0]
                w_old = fr_im1.w[ID]    # The opening of the cell in the previous time
                # - Switch in function of the cell beeing a channel or a tip element - #
                if not ID in fr_i.EltTip:
                    Elastic_P_vec[i] = 0.5 * (w[crack_ind]*traction_i[ID]-traction_im1[ID]*w_old)/dt
                else:
                    # - If a tip element, identify the corresponding filling fraction and apply it to the result - #
                    tip_ind = np.where(fr_i.EltTip == ID)[0]
                    Elastic_P_vec[i] = fr_i.FillF[tip_ind] * 0.5 * (w[crack_ind]*traction_i[ID] -
                                                                    traction_im1[ID]*w_old)/dt

            # * -- Export the total dissipated power -- * #
            output[split] = cell_area * np.sum(Elastic_P_vec)
        else:
            output[split] = 0.
    return output

#-----------------------------------------------------------------------------------------------------------------------

def get_External_P(fr_im1, fr_i, Fluid, Solid, SimProp, fr_i_mesh, x, y, cells):
    """This function calculates the external power added to the system.

    :param fr_im1: fracture object of the previous time step - see related documentation
    :param fr_i: fracture object of the current time step - see related documentation
    :param Fluid: Instance of the class FluidProperties - see related documentation
    :param Solid: Instance of the class MaterialProperties - see related documentation
    :param SimProp: Instance of the class SimulationProperties - see related documentation
    :param fr_i_mesh: mesh object corresponding to the fracture object of the current time step -
                      see related documentation
    :param x: a list containing x coordinates of the intersections of the tip segments with the grid
    :param y: a list containing y coordinates of the intersections of the tip segments with the grid
    :return: the value of the external power added to the system
    """

    # * -- The external energy is composed of a gravity component and the injection -- * #
    # - Switch in function of activated gravity - #
    if SimProp.gravity:
        ext_inj = get_External_injection(fr_im1, fr_i, cells)
        ext_gravi = get_External_gravity(fr_i, Fluid, Solid, SimProp, fr_i_mesh, x, y, cells)
        output = [ext_inj[0] + ext_gravi[0]]
        for i in range(len(cells) - 1):
            output = [ext_inj[i] + ext_gravi[i]]
        return output

    else:
        return get_External_injection(fr_im1, fr_i, cells)

#-----------------------------------------------------------------------------------------------------------------------

def get_External_injection(fr_im1, fr_i, cells):
    """This function calculates the external power added to the system due to the fluid injection.

    :param fr_im1: fracture object of the previous time step - see related documentation
    :param fr_i: fracture object of the current time step - see related documentation
    :return: the value of the injection related external power
    """
    # Note: 1) this is coded up for only one injection point

    # * -- The external power added by injection is given by the net pressure times the injection rate -- * #
    #ToDo: to be generalized
    output = [0.] * len(cells)
    # * -- We extract the averaged square of the velocity and calculate the local component of the dissipation -- * #
    for split in range(len(cells)):
        if len(fr_i.source) != 0 and (fr_i.time - fr_im1.time) != 0. and len(np.intersect1d(fr_i.source[0],
                                                                                            cells[split])):
            output[split] = fr_i.pNet[fr_i.source[0]] * (fr_i.injectedVol - fr_im1.injectedVol) / \
                            (fr_i.time - fr_im1.time)
        else:
            output[split] = 0.

    return output

#-----------------------------------------------------------------------------------------------------------------------

def get_External_gravity(fr_i, Fluid, Solid, SimProp, fr_i_mesh, x, y, cells):
    """This function calculates the external power added to the system due to gravitational fluid flow.

    :param fr_i: fracture object of the current time step - see related documentation
    :param Fluid: Instance of the class FluidProperties - see related documentation
    :param Solid: Instance of the class MaterialProperties - see related documentation
    :param SimProp: Instance of the class SimulationProperties - see related documentation
    :param fr_i_mesh: mesh object corresponding to the fracture object of the current time step -
                      see related documentation
    :param x: a list containing x coordinates of the intersections of the tip segments with the grid
    :param y: a list containing y coordinates of the intersections of the tip segments with the grid
    :return: the value of the external power created by gravitational flow
    """

    # * -- Import a function allowing to find the normal of the fracture propagation -- * #
    from level_set.continuous_front_reconstruction import findangle

    # * -- Extract some basical information -- * #
    cell_area = fr_i_mesh.hx * fr_i_mesh.hy     # the surface of one cell (regular grid)
    w = fr_i.w[fr_i.EltCrack]                   # the opening of all cells in the crack
    gravity = Solid.gravityValue                # value of gravitational acceleration
    rho_f = Fluid.density                       # fluid density
    rho_s = Solid.density

    # * -- Get the fluid velocity of the cells, including the gravity term -- * #
    fluid_vel_list, waste = get_velocity_as_vector(Solid, Fluid, [fr_i], SimProp)
    fluid_vel = fluid_vel_list[0]

    output = [0.] * len(cells)
    # * -- We extract the averaged square of the velocity and calculate the local component of the dissipation -- * #
    for split in range(len(cells)):
        common_cells = np.intersect1d(fr_i.EltCrack, cells[split])
        nEltCrack = common_cells.size
        # * -- initialize the solution vectors -- * #
        Vy = np.zeros(nEltCrack)
        Vx = np.zeros(nEltCrack)
        External_p_gravity_vec = np.zeros(nEltCrack)

        # * -- Define the velocity at the center as the average of the velocity on the edges -- * #
        for i in range(nEltCrack):
            ID = common_cells[i]
            crack_ind = np.where(fr_i.EltCrack == ID)[0][0]
            # - Switch if a tip element is encountered - #
            if not ID in fr_i.EltTip:
                for nei, neiID in enumerate(fr_i_mesh.NeiElements[ID]):
                    if neiID not in fr_i.EltTip:
                        Vy[i] += fluid_vel[2 * nei + 1, crack_ind]
                        Vx[i] += fluid_vel[2 * nei, crack_ind]
                    else:
                        tip_ind = np.where(fr_i.EltTip == neiID)[0][0]  # tip index
                        coord_zero_vertex = fr_i_mesh.VertexCoor[fr_i_mesh.Connectivity[neiID][fr_i.ZeroVertex[tip_ind]]]
                        [alpha, xint, yint] = findangle(x[tip_ind][0], y[tip_ind][0], x[tip_ind][1], y[tip_ind][1],
                                                        coord_zero_vertex[0], coord_zero_vertex[1],
                                                        100 * np.sqrt(np.finfo(float).eps))
                        normal_x = xint - coord_zero_vertex[0]
                        normal_y = yint - coord_zero_vertex[1]
                        normal_x = normal_x / np.sqrt(normal_x ** 2 + normal_y ** 2)
                        normal_y = normal_y / np.sqrt(normal_x ** 2 + normal_y ** 2)
                        if normal_y == 0:
                            angle = np.pi / 2
                        else:
                            angle = np.tan(normal_x / normal_y)
                        Vx[i] += np.cos(angle) * fr_i.v[tip_ind]
                        Vy[i] += np.sin(angle) * fr_i.v[tip_ind]

                # - We average only the velocities - #
                Vy[i] = Vy[i]/4
                Vx[i] = Vx[i]/4

                # - The external power is velocity * opening * flui density * gravitational acceleration - #
                External_p_gravity_vec[i] = Vy[i] * w[crack_ind] * (rho_s[ID] - rho_f) * gravity[2 * ID + 1] + \
                                            Vx[i] * w[crack_ind] * (rho_s[ID] - rho_f) * gravity[2 * ID]
            else:
                # - For tip elements the normal of the propagation direction is. - #
                tip_ind = np.where(fr_i.EltTip==ID)[0][0]
                # - The following is to calculate the normal - #
                coord_zero_vertex = fr_i_mesh.VertexCoor[fr_i_mesh.Connectivity[ID][fr_i.ZeroVertex[tip_ind]]]
                [alpha, xint, yint] = findangle(x[tip_ind][0], y[tip_ind][0], x[tip_ind][1], y[tip_ind][1],
                                                coord_zero_vertex[0], coord_zero_vertex[1],
                                                100*np.sqrt(np.finfo(float).eps))
                normal_x = xint - coord_zero_vertex[0]
                normal_y = yint - coord_zero_vertex[1]
                if normal_x ** 2 + normal_y ** 2 != 0:
                    normal_y = normal_y / np.sqrt(normal_x ** 2 + normal_y ** 2)
                    normal_x = normal_x / np.sqrt(normal_x ** 2 + normal_y ** 2)
                else:
                    normal_y = 0.
                    normal_x = 0.
                # - We multiply the fracture velocity (= fluid velocity) by the normal in y as we assume g in -y - #
                External_p_gravity_vec[i] = fr_i.FillF[tip_ind] * w[crack_ind] * (rho_s[ID] - rho_f)\
                                            * np.abs(fr_i.v[tip_ind]) * (normal_y * gravity[2 * ID + 1]
                                                                         + normal_x * gravity[2 * ID])

        output[split] = cell_area * np.sum(External_p_gravity_vec)

        # from utilities.utility import plot_as_matrix
        # K = np.ones((fr_i_mesh.NumberOfElts,), ) * np.nan
        # K[common_cells] = Vy
        # # K[fr_i.EltTip] = 50
        # plot_as_matrix(K, fr_i_mesh)

    # * -- Export the total dissipated power by multiplying with the uniform cell_area -- * #
    return output

#-----------------------------------------------------------------------------------------------------------------------

def get_leakOff_P(fr_im1, fr_i, fr_i_mesh, Solid, cells):
    """This function calculates the dissipated power due to fluid leak-off to the medium.

    :param fr_im1: fracture object of the previous time step - see related documentation
    :param fr_i: fracture object of the current time step - see related documentation
    :param fr_i_mesh: mesh object corresponding to the fracture object of the current time step -
                      see related documentation
    :param Solid: Instance of the class MaterialProperties - see related documentation
    :return: the value of the external power dissipated by fluid leak-off
    """

    # * -- Check if we do not have a zero leak-off case -- * #
    dt = np.abs(fr_i.time - fr_im1.time) # the time step
    output = [0.] * len(cells)
    # * -- We extract the averaged square of the velocity and calculate the local component of the dissipation -- * #
    for split in range(len(cells)):
        if sum(Solid.Cprime) != 0 and dt != 0.:
            # * -- Extract some base parameters -- * #
            cell_area = fr_i_mesh.hx * fr_i_mesh.hy     # the cell area (constant cell grid)
            common_cells = np.intersect1d(fr_i.EltCrack, cells[split])
            nEltCrack = common_cells.size

            # * -- Initialize the solution vector -- * #
            leakOff_P_vec = np.zeros(nEltCrack)

            # * -- Loop over all the elements in the crack -- * #
            for i in range(nEltCrack):
                ID = common_cells[i]   # The index of the cell in the global context

                # - calculate the leak-off rate - #
                # fr_i.LkOff stores the leaked volume during the last time step of this cell.
                v_lkOff = fr_i.LkOff[ID] / (dt * cell_area)

                # - We average the tractions between two time-steps - #
                # fr_i.LkOff stores the leaked volume during the last time step of this cell.
                pf_avg = (fr_i.pFluid[ID] + fr_im1.pFluid[ID]) / 2

                # - Switch in function of the cell being a channel or a tip element - #
                if not ID in fr_i.EltTip:
                    leakOff_P_vec[i] = v_lkOff * pf_avg
                else:
                    # - If a tip element, identify the corresponding filling fraction and apply it to the result - #
                    tip_ind = np.where(fr_i.EltTip == ID)[0]
                    leakOff_P_vec[i] = fr_i.FillF[tip_ind] * v_lkOff * pf_avg

            # * -- Export the total dissipated power -- * #
            output[split] = 2 * cell_area * np.sum(leakOff_P_vec)
        else:
            # - If no leak-off is there the dissipated power is simply zero - #
            output[split] = 0.

    return output

#-----------------------------------------------------------------------------------------------------------------------

def get_closure_geometry(Fr_list, centre, layer_indices = None):
    """This function calculates the geometry of a fracture during closure.

    :param Fr_list: list of fracture objects to be analyzed - see related documentation
    :param centre: the location around which closure is to be analyzed, must be a single point (list with two entries)
    :param layer_indices: the indices of the elements within the zone to be analyzed (note that these need to be from
                          the mesh encountered during the closure of the fracture)
    :return: the value of the breadth, height and average radius during closure.
    """
    log = logging.getLogger('PyFrac.get_closure_geometry')

    # We need to ensure that we have indices to analyze. Set them to all indices if none are provided.
    if layer_indices == None:
        layer_indices = np.arange(Fr_list[-1].mesh.NumberOfElts)

    closing_breadth = np.zeros(len(Fr_list))        # Closed fracture radius in the horizontal direction
    closing_height = np.zeros(len(Fr_list))         # Closed fracture radius in the vertical direction
    closing_radius_min = np.zeros(len(Fr_list))     # Closed fracture minimum distance
    closing_radius_average = np.zeros(len(Fr_list)) # Closed fracture average distance

    height_rec = False      # boolean to say if the height has completely receded
    breadth_rec = False     # boolean to say if the breadth has completely receded
    fully_closed = False    # boolean to say if the fracture has fully closed

    # We pre-assign a value to the closure
    t_c = Fr_list[-1].time
    tr_breadth = Fr_list[-1].time
    tr_height = Fr_list[-1].time

    for i in range(len(Fr_list)):
        if len(np.intersect1d(Fr_list[i].closed, layer_indices)) != 0:
            if isinstance(Fr_list[i].mesh, int):
                mesh = Fr_list[Fr_list[i].mesh].mesh
            else:
                mesh = Fr_list[i].mesh

            height_elements = layer_indices[np.where(mesh.CenterCoor[layer_indices, 1] == centre[1])]
            breadth_elements = layer_indices[np.where(mesh.CenterCoor[layer_indices, 0] == centre[0])]

            closed_height_elements = np.intersect1d(Fr_list[i].closed, height_elements)
            if len(closed_height_elements) != 0:
                if not height_rec:
                    tr_height = Fr_list[i].time
                    height_rec = True
                c_c = mesh.CenterCoor[closed_height_elements]  # cell_center
                r_c = np.asarray([np.linalg.norm(c_c[e] - centre) for e in range(len(c_c))])
                closing_height[i] = r_c.min()

            closed_breadth_elements = np.intersect1d(Fr_list[i].closed, breadth_elements)
            if len(closed_breadth_elements) != 0:
                if not breadth_rec:
                    tr_breadth = Fr_list[i].time
                    breadth_rec = True
                c_c = mesh.CenterCoor[closed_breadth_elements]  # cell_center
                r_c = np.asarray([np.linalg.norm(c_c[e] - centre) for e in range(len(c_c))])
                closing_breadth[i] = r_c.min()
            else:
                c_c = mesh.CenterCoor[breadth_elements]  # cell_center
                r_c = np.asarray([np.linalg.norm(c_c[e] - centre) for e in range(len(c_c))])
                closing_breadth[i] = r_c.max()

            if closing_breadth[i] == 0 and closing_height[i] == 0 and not fully_closed:
                t_c = Fr_list[i-1].time
                fully_closed = True

            c_c = mesh.CenterCoor[np.intersect1d(Fr_list[i].closed, layer_indices)]  # cell_center
            r_c = np.asarray([np.linalg.norm(c_c[e] - centre) for e in range(len(c_c))])
            closing_radius_min[i] = r_c.min()
            closing_radius_average[i] = np.mean(np.extract(r_c <= closing_radius_min[i] + mesh.cellDiag / 3., r_c))


    # Here we issue a warning if the fracture is not fully closed
    if not fully_closed:
        log.warning("The fracture is not fully closed! Closure times might be wrong!")

    closure_info = {'tc': t_c,
            'tr_breadth': tr_breadth,
            'tr_height': tr_height,
            'closing_breadth': closing_breadth,
            'closing_height': closing_height,
            'closing_radius_min': closing_radius_min,
            'closing_radius_avg': closing_radius_average}

    return closure_info