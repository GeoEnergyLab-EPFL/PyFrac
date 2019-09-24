# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on 12.06.17.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2019.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# local

import numpy as np
from scipy.interpolate import griddata
import dill
import os
import re
import sys

from utility import ReadFracture
from HF_analytical import HF_analytical_sol, get_fracture_dimensions_analytical
from labels import *
# import FractureInitialization


if 'win32' in sys.platform or 'win64' in sys.platform:
    slash = '\\'
else:
    slash = '/'

#-----------------------------------------------------------------------------------------------------------------------


def load_fractures(address=None, sim_name='simulation', time_period=0.0, time_srs=None, step_size=1):
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

    Returns:
        fracture_list(list):            -- a list of fractures.

    """

    print('Returning fractures...')

    if address is None:
        address = '.' + slash + '_simulation_data_PyFrac'

    if address[-1] is not slash:
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
    sim_full_path = address +  sim_full_name
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
    while fileNo < 5000:

        # trying to load next file. exit loop if not found
        try:
            ff = ReadFracture(sim_full_path + slash + sim_full_name + '_file_' + repr(fileNo))
        except FileNotFoundError:
            break

        fileNo += step_size

        if 1. - next_t / ff.time >= -1e-8:
            # if the current fracture time has advanced the output time period
            print('Returning fracture at ' + repr(ff.time) + ' s')

            fracture_list.append(ff)

            if t_srs_given:
                if t_srs_indx < len(time_srs) - 1:
                    t_srs_indx += 1
                    next_t = time_srs[t_srs_indx]
                if ff.time > max(time_srs):
                    break
            else:
                next_t = ff.time + time_period

    if fileNo >= 5000:
        raise SystemExit('too many files.')

    if len(fracture_list) == 0:
        raise ValueError("Fracture list is empty")

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

    variable_list = []
    time_srs = []

    if variable is 'time' or variable is 't':
        for i in fracture_list:
            variable_list.append(i.time)
            time_srs.append(i.time)

    elif variable is 'width' or variable is 'w' or variable is 'surface':
        for i in fracture_list:
            variable_list.append(i.w)
            time_srs.append(i.time)

    elif variable is 'fluid pressure' or variable is 'pf':
        for i in fracture_list:
            variable_list.append(i.pFluid)
            time_srs.append(i.time)

    elif variable is 'Net pressure' or variable is 'pn':
        for i in fracture_list:
            variable_list.append(i.pNet)
            time_srs.append(i.time)

    elif variable is 'front velocity' or variable is 'v':
        for i in fracture_list:
            vel = np.full((i.mesh.NumberOfElts, ), np.nan)
            vel[i.EltTip] = i.v
            variable_list.append(vel)
            time_srs.append(i.time)

    elif variable is 'Reynolds number' or variable is 'Re':
        if fracture_list[-1].ReynoldsNumber is None:
            raise SystemExit(err_var_not_saved)
        for i in fracture_list:
            if edge < 0 or edge > 4:
                raise ValueError('Edge can be an integer between and including 0 and 4.')
            if edge < 4:
                variable_list.append(i.ReynoldsNumber[edge])
                time_srs.append(i.time)
            elif i.ReynoldsNumber is not None:
                variable_list.append(np.mean(i.ReynoldsNumber, axis=0))
                time_srs.append(i.time)
            else:
                variable_list.append(np.full((i.mesh.NumberOfElts, ), np.nan))

    elif variable is 'fluid flux' or variable is 'ff':
        if fracture_list[-1].fluidFlux is None:
            raise SystemExit(err_var_not_saved)
        for i in fracture_list:
            if edge < 0 or edge > 4:
                raise ValueError('Edge can be an integer between and including 0 and 4.')
            if edge < 4:
                variable_list.append(i.fluidFlux[edge])
                time_srs.append(i.time)
            elif i.fluidFlux is not None:
                variable_list.append(np.mean(i.fluidFlux, axis=0))
                time_srs.append(i.time)
            else:
                variable_list.append(np.full((i.mesh.NumberOfElts,), np.nan))

    elif variable is 'fluid velocity' or variable is 'fv':
        if fracture_list[-1].fluidVelocity is None:
            raise SystemExit(err_var_not_saved)
        for i in fracture_list:
            if edge < 0 or edge > 4:
                raise ValueError('Edge can be an integer between and including 0 and 4.')
            if edge < 4:
                variable_list.append(i.fluidVelocity[edge])
                time_srs.append(i.time)
            elif i.fluidFlux is not None:
                variable_list.append(np.mean(i.fluidVelocity, axis=0))
                time_srs.append(i.time)
            else:
                variable_list.append(np.full((i.mesh.NumberOfElts, ), np.nan))

    elif variable in ('front_dist_min', 'd_min', 'front_dist_max', 'd_max', 'front_dist_mean', 'd_mean'):
        for i in fracture_list:
            # coordinate of the zero vertex in the tip cells
            vertex_coord_tip = i.mesh.VertexCoor[i.mesh.Connectivity[i.EltTip, i.ZeroVertex]]
            if variable is 'front_dist_mean' or variable is 'd_mean':
                variable_list.append(np.mean((vertex_coord_tip[:, 0] ** 2 +
                                              vertex_coord_tip[:, 1] ** 2) ** 0.5 + i.l))
            elif variable is 'front_dist_max' or variable is 'd_max':
                variable_list.append(max((vertex_coord_tip[:, 0] ** 2 +
                                          vertex_coord_tip[:, 1] ** 2) ** 0.5 + i.l))
            elif variable is 'front_dist_min' or variable is 'd_min':
                variable_list.append(min((vertex_coord_tip[:, 0] ** 2 +
                                          vertex_coord_tip[:, 1] ** 2) ** 0.5 + i.l))
            time_srs.append(i.time)
    elif variable is 'mesh':
        for i in fracture_list:
            variable_list.append(i.mesh)
            time_srs.append(i.time)

    elif variable is 'efficiency' or variable is 'ef':
        for i in fracture_list:
            variable_list.append(i.efficiency)
            time_srs.append(i.time)
            
    elif variable is 'volume' or variable is 'V':
        for i in fracture_list:
            variable_list.append(i.FractureVolume)
            time_srs.append(i.time)
            
    elif variable is 'leak off' or variable is 'lk':
        for i in fracture_list:
            variable_list.append(i.LkOff)
            time_srs.append(i.time)
            
    elif variable is 'leaked off volume' or variable is 'lkv':
        for i in fracture_list:
            variable_list.append(sum(i.LkOffTotal[i.EltCrack]))
            time_srs.append(i.time)
            
    elif variable is 'aspect ratio' or variable is 'ar':
        for fr in fracture_list:
            x_coords = np.hstack((fr.Ffront[:, 0], fr.Ffront[:, 2]))
            x_len = np.max(x_coords) - np.min(x_coords)
            y_coords = np.hstack((fr.Ffront[:, 1], fr.Ffront[:, 3]))
            y_len = np.max(y_coords) - np.min(y_coords)
            variable_list.append(x_len / y_len)
            time_srs.append(fr.time)
            
    else:
        raise ValueError('The variable type is not correct.')

    if not return_time:
        return variable_list
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

    if variable not in supported_variables:
        raise ValueError(err_msg_variable)

    return_list = []

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
                value_point = griddata(fracture_list[i].mesh.CenterCoor,
                                       var_values[i],
                                       point,
                                       method='linear',
                                       fill_value=np.nan)
                if np.isnan(value_point):
                    print('Point outside fracture.')

                return_list.append(value_point[0])

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
    if zero_cell == np.nan:
        raise ValueError("The given point does not lie in the grid!")

    if orientation is 'vertical':
        sampling_cells = np.hstack((np.arange(zero_cell, 0, -mesh.nx)[::-1],
                                    np.arange(zero_cell, mesh.NumberOfElts, mesh.nx)))
    elif orientation is 'horizontal':
        sampling_cells = np.arange(zero_cell // mesh.nx * mesh.nx, (zero_cell // mesh.nx + 1) * mesh.nx)

    elif orientation is 'increasing':
        bottom_half = np.arange(zero_cell, 0, -mesh.nx - 1)
        bottom_half = np.delete(bottom_half, np.where(mesh.CenterCoor[bottom_half, 0] >
                                                      mesh.CenterCoor[zero_cell, 0])[0])
        top_half = np.arange(zero_cell, mesh.NumberOfElts, mesh.nx + 1)
        top_half = np.delete(top_half, np.where(mesh.CenterCoor[top_half, 0] <
                                                mesh.CenterCoor[zero_cell, 0])[0])
        sampling_cells = np.hstack((bottom_half[::-1], top_half))

    elif orientation is 'decreasing':
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

    if regime is 'E_K':
        Kc_1 = mat_prop.Kc1
    else:
        Kc_1 = None

    if regime is 'E_E':
        Cij = mat_prop.Cij
    else:
        Cij = None

    if regime is 'MDR':
        density = fluid_prop.density
    else:
        density = None

    if regime in ['M', 'MDR', 'Mt', 'PKN']:
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

                from mesh import CartesianMesh
                mesh_i = CartesianMesh(x_len, y_len, 151, 151)
            else:
                mesh_i = mesh
            mesh_list.append(mesh_i)

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
                                                        required=required_string[variable])

            if variable is 'time' or variable is 't':
                return_list.append(t)
            elif variable is 'width' or variable is 'w':
                return_list.append(w)
            elif variable is 'net pressure' or variable is 'pn':
                return_list.append(p)
            elif variable is 'front velocity' or variable is 'v':
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
            if variable is 'radius' or variable is 'r':
                return_list.append(x_len)
            elif variable is 'front_dist_min' or variable is 'd_min':
                return_list.append(y_len)
            elif variable is 'front_dist_max' or variable is 'd_max':
                return_list.append(x_len)
            elif variable is 'front_dist_mean' or variable is 'd_mean':
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

    from mesh import CartesianMesh
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


    if regime is 'E_K':
        Kc_1 = mat_prop.Kc1
    else:
        Kc_1 = None

    if regime is 'MDR':
        density = fluid_prop.density
    else:
        density = None

    if regime in ('M', 'Mt', 'PKN', 'MDR'):
        if fluid_prop is None:
            raise ValueError('Fluid properties required to evaluate analytical solution')
        muPrime = fluid_prop.muPrime
    else:
        muPrime = None

    if samp_cell is None:
        samp_cell = int(len(mat_prop.Kprime) / 2)

    x_len, y_len = get_fracture_dimensions_analytical(regime,
                                                      np.max(time_srs),
                                                      mat_prop.Eprime,
                                                      inj_prop.injectionRate[1, 0],
                                                      muPrime,
                                                      Kprime=mat_prop.Kprime[samp_cell],
                                                      Cprime=mat_prop.Cprime[samp_cell],
                                                      Kc_1=Kc_1,
                                                      h=h,
                                                      density=density,
                                                      gamma=gamma)

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

    if variable not in supported_variables:
        raise ValueError(err_msg_variable)

    return_list = []

    var_values, time_list = get_fracture_variable(fracture_list,
                                                    variable,
                                                    edge=edge,
                                                    return_time=True)

    if point is None:
        return_list = var_values
    else:
        for i in range(len(fracture_list)):
            value_point = griddata(fracture_list[i].mesh.CenterCoor,
                                   var_values[i],
                                   point,
                                   method='linear',
                                   fill_value=np.nan)
            if np.isnan(value_point):
                print('Point outside fracture.')
            return_list.append(value_point[0])

    if file_name[-4:] != '.csv':
        file_name = file_name + '.csv'

    return_list_np = np.asarray(return_list)
    np.savetxt(file_name, return_list_np, delimiter=',')


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

def get_extermities_cells(Fr_list):
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

    intercepts = []

    for fr in fr_list:
        pnt_cell = fr.mesh.locate_element(point[0], point[1])   # the cell in which the given point lie
        pnt_cell_y = fr.mesh.CenterCoor[pnt_cell, 1]            # the y coordinate of the cell
        cells_x_axis = np.where(fr.mesh.CenterCoor[:, 1] == pnt_cell_y)[0]      # all the cells with the same y coord
        tipCells_x_axis = np.intersect1d(fr.EltTip, cells_x_axis)               # the tip cells with the same y coord

        # the code bellow remove the tip cells which are directly at right and left of the cell containing the point but
        # have the front line partially passing through them. For them, the horizontal line drawn from the given point
        # will pass through the cell but not from the front line.
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
            if fr.mesh.CenterCoor[tipCells_x_axis[0], 0] < point[0]:
                lft_cell = tipCells_x_axis[0]
                rgt_cell = tipCells_x_axis[1]
            else:
                lft_cell = tipCells_x_axis[1]
                rgt_cell = tipCells_x_axis[0]
        else:
            lft_cell = np.nan
            rgt_cell = np.nan

        pnt_cell_x = fr.mesh.CenterCoor[pnt_cell, 0]
        cells_y_axis = np.where(fr.mesh.CenterCoor[:, 0] == pnt_cell_x)[0]
        tipCells_y_axis = np.intersect1d(fr.EltTip, cells_y_axis)

        # the code bellow remove the tip cells which are directly at top and bottom of the cell containing the point but
        # have the front line partially passing through them. For them, the vertical line drawn from the given point
        # will pass through the cell but not from the front line.
        if len(tipCells_y_axis) > 2:
            invalid_cell = np.full(len(tipCells_y_axis), True, dtype=bool)
            for indx, cell in enumerate(tipCells_y_axis):
                in_tip_cells = np.where(fr.EltTip == cell)[0]
                if (point[0] > fr.Ffront[in_tip_cells, 0] and point[0] <= fr.Ffront[in_tip_cells, 2]) or (
                        point[0] < fr.Ffront[in_tip_cells, 0] and point[0] >= fr.Ffront[in_tip_cells, 2]):
                    invalid_cell[indx] = False
            tipCells_y_axis = np.delete(tipCells_y_axis, np.where(invalid_cell)[0])

        if len(tipCells_y_axis) == 2:
            if fr.mesh.CenterCoor[tipCells_y_axis[0], 0] < point[1]:
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

        intrcp_top = intrcp_btm = intrcp_lft = intrcp_rgt = [np.nan] # set to nan if not available

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
