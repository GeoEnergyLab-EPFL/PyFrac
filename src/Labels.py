# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Tue July 10 2018.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2017.
All rights reserved. See the LICENSE.TXT file for more details.
"""

from src.Properties import LabelProperties

Fig_labels = {
    't': 'Time',
    'time': 'Time',
    'w': 'Fracture Width',
    'width': 'Fracture Width',
    'p': 'Pressure',
    'pressure': 'Pressure',
    'front velocity': 'Front Velocity',
    'v': 'Front Velocity',
    'Reynolds number': 'Reynold\'s number',
    'Re': 'Reynold\'s number',
    'fluid flux': 'Fluid Flux',
    'ff': 'Fluid Flux',
    'fluid velocity': 'Fluid Velocity',
    'fv': 'Fluid Velocity',
    'front_dist_min': 'Closest Distance to Front',
    'd_min': 'Closest Distance to Front',
    'front_dist_max': 'Farthest Distance to Front',
    'd_max': 'Farthest Distance to Front',
    'front_dist_mean': 'Mean Distance to Front',
    'd_mean': 'Mean Distance to Front',
    'V': 'Total Volume',
    'volume': 'Total Volume',
    'lk': 'Leaked off Volume',
    'leaked off': 'Leaked off Volume',
    'lkv': 'Total Leaked of Volume',
    'leaked off volume': 'Total Leaked of Volume',
    'ar': 'Aspect Ratio',
    'aspect ratio': 'Aspect Ratio',
    'efficiency': 'Fracture Efficiency',
    'ef': 'Fracture Efficiency',
    'mesh': 'Mesh',
    'footprint': 'Fracture Footprint'
}
labels = {
    't': 'time',
    'time': 'time',
    'w': 'width',
    'width': 'width',
    'p': 'pressure',
    'pressure': 'pressure',
    'front velocity': 'front velocity',
    'v': 'Front Velocity',
    'Reynolds number': 'Reynold\'s number',
    'Re': 'Reynold\'s number',
    'fluid flux': 'fluid flux',
    'ff': 'fluid flux',
    'fluid velocity': 'fluid velocity',
    'fv': 'fluid velocity',
    'front_dist_min': '$R_{min}$',
    'd_min': '$R_{min}$',
    'front_dist_max': '$R_{max}$',
    'd_max': '$R_{max}$',
    'front_dist_mean': '$R_{mean}$',
    'd_mean': '$R_{mean}$',
    'V': 'total volume',
    'volume': 'total volume',
    'lk': 'leaked off volume',
    'leaked off': 'leaked off volume',
    'lkv': 'total leaked off volume',
    'leaked off volume': 'total leaked off volume',
    'ar': 'aspect ratio',
    'aspect ratio': 'aspect ratio',
    'efficiency': 'fracture efficiency',
    'ef': 'fracture efficiency',
    'mesh': '',
    'footprint': ''
}

units = {
    't': '($s$)',
    'time': '($s$)',
    'w': ' ($mm$)',
    'width': ' ($mm$)',
    'p': ' ($MPa$)',
    'pressure': ' ($MPa$)',
    'front velocity': ' ($m/s$)',
    'v': ' ($m/s$)',
    'Reynolds number': '',
    'Re': '',
    'fluid flux': ' ($m^3/s$)',
    'ff': ' ($m^3/s$)',
    'fluid velocity': ' ($m/s$)',
    'fv': ' ($m/s$)',
    'front_dist_min': ' ($meters$)',
    'd_min': ' ($meters$)',
    'front_dist_max': ' ($meters$)',
    'd_max': ' ($meters$)',
    'front_dist_mean': ' ($meters$)',
    'd_mean': ' ($meters$)',
    'V': ' $m^3$',
    'volume': ' $m^3$',
    'lk': ' $m^3$',
    'leaked off': ' $m^3$',
    'lkv': ' $m^3$',
    'leaked off volume': ' $m^3$',
    'ar': '',
    'aspect ratio': '',
    'efficiency': '',
    'ef': '',
    'mesh': '',
    'footprint': ''
}

unit_conversion = {
    't': 1,
    'time': 1,
    'w': 1.e-3,
    'width': 1.e-3,
    'p': 1.e6,
    'pressure': 1.e6,
    'front velocity': 1.,
    'v': 1.,
    'Reynolds number': 1.,
    'Re': 1.,
    'fluid flux': 1.,
    'ff': 1.,
    'fluid velocity': 1.,
    'fv': 1.,
    'front_dist_min': 1.,
    'd_min': 1.,
    'front_dist_max': 1.,
    'd_max': 1.,
    'front_dist_mean': 1.,
    'd_mean': 1.,
    'V': 1.,
    'volume': 1.,
    'lk': 1.,
    'leaked off': 1.,
    'lkv': 1.,
    'leaked off volume': 1.,
    'ar': 1.,
    'aspect ratio': 1.,
    'efficiency': 100.,
    'ef': 100.,
    'mesh': None,
    'footprint': None
}
supported_variables = ('w','width','p','pressure', 'front velocity','v',
                       'Reynolds number', 'Re', 'fluid flux', 'ff',
                       'fluid velocity', 'fv', 'front_dist_min', 'd_min',
                       'front_dist_max', 'd_max', 'front_dist_mean',
                       'd_mean', 'mesh', 'footprint', 't', 'time', 'volume',
                       'V', 'lk', 'leaked off', 'lkv', 'leaked off volume',
                       'ar', 'aspect ratio', 'efficiency', 'ef')

required_string = {
    't': '100000',
    'time': '100000',
    'w': '000100',
    'width': '000100',
    'p': '001000',
    'pressure': '001000',
    'front velocity': '000010',
    'v': '000010',
    'front_dist_min': '010000',
    'd_min': '010000',
    'front_dist_max': '010000',
    'd_max': '010000',
    'front_dist_mean': '010000',
    'd_mean': '010000',
    'radius': '010000',
    'r': '010000'
}

err_msg_variable = 'Given variable is not supported. Select one of the following:\n' \
                    '-- \'w\' or \'width\'\n' \
                    '-- \'p\' or \'pressure\'\n' \
                    '-- \'Re\' or \'Reynolds number\'\n' \
                    '-- \'v\' or \'front velocity\'\n' \
                    '-- \'ff\' or \'fluid flux\'\n' \
                    '-- \'fv\' or \'fluid velocity\'\n' \
                    '-- \'d_min\' or \'front_dist_min\'\n' \
                    '-- \'d_max\' or \'front_dist_max\'\n' \
                    '-- \'d_mean\' or \'front_dist_mean\'\n' \
                    '-- \'mesh\'\n' \
                    '-- \'footprint\'\n' \
                    '-- \'V\' or \'volume\'\n' \
                    '-- \'lk\' or \'leaked off\'\n' \
                    '-- \'lkv\' or \'leaked off volume\'\n' \
                    '-- \'ar\' or \'aspect ratio\'\n' \

supported_projections = ('2D', '2D_image', '2D_contours', '3D')
err_msg_projection = 'Given projection is not supported. Select one of the following:\n' \
                    '-- \'2D\'\n' \
                    '-- \'3D\'\n' \
                    '-- \'2D_image\'\n' \
                    '-- \'2D_contours\'\n'

err_var_not_saved = "The required variable is not available. Probably, saving of the variable was not\n" \
                    " enabled during the simulation. Check SimulationProperties class documentation."

def get_labels(variable, data_subset='wm', projection='2D'):
    """
    This function returns suitable labels for the given variable, location on the mesh and the projection type

    Arguments:
        variable (string)       -- the given variable for which the labels are to be provided. Possible options are
                                   given in the supported_variables list.
        data_subset (string)    -- the subset of data which is plotted. Possible options are:
                                    -- 'whole mesh' or 'wm'
                                    -- 'slice' or 's'
                                    -- 'point' or 'p'
        projection (string)     -- the string specifying the projection. Possible options are given in
                                   supported_projections list.
    :return:
    """
    label_prop = LabelProperties()
    if data_subset in ('slice', 's'):
        label_prop.xLabel = 'coordinates ($x,y$)'
        label_prop.yLabel = labels[variable]
    elif data_subset in ('point', 'p'):
        label_prop.xLabel = 'time ($s$)'
        label_prop.yLabel = labels[variable]

    if projection is '3D':
        label_prop.zLabel = labels[variable]

    label_prop.colorbarLabel = labels[variable] + units[variable]
    label_prop.units = units[variable]
    label_prop.unitConversion = unit_conversion[variable]
    label_prop.figLabel = Fig_labels[variable]
    label_prop.legend = labels[variable]

    return label_prop