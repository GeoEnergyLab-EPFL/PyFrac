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
    'Rn': 'Reynold\'s number',
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
    'mesh': 'mesh',
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
    'Rn': 'Reynold\'s number',
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
    'Rn': '',
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
    'Rn': 1.,
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
    'mesh': None,
    'footprint': None
}
supported_variables = ('w','width','p','pressure', 'front velocity','v', 'Reynolds number', 'Rn', 'fluid flux',
             'ff', 'fluid velocity', 'fv', 'front_dist_min', 'd_min', 'front_dist_max', 'd_max',
             'front_dist_mean', 'd_mean', 'mesh', 'footprint', 't', 'time')
err_msg_variable = 'Given variable is not supported. Select one of the following:\n' \
                    '-- \'w\' or \'width\'\n' \
                    '-- \'p\' or \'pressure\'\n' \
                    '-- \'Rn\' or \'Reynolds number\'\n' \
                    '-- \'v\' or \'front velocity\'\n' \
                    '-- \'ff\' or \'fluid flux\'\n' \
                    '-- \'fv\' or \'fluid velocity\'\n' \
                    '-- \'d_min\' or \'front_dist_min\'\n' \
                    '-- \'d_max\' or \'front_dist_max\'\n' \
                    '-- \'d_mean\' or \'front_dist_mean\'\n' \
                    '-- \'mesh\'\n' \
                    '-- \'footprint\'\n'

supported_projections = ('2D', '2D_image', '2D_contours', '3D')
err_msg_projection = 'Given projection is not supported. Select one of the following:\n' \
                    '-- \'2D\'\n' \
                    '-- \'3D\'\n' \
                    '-- \'2D_image\'\n' \
                    '-- \'2D_contours\'\n'

err_var_not_saved = "The required variable is not available. Probably, saving of the variable was not\n" \
                    " enabled during the simulation. Check SimulationProperties class documentation."
def get_labels(variable, location, projection):

    label_prop = LabelProperties()
    if location in ('slice', 's'):
        label_prop.xLabel = 'coordinates ($x,y$)'
        label_prop.yLabel = labels[variable]
    elif location in ('point', 'p'):
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