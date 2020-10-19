# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo on 01.09.20.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""

testnames = dict(
    radial_M_explicit_newfront={'front_reconstruction': 'LS_continousfront',
                                'front_advancement': 'explicit',
                                'vertex': 'M',
                                'simulparam': 'radial_M'},

    radial_M_explicit_oldfront={'front_reconstruction': 'ILSA_orig',
                                'front_advancement': 'explicit',
                                'vertex': 'M',
                                'simulparam': 'radial_M'},

    radial_M_implicit_newfront={'front_reconstruction': 'LS_continousfront',
                                'front_advancement': 'implicit',
                                'vertex': 'M',
                                'simulparam': 'radial_M'},

    radial_M_implicit_oldfront={'front_reconstruction': 'ILSA_orig',
                                'front_advancement': 'implicit',
                                'vertex': 'M',
                                'simulparam': 'radial_M'},

    radial_K_explicit_newfront={'front_reconstruction': 'LS_continousfront',
                                'front_advancement': 'explicit',
                                'vertex': 'K',
                                'simulparam': 'radial_K'},

    radial_K_explicit_oldfront={'front_reconstruction': 'ILSA_orig',
                                'front_advancement': 'explicit',
                                'vertex': 'K',
                                'simulparam': 'radial_K'},

    radial_K_implicit_newfront={'front_reconstruction': 'LS_continousfront',
                                'front_advancement': 'implicit',
                                'vertex': 'K',
                                'simulparam': 'radial_K'},

    radial_K_implicit_oldfront={'front_reconstruction': 'ILSA_orig',
                                'front_advancement': 'implicit',
                                'vertex': 'K',
                                'simulparam': 'radial_K'})

simulparam = dict(
    radial_M =  {   'Lx': 0.3, 'Ly': 0.3,
                    'Nx': 41, 'Ny': 41,
                    'nu': 0.4, 'youngs_mod': 3.3e10, 'K_Ic': 0.5,
                    'Cl': 0.,  'Q0': 0.001,          'viscosity': 1.1e-3,
                    'finalTime': 1e5, 'initialR': 0.1},

    radial_K = {   'Lx': 0.3, 'Ly': 0.3,
                   'Nx': 41, 'Ny': 41,
                   'nu': 0.4, 'youngs_mod': 3.3e10, 'K_Ic': 1e6,
                   'Cl': 0.,  'Q0': 0.001,          'viscosity': 0.,
                   'finalTime': 1e5, 'initialR': 0.15})



tolerances = dict(
    radial_M_explicit_newfront= { 'radius_toll':   [0.05],
                                  'w_center_toll': [0.05],
                                  'w_section_toll_max_value': [0.0064, 0.011, 0.0164, 0.011, 0.018],
                                  'w_section_toll_cumulative_value': [0.078, 0.136, 0.18, 0.2, 0.19],
                                  'p_section_toll_max_value': [0.31, 0.072, 0.025, 0.029, 0.0091],
                                  'p_section_toll_cumulative_value': [1.34, 0.364, 0.148, 0.11, 0.042]
                                  },

    radial_M_explicit_oldfront= { 'radius_toll':   [0.05],
                                  'w_center_toll': [0.05],
                                  'w_section_toll_max_value': [0.0063, 0.0082, 0.011, 0.011, 0.0166],
                                  'w_section_toll_cumulative_value': [0.081, 0.127, 0.175, 0.196, 0.193],
                                  'p_section_toll_max_value': [0.31, 0.072, 0.026, 0.029, 0.0091],
                                  'p_section_toll_cumulative_value': [1.44, 0.32, 0.149, 0.107, 0.045]
                                  },

    radial_M_implicit_newfront= { 'radius_toll':   [0.05],
                                  'w_center_toll': [0.05],
                                  'w_section_toll_max_value': [0.011, 0.015, 0.024, 0.02, 0.024],
                                  'w_section_toll_cumulative_value': [0.12, 0.15, 0.24, 0.35, 0.28],
                                  'p_section_toll_max_value': [0.32, 0.072, 0.0373, 0.091, 0.0092],
                                  'p_section_toll_cumulative_value': [1.71, 0.36, 0.21, 0.27, 0.039]
                                  },

    radial_M_implicit_oldfront= { 'radius_toll':   [0.05],
                                  'w_center_toll': [0.05],
                                  'w_section_toll_max_value': [0.011, 0.012, 0.018, 0.011, 0.0249],
                                  'w_section_toll_cumulative_value': [0.085, 0.13, 0.185, 0.187, 0.259],
                                  'p_section_toll_max_value': [0.31, 0.072, 0.037, 0.026, 0.0092],
                                  'p_section_toll_cumulative_value': [1.21, 0.295, 0.178, 0.097, 0.039]
                                  },

    radial_K_explicit_newfront= { 'radius_toll':   [0.05],
                                  'w_center_toll': [0.075],
                                  'w_section_toll_max_value': [0.007541, 0.02500, 0.05338, 0.05069, 0.05767],
                                  'w_section_toll_cumulative_value': [0.08947, 0.2514, 0.4729, 0.7319, 0.7217],
                                  'p_section_toll_max_value': [0.4796, 0.2004, 0.1037, 0.07175, 0.05803],
                                  'p_section_toll_cumulative_value': [6.592, 4.415, 2.480, 1.602, 0.6789]
                                  },

    radial_K_explicit_oldfront= { 'radius_toll':   [0.05],
                                  'w_center_toll': [0.078],
                                  'w_section_toll_max_value': [0.007027, 0.02294, 0.05190, 0.05028, 0.05925],
                                  'w_section_toll_cumulative_value': [0.08875, 0.2517, 0.4895, 0.6835, 0.7385],
                                  'p_section_toll_max_value': [0.4796, 0.2004, 0.1037, 0.07175, 0.05803],
                                  'p_section_toll_cumulative_value': [6.597, 4.427, 2.491, 1.602, 0.6814]
                                  },

    radial_K_implicit_newfront= { 'radius_toll':   [0.074],
                                  'w_center_toll': [0.21],
                                  'w_section_toll_max_value': [0.01134, 0.01647, 0.09363, 0.1085, 0.2023],
                                  'w_section_toll_cumulative_value': [0.1246, 0.2620, 1.2202, 1.440, 2.780],
                                  'p_section_toll_max_value': [0.4752, 0.1930, 0.1029, 0.07285, 0.05803],
                                  'p_section_toll_cumulative_value': [6.782, 3.902, 2.924, 1.982, 1.042]
                                  },

    radial_K_implicit_oldfront= { 'radius_toll':   [0.062],
                                  'w_center_toll': [0.185],
                                  'w_section_toll_max_value': [0.01609, 0.01351, 0.04430, 0.09152, 0.07827],
                                  'w_section_toll_cumulative_value': [0.2904, 0.1846, 0.5634, 2.250, 0.9036],
                                  'p_section_toll_max_value': [0.5016, 0.2011, 0.1055, 0.0733, 0.05803],
                                  'p_section_toll_cumulative_value': [11.39, 4.346, 2.655, 0.8088, 0.6952]
}
)