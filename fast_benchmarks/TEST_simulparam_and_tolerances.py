# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo on 01.09.20.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""
import numpy as np
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
                   'finalTime': 5e2, 'initialR': 0.15})



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
                                  'w_section_toll_max_value': np.array([7. / 1000., 1.1 / 100., 1.7 / 100., 1.1 / 100., 1.8 / 100]),
                                  'w_section_toll_cumulative_value': np.array([8. / 100., 14. / 100., 0.18, 0.2, 0.19]),
                                  'p_section_toll_max_value': np.array([7. / 1000., 1.1 / 100., 1.7 / 100., 1.1 / 100., 1.8 / 100]),
                                  'p_section_toll_cumulative_value': np.array([8. / 100., 14. / 100., 0.18, 0.2, 0.19])
                                  },

    radial_M_implicit_oldfront= { 'radius_toll':   [0.05],
                                  'w_center_toll': [0.05],
                                  'w_section_toll_max_value': np.array([7. / 1000., 1.1 / 100., 1.7 / 100., 1.1 / 100., 1.8 / 100]),
                                  'w_section_toll_cumulative_value': np.array([8. / 100., 14. / 100., 0.18, 0.2, 0.19]),
                                  'p_section_toll_max_value': np.array([7. / 1000., 1.1 / 100., 1.7 / 100., 1.1 / 100., 1.8 / 100]),
                                  'p_section_toll_cumulative_value': np.array([8. / 100., 14. / 100., 0.18, 0.2, 0.19])
                                  },

    radial_K_explicit_newfront= { 'radius_toll':   [0.05],
                                  'w_center_toll': [0.05],
                                  'w_section_toll_max_value': np.array([7. / 1000., 1.1 / 100., 1.7 / 100., 1.1 / 100., 1.8 / 100]),
                                  'w_section_toll_cumulative_value': np.array([8. / 100., 14. / 100., 0.18, 0.2, 0.19]),
                                  'p_section_toll_max_value': np.array([7. / 1000., 1.1 / 100., 1.7 / 100., 1.1 / 100., 1.8 / 100]),
                                  'p_section_toll_cumulative_value': np.array([8. / 100., 14. / 100., 0.18, 0.2, 0.19])
                                  },

    radial_K_explicit_oldfront= { 'radius_toll':   [0.05],
                                  'w_center_toll': [0.05],
                                  'w_section_toll_max_value': np.array([7. / 1000., 1.1 / 100., 1.7 / 100., 1.1 / 100., 1.8 / 100]),
                                  'w_section_toll_cumulative_value': np.array([8. / 100., 14. / 100., 0.18, 0.2, 0.19]),
                                  'p_section_toll_max_value': np.array([7. / 1000., 1.1 / 100., 1.7 / 100., 1.1 / 100., 1.8 / 100]),
                                  'p_section_toll_cumulative_value': np.array([8. / 100., 14. / 100., 0.18, 0.2, 0.19])
                                  },

    radial_K_implicit_newfront= { 'radius_toll':   [0.05],
                                  'w_center_toll': [0.05],
                                  'w_section_toll_max_value': np.array([7. / 1000., 1.1 / 100., 1.7 / 100., 1.1 / 100., 1.8 / 100]),
                                  'w_section_toll_cumulative_value': np.array([8. / 100., 14. / 100., 0.18, 0.2, 0.19]),
                                  'p_section_toll_max_value': np.array([7. / 1000., 1.1 / 100., 1.7 / 100., 1.1 / 100., 1.8 / 100]),
                                  'p_section_toll_cumulative_value': np.array([8. / 100., 14. / 100., 0.18, 0.2, 0.19])
                                  },

    radial_K_implicit_oldfront= { 'radius_toll':   [0.05],
                                  'w_center_toll': [0.05],
                                  'w_section_toll_max_value': np.array([7. / 1000., 1.1 /100., 1.7/100., 1.1 /100., 1.8/100]),
                                  'w_section_toll_cumulative_value': np.array([8. / 100.,  14./100.,  0.18,     0.2,       0.19]),
                                  'p_section_toll_max_value': np.array([7. / 1000., 1.1 /100., 1.7/100., 1.1 /100., 1.8/100]),
                                  'p_section_toll_cumulative_value': np.array([8. / 100.,  14./100.,  0.18,     0.2,       0.19])
}
)

