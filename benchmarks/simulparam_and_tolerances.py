# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo on 01.09.20.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""
import numpy as np
tollerance = 1.20
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
                                'simulparam': 'radial_K'},

    radial_MtoK_explicit_oldfront = {'front_reconstruction': 'ILSA_orig',
                                     'front_advancement': 'explicit',
                                     'path': 'M_to_K',
                                     'simulparam': 'radial_MtoK'},

    radial_MtoK_explicit_newfront = {'front_reconstruction': 'LS_continousfront',
                                     'front_advancement': 'explicit',
                                     'path': 'M_to_K',
                                     'simulparam': 'radial_MtoK'},

    radial_MtoK_implicit_oldfront = {'front_reconstruction': 'ILSA_orig',
                                     'front_advancement': 'implicit',
                                     'path': 'M_to_K',
                                     'simulparam': 'radial_MtoK'},

    radial_MtoK_implicit_newfront = {'front_reconstruction': 'LS_continousfront',
                                     'front_advancement': 'implicit',
                                     'path': 'M_to_K',
                                     'simulparam': 'radial_MtoK'}
)

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
                   'finalTime': 1e5, 'initialR': 0.15},

    radial_MtoK={  'Lx': 2, 'Ly': 2,
                   'Nx': 41, 'Ny': 41,
                   'nu': 0.4, 'youngs_mod': 3.3e10, 'K_Ic': 1e6,
                   'Cl': 0., 'Q0': 0.01, 'viscosity': 0.001,
                   'finalTime': 1e9, 'initialTime': 0.05},
)



tolerances = dict(
    radial_M_explicit_newfront= { 'radius_toll':   [0.05],
                                  'w_center_toll': [0.05],
                                  'w_section_toll_max_value': tollerance*np.asarray([0.010114364763255092, 0.012659558692028283, 0.009118861299134862, 0.01180792636365191, 0.018547012986371514]),
                                  'w_section_toll_cumulative_value': tollerance*np.asarray([0.0746562877882185, 0.08251420936651116, 0.13431216672809554, 0.18320178050319896, 0.19922975411994992]),
                                  'p_section_toll_max_value': tollerance*np.asarray([1.0290667014704036, 0.07219088445562821, 0.025418768701084754, 0.014069425647468131, 0.009047589037292976]),
                                  'p_section_toll_cumulative_value': tollerance*np.asarray([4.863396385255593, 0.328976763167172, 0.1416254935121381, 0.08261348314046424, 0.04304543905820167])
                                  },

    radial_M_explicit_oldfront= { 'radius_toll':   [0.05],
                                  'w_center_toll': [0.05],
                                  'w_section_toll_max_value': tollerance*np.asarray([0.003463105575302189, 0.012551540786586601, 0.009266427355161033, 0.012018570358160008, 0.01776358574040192]),
                                  'w_section_toll_cumulative_value': tollerance*np.asarray([0.04940303998361687, 0.08528659756096339, 0.1405237858521095, 0.18768907094087145, 0.19938681590719642]),
                                  'p_section_toll_max_value': tollerance*np.asarray([0.34421168713300787, 0.07207177614891092, 0.025411591762158847, 0.014070919785078587, 0.009048557957832139]),
                                  'p_section_toll_cumulative_value': tollerance*np.asarray([1.7340578109726303, 0.3177133798741494, 0.11754940287272783, 0.07479601613864724, 0.04422871211938955])
                                  },

    radial_M_implicit_newfront= { 'radius_toll':   [0.05],
                                  'w_center_toll': [0.05],
                                  'w_section_toll_max_value': tollerance*np.asarray([0.004880975343805993, 0.010262367304877278, 0.012095130290441514, 0.022877454209472875, 0.02418652647775324]),
                                  'w_section_toll_cumulative_value': tollerance*np.asarray([0.07159952843060052, 0.13737053759671722, 0.18007063436978632, 0.25846963749054014, 0.2757035962019126]),
                                  'p_section_toll_max_value': tollerance*np.asarray([0.4030735794640704, 0.07223606649612363, 0.07309287073423798, 0.014127970716044061, 0.009098165166454383]),
                                  'p_section_toll_cumulative_value': tollerance*np.asarray([1.9974664294027917, 0.3293929059902019, 0.29002723906615946, 0.07568889201933869, 0.039251737583412265])
                                  },

    radial_M_implicit_oldfront= { 'radius_toll':   [0.05],
                                  'w_center_toll': [0.05],
                                  'w_section_toll_max_value': tollerance*np.asarray([0.00818235169174042, 0.010397051760354145, 0.009724919797212461, 0.018862003118571558, 0.021380511464288356]),
                                  'w_section_toll_cumulative_value': tollerance*np.asarray([0.061705465074829484, 0.1254053306251428, 0.14331104055709035, 0.22195730553544996, 0.2382750327902585]),
                                  'p_section_toll_max_value': tollerance*np.asarray([0.6324749522901629, 0.07217670485181127, 0.08263089645802132, 0.014053257156558252, 0.00909302901901285]),
                                  'p_section_toll_cumulative_value': tollerance*np.asarray([2.518828529325425, 0.2970705569800288, 0.2950766700128139, 0.07217006082689227, 0.03626527895203865])
                                  },

    radial_K_explicit_newfront= { 'radius_toll':   [0.05],
                                  'w_center_toll': [0.079],
                                  'w_section_toll_max_value': tollerance*np.asarray([0.005576723638976034, 0.0163119153974007, 0.0405708427174259, 0.059464341846782676, 0.05978548607735765]),
                                  'w_section_toll_cumulative_value': tollerance*np.asarray([0.08578384605444545, 0.20689989958674843, 0.4371339510854303, 0.4946198546294071, 0.7397569240418214]),
                                  'p_section_toll_max_value': tollerance*np.asarray([0.505127187853307, 0.20109475552592493, 0.10563619334172399, 0.07382143663099446, 0.0580238769751372]),
                                  'p_section_toll_cumulative_value': tollerance*np.asarray([8.030251805473839, 4.374320443229788, 0.6071746082995599, 1.727778006524236, 0.6810039762800363])
                                  },

    radial_K_explicit_oldfront= { 'radius_toll':   [0.05],
                                  'w_center_toll': [0.081],
                                  'w_section_toll_max_value': tollerance*np.asarray([0.004657364168882336, 0.015365379786673562, 0.04124571315412114, 0.06307185596063047, 0.06307286516888941]),
                                  'w_section_toll_cumulative_value': tollerance*np.asarray([0.08433078356975027, 0.21492750126712817, 0.443137597757512, 0.5483068270885866, 0.7400532875545803]),
                                  'p_section_toll_max_value': tollerance*np.asarray([0.505127187853307, 0.20109475552592493, 0.10563619334172399, 0.07382143663099446, 0.0580238769751372]),
                                  'p_section_toll_cumulative_value': tollerance*np.asarray([8.0422407805359, 4.395812116944476, 0.6099044318389214, 1.7410417139560777, 0.6798303904637699])
                                  },

    radial_K_implicit_newfront= { 'radius_toll':   [0.074],
                                  'w_center_toll': [0.21],
                                  'w_section_toll_max_value': tollerance*np.asarray([0.018359540892071917, 0.01571560013620088, 0.04466881695605066, 0.19301207711285495, 0.16578774995491702]),
                                  'w_section_toll_cumulative_value': tollerance*np.asarray([0.29438692338900774, 0.22817062344254876, 0.5577647811989626, 3.0943913646854515, 3.36992046733513]),
                                  'p_section_toll_max_value': tollerance*np.asarray([0.505127187853307, 0.20109475552592493, 0.10563619334172399, 0.07382143663099446, 0.0580238769751372]),
                                  'p_section_toll_cumulative_value': tollerance*np.asarray([11.674538269556486, 4.427801457521251, 0.6640452988667384, 0.9981660517343405, 1.1768907768589338])
                                  },

    radial_K_implicit_oldfront= { 'radius_toll':   [0.062],
                                  'w_center_toll': [0.185],
                                  'w_section_toll_max_value': tollerance*np.asarray([0.01252526754957263, 0.023417182611892313, 0.04529184785762364, 0.11003074255575, 0.07077950153844755]),
                                  'w_section_toll_cumulative_value': tollerance*np.asarray([0.22105493672659873, 0.2417927047460269, 0.5312834784926783, 1.992671257583858, 0.9252994045224927]),
                                  'p_section_toll_max_value': tollerance*np.asarray([0.505127187853307, 0.20109475552592493, 0.10563619334172399, 0.07382143663099446, 0.0580238769751372]),
                                  'p_section_toll_cumulative_value': tollerance*np.asarray([10.770874986865927, 4.393265945008475, 0.6492505338105509, 0.7834893918056701, 0.7035883579162517])
                                },

    radial_MtoK_explicit_oldfront = {'front_reconstruction': 'ILSA_orig',
                                     'front_advancement': 'explicit',
                                     'simulparam': 'radial_MtoK'},

    radial_MtoK_explicit_newfront = {'front_reconstruction': 'LS_continousfront',
                                     'front_advancement': 'explicit',
                                     'simulparam': 'radial_MtoK'},

    radial_MtoK_implicit_oldfront = {'front_reconstruction': 'ILSA_orig',
                                     'front_advancement': 'implicit',
                                     'simulparam': 'radial_MtoK'},
    radial_MtoK_implicit_newfront = {'front_reconstruction': 'LS_continousfront',
                                     'front_advancement': 'implicit',
                                     'simulparam': 'radial_MtoK'}
)