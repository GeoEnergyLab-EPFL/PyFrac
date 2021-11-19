# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Tue Dec 27 17:41:56 2016.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# external imports
import numpy as np

def mapping_old_indexes(new_mesh, mesh, direction = None):
    """
    Function to get the mapping of the indexes
    """
    dne = (new_mesh.NumberOfElts - mesh.NumberOfElts)
    dnx = (new_mesh.nx - mesh.nx)
    dny = (new_mesh.ny - mesh.ny)

    old_indexes = np.array(list(range(0, mesh.NumberOfElts)))

    if direction == 'top':
        new_indexes = old_indexes
    elif direction == 'bottom':
        new_indexes = old_indexes + dne
    elif direction == 'left':
        new_indexes = old_indexes + (np.floor(old_indexes / mesh.nx) + 1) * dnx
    elif direction == 'right':
        new_indexes = old_indexes + np.floor(old_indexes / mesh.nx) * dnx
    elif direction == 'horizontal':
        new_indexes = old_indexes + (np.floor(old_indexes / mesh.nx) + 1 / 2) * dnx
    elif direction == 'vertical':
        new_indexes = old_indexes + dne / 2
    else:
        new_indexes = old_indexes + 1 / 2 * dny * new_mesh.nx + (np.floor(old_indexes / mesh.nx) + 1 / 2) * dnx

    return new_indexes.astype(int)