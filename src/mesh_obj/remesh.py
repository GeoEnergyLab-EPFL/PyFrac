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
    # differences
    dne = (new_mesh.NumberOfElts - mesh.NumberOfElts)
    dnx = (new_mesh.nx - mesh.nx)
    dny = (new_mesh.ny - mesh.ny)

    old_indexes = np.array(list(range(0, mesh.NumberOfElts)))

    # If the direction is not none, then we have some sides to which we extend
    if not direction == None:
        # Now we loop over all five sides to get the new elements
        for side in range(4):
            # Check if the corresponding side is prone to a mesh extension (extension_side is a boolean indicating this)
            if direction[side]:
                # --- Extension in y --- #

        # if direction == 'top':
        #     new_indexes = old_indexes
        # elif direction == 'bottom':
        #     new_indexes = old_indexes + dne
        # elif direction == 'left':
        #     new_indexes = old_indexes + (np.floor(old_indexes / mesh.nx) + 1) * dnx
        # elif direction == 'right':
        #     new_indexes = old_indexes + np.floor(old_indexes / mesh.nx) * dnx
        # elif direction == 'horizontal':
        #     new_indexes = old_indexes + (np.floor(old_indexes / mesh.nx) + 1 / 2) * dnx
        # elif direction == 'vertical':
        #     new_indexes = old_indexes + dne / 2
        # else:
        #     new_indexes = old_indexes + 1 / 2 * dny * new_mesh.nx + (np.floor(old_indexes / mesh.nx) + 1 / 2) * dnx

    return new_indexes.astype(int)

def get_extensionsal_domain_limits(extension_sides, extension_factor, old_mesh, symmetric, logger):
    # get the initial values
    nx_init = old_mesh.nx
    ny_init = old_mesh.ny
    hx_init = old_mesh.hx
    hy_init = old_mesh.hx

    # Initiate the new solutions
    old_limits = old_mesh.domainLimits
    old_elems = [nx_init, ny_init]
    new_dirs = extension_sides

    # Now we loop over all five sides to get the new elements
    for side in range(4):
        # Check if the corresponding side is prone to a mesh extension (extension_side is a boolean indicating this)
        if extension_sides[side]:
            # --- Extension in y --- #
            if side == 0 or side == 1:
                # If we use symmetric properties we need to keep the symmetry of the problem. So we need to extend
                # symmetrically. We also extend by a mean value if both boundaries get touched.
                if symmetric or extension_sides[0]*extension_sides[1]:
                    # Calculating the number of elements to add
                    elems_add = int(ny_init * (np.mean(extension_factor[::2]) - 1))
                    # As we always have an odd number of elements we want to add an even number (to again have an odd
                    # number)
                    if elems_add % 2 != 0:
                        elems_add = elems_add + 1

                    # For symmetry, we extend in both directions
                    logger.info("Remeshing by extending in both vertical directions...")
                    new_limits = [[old_limits[2], old_limits[3]],
                                  [old_limits[0] - elems_add * hy_init, old_limits[1] + elems_add * hy_init]]
                    # Because now we already extended in both y directions we set the booleans to False
                    # (not to extend twice)
                    extension_sides[1] = False
                    extension_sides[0] = False
                    # We ensure we know where we did extend to
                    new_dirs[1] = True
                    new_dirs[0] = True

                    # Get the new amount of elements
                    elems = [old_elems[0], old_elems[1] + 2 * elems_add]
                else:
                    # Calculating the number of elements to add
                    elems_add = int(ny_init * (extension_factor[side] - 1))
                    # As we always have an odd number of elements we want to add an even number (to again have an odd
                    # number)
                    if elems_add % 2 != 0:
                        elems_add = elems_add + 1

                    # Extend in function of which boundary got hit
                    if side == 0:
                        # for no symmetry, we extend in the corresponding direction
                        logger.info("Extending mesh towards negative y...")
                        new_limits = [[old_limits[2], old_limits[3]],
                                      [old_limits[0] - elems_add * hy_init, old_limits[1]]]
                    elif side == 1:
                        # for no symmetry, we extend in the corresponding direction
                        logger.info("Extending mesh towards positive y...")
                        new_limits = [[old_limits[2], old_limits[3]],
                                      [old_limits[0], old_limits[1] + elems_add * hy_init]]

                    # Get the new amount of elements
                    elems = [old_elems[0], old_elems[1] + elems_add]
                    # We ensure we know where we did extend to
                    new_dirs[side] = True

            # --- Extension in x --- #
            if side == 2 or side == 3:
                # If we use symmetric properties we need to keep the symmetry of the problem. So we need to extend
                # symmetrically. We extend by a mean value in that case and if both boundaries get touched.
                if symmetric or extension_sides[2] * extension_sides[3]:
                    # Calculating the number of elements to add
                    elems_add = int(nx_init * (np.mean(extension_factor[2::]) - 1))
                    # As we always have an odd number of elements we want to add an even number (to again have an odd
                    # number)
                    if elems_add % 2 != 0:
                        elems_add = elems_add + 1

                    # For symmetry, we extend in both directions
                    logger.info("Remeshing by extending in both horizontal directions...")
                    new_limits = [[old_limits[2] - elems_add * hx_init, old_limits[3] + hx_init],
                                  [old_limits[0], old_limits[1]]]
                    # Because now we already extended in both y directions we set the booleans to False
                    # (not to extend twice)
                    extension_sides[3] = False
                    extension_sides[2] = False
                    # We ensure we know where we did extend to
                    new_dirs[3] = True
                    new_dirs[2] = True

                    # Get the new amount of elements
                    elems = [old_elems[0] + 2 * elems_add, old_elems[1]]
                else:
                    # Calculating the number of elements to add
                    elems_add = int(nx_init * (extension_factor[side] - 1))
                    # As we always have an odd number of elements we want to add an even number (to again have an odd
                    # number)
                    if elems_add % 2 != 0:
                        elems_add = elems_add + 1

                    # Extend in function of which boundary got hit
                    if side == 0:
                        # for no symmetry, we extend in the corresponding direction
                        logger.info("Extending mesh towards negative x...")
                        new_limits = [[old_limits[2] - elems_add * hx_init, old_limits[3]],
                                      [old_limits[0], old_limits[1]]]
                    elif side == 1:
                        # for no symmetry, we extend in the corresponding direction
                        logger.info("Extending mesh towards positive x...")
                        new_limits = [[old_limits[2], old_limits[3] + elems_add * hx_init],
                                      [old_limits[0], old_limits[1]]]

                    # Get the new amount of elements
                    elems = [old_elems[0] + elems_add, old_elems[1]]
                    # We ensure we know where we did extend to
                    new_dirs[side] = True

        # assigne the new limits and elements
        old_limits = new_limits
        old_elems = elems

        # ensure not to check a side twice
        extension_sides[side] = False

    return old_limits, old_elems, new_dirs