# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Tue Dec 27 17:41:56 2016.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# external imports
import numpy as np
import logging
import json
import subprocess
import pickle
from array import array
import os, sys

# internal imports
from solid.elasticity_kernels.isotropic_R0_elem import load_isotropic_elasticity_matrix
from solid.elasticity_toeplitz import elasticity_matrix_toepliz


def get_Cij_Matrix(youngs_mod, nu):
    k = youngs_mod / (3 * (1 - 2 * nu))
    la = (3 * k * (3 * k - youngs_mod)) / (9 * k - youngs_mod)
    mu = 3 / 2 * (k - la)

    Cij = np.zeros((6, 6), dtype=np.float64)
    Cij[0][0] = (la + 2 * mu) * (1 + 0.00007)
    Cij[0][2] = la * (1 + 0.00005)
    Cij[2][2] = (la + 2 * mu) * (1 + 0.00009)
    Cij[3][3] = mu * (1 + 0.00001)
    Cij[5][5] = mu * (1 + 0.00003)
    Cij[0][1] = Cij[0][0] - 2 * Cij[5][5]

    return Cij


# --------------------------------------------------------------------------------------------------------------------------
def load_TI_elasticity_matrix(Lx, Ly, nx, ny, Cij, TI_KernelExecPath, toeplitz = False):
    """
    Create the elasticity matrix for transversely isotropic materials.  It is under development and will be refactored
    soon.

    Args:
        Mesh (object CartesianMesh):        -- a mesh object describing the domain.
        mat_prop (MaterialProperties):      -- the MaterialProperties object giving the material properties.
        TI_KernelExecPath (string):         -- the folder containing the executable to calculate transverse isotropic
                                               kernel or kernel with free surface.
        toeplitz (bool)                     -- if that is true then the matrix will be compressed exploiting its Toeplitz structure

    Returns:
        C (ndarray):                        -- the elasticity matrix.
    """
    log = logging.getLogger('PyFrac.load_TI_elasticity_matrix')
    data = {'Solid parameters': {'C11': Cij[0][0],
                                 'C12': Cij[0][1],
                                 'C13': Cij[0][2],
                                 'C33': Cij[2][2],
                                 'C44': Cij[3][3]},
            'Mesh': {'L1': Lx,
                     'L3': Ly,
                     'n1': nx,
                     'n3': ny},
            'Options':{'toeplitz_compr': toeplitz}
            }

    log.info('Writing parameters to a file...')
    curr_directory = os.getcwd()
    os.chdir(TI_KernelExecPath)
    with open('stiffness_matrix.json', 'w') as outfile:
        json.dump(data, outfile, indent=3)

    if "win32" in sys.platform or "win64" in sys.platform:
        suffix = ""
    else:
        suffix = "./"

    # Read the elasticity matrix from the npy file
    log.info('running C++ process...')
    subprocess.run(suffix + 'TI_elasticity_kernel', shell=True)

    log.info('Reading global TI elasticity matrix...')
    if toeplitz:
        amount_of_data_to_get = data['Mesh']['n1'] * data['Mesh']['n3']
    else:
        amount_of_data_to_get = pow(data['Mesh']['n1'] * data['Mesh']['n3'], 2)
    try:
        file = open('StrainResult.bin', "rb")
        C = array('d')
        C.fromfile(file, amount_of_data_to_get)

        if toeplitz:
            C = np.asarray(C)
        else:
            C = np.reshape(C,
                           (data['Mesh']['n1'] * data['Mesh']['n3'],
                            data['Mesh']['n1'] * data['Mesh']['n3']))


    except FileNotFoundError:
        # if 'CMatrix' file is not found
        raise SystemExit('file not found')

    os.chdir(curr_directory)

    return C


# ----------------------------------------------------------------------------------------------------------------------


def load_elasticity_matrix(Mesh, EPrime):
    """
    The function loads the elasticity matrix from the saved file. If the loaded matrix is not compatible with respect
    to the current mesh or plain strain modulus, the compatible matrix is computed and saved in a file. If the file is
    not found, the elasticity matrix is computed and saved in a file with the name 'CMatrix'.

    Arguments:
        Mesh (CartesianMesh):           -- a mesh object describing the domain.
        EPrime (float):                 -- plain strain modulus.

    Returns:
         C (ndarray):                   -- the elasticity matrix.
    """
    log = logging.getLogger('PyFrac.load_elasticity_matrix')
    log.info('Reading global elasticity matrix...')
    try:
        with open('CMatrix', 'rb') as input_file:
            (C, MeshLoaded, EPrimeLoaded) = pickle.load(input_file)
        # check if the loaded matrix is correct with respect to the current mesh and plain strain modulus
        if (Mesh.nx, Mesh.ny, Mesh.Lx, Mesh.Ly, EPrime) == (MeshLoaded.nx, MeshLoaded.ny, MeshLoaded.Lx,
                                                            MeshLoaded.Ly, EPrimeLoaded):
            return C
        else:
            log.warning(
                'The loaded matrix is not correct with respect to the current mesh or the current plain strain modulus.'
                '\nMaking global matrix...')
            C = load_isotropic_elasticity_matrix(Mesh, EPrime)
            Elast = (C, Mesh, EPrime)
            with open('CMatrix', 'wb') as output:
                pickle.dump(Elast, output, -1)
            log.info("Done!")
            return C
    except FileNotFoundError:
        # if 'CMatrix' file is not found
        log.error('file not found\nBuilding the global elasticity matrix...')
        C = load_isotropic_elasticity_matrix(Mesh, EPrime)
        Elast = (C, Mesh, EPrime)
        with open('CMatrix', 'wb') as output:
            pickle.dump(Elast, output, -1)
        log.info("Done!")
        return C

#-----------------------------------------------------------------------------------------------------------------------

def TI_plain_strain_modulus(alpha, Cij):
    """
    This function computes the plain strain elasticity modulus in transverse isotropic medium. The modulus is a function
    of the orientation of the fracture front with respect to the bedding plane. This functions is used for the tip
    inversion and for evaluation of the fracture volume for the case of TI elasticity.

    Arguments:
        alpha (ndarray-float)             -- the angle inscribed by the perpendiculars drawn on the front from the \
                                             ribbon cell centers.
        Cij (ndarray-float)               -- the TI stiffness matrix in the canonical basis

    Returns:
        E' (ndarray-float)               -- plain strain TI elastic modulus.
    """

    C11 = Cij[0, 0]
    C12 = Cij[0, 1]
    C13 = Cij[0, 2]
    C33 = Cij[2, 2]
    C44 = Cij[3, 3]

    # we use the same notation for the elastic paramateres as S. Fata et al. (2013).

    alphag = (C11 * (C11-C12) * np.cos(alpha) ** 4 + (C11 * C13
                 - C12 * (C13 + 2 * C44)) * (np.cos(alpha) * np.sin(alpha)) ** 2
                 - (C13 ** 2 - C11 * C33 + 2 * C13 * C44) * np.sin(alpha) ** 4
                 + C11 * C44 * np.sin(2 * alpha) ** 2) / (C11 * (C11 - C12) * np.cos(alpha) ** 2
                                                          + 2 * C11 * C44 * np.sin(alpha) ** 2)

    gammag = ((C11 * np.cos(alpha) ** 4 + 2 * C13 * (np.cos(alpha) * np.sin(alpha)) ** 2
                 + C33 * np.sin(alpha) ** 4 + C44 * np.sin(2 * alpha) ** 2) / C11) ** 0.5

    deltag = ((C11 - C12) * (C11 + C12) * np.cos(alpha) ** 4
                 + 2 * (C11 - C12) * C13 * (np.cos(alpha) * np.sin(alpha)) ** 2
                 + (- C13 ** 2 + C11 * C33) * np.sin(alpha) ** 4
                 + C11 * C44 * np.sin(2 * alpha) ** 2) / (C11 * (2 * (alphag + gammag)) ** 0.5)

    Eprime = 2 * deltag / gammag

    return Eprime
#-----------------------------------------------------------------------------------------------------------------------


class load_TI_elasticity_matrix_toepliz(elasticity_matrix_toepliz):

    def __init__(self, Mesh, Cij, TI_KernelExecPath, C_precision=np.float64):

        useHMATdot = False # because one needs to implement TI in HMAT
        elas_prop_HMAT = None
        self.TI_KernelExecPath = TI_KernelExecPath
        super().__init__(Mesh, Cij, elas_prop_HMAT, C_precision, useHMATdot, kerneltype='Transverse-Isotropic')

    def reload_toepliz_Coe(self, Lx, Ly, nx, ny, hx, hy, mat_prop):
        out = load_TI_elasticity_matrix(Lx, Ly, nx, ny, mat_prop, self.TI_KernelExecPath, toeplitz = True)
        return out[0], out