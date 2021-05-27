# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Tue Dec 27 17:41:56 2016.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""
from numba import njit
from numba import config, threading_layer
from scipy.sparse.linalg import LinearOperator
import numpy as np
import logging
import json
import subprocess
import pickle
from array import array
import os, sys

def load_isotropic_elasticity_matrix(Mesh, Ep):
    """
    Evaluate the elasticity matrix for the whole mesh.
    Arguments:
        Mesh (object CartesianMesh):    -- a mesh object describing the domain.
        Ep (float):                     -- plain strain modulus.
    Returns:
        ndarray-float:                  -- the elasticity matrix.
    """

    """
    a and b are the half breadth and height of a cell
     ___________________________________
    |           |           |           |
    |           |           |           |
    |     .     |     .     |     .     |
    |           |           |           |
    |___________|___________|___________|
    |           |     ^     |           |
    |           |   b |     |           |
    |     .     |     .<--->|     .     |
    |           |        a  |           |
    |___________|___________|___________|
    |           |           |           |
    |           |           |           |
    |     .     |     .     |     .     |
    |           |           |           |
    |___________|___________|___________|
       
    """

    a = Mesh.hx / 2.
    b = Mesh.hy / 2.
    Ne = Mesh.NumberOfElts

    C = np.empty([Ne, Ne], dtype=np.float32)

    for i in range(0, Ne):
        x = Mesh.CenterCoor[i, 0] - Mesh.CenterCoor[:, 0]
        y = Mesh.CenterCoor[i, 1] - Mesh.CenterCoor[:, 1]

        C[i] = (Ep / (8. * np.pi)) * (
                np.sqrt(np.square(a - x) + np.square(b - y)) / ((a - x) * (b - y)) + np.sqrt(
            np.square(a + x) + np.square(b - y)
        ) / ((a + x) * (b - y)) + np.sqrt(np.square(a - x) + np.square(b + y)) / ((a - x) * (b + y)) + np.sqrt(
            np.square(a + x) + np.square(b + y)) / ((a + x) * (b + y)))

    return C
# -----------------------------------------------------------------------------------------------------------------------

def get_isotropic_el_self_eff(hx, hy, Ep):
    """
    Evaluate the self effect term (diagonal value) for isotropic elasticity.
    Arguments:
        hx (float):                     -- x size of a mesh cell
        hy (float):                     -- y size of a mesh cell
        Ep (float):                     -- plain strain modulus.
    Returns:
        ndarray-float:                  -- the diagonal term.
    """

    a = hx / 2.                           #Lx/nx-1
    b = hy / 2.                           #Ly/ny-1
    bb = b * b
    aa = a * a
    sqrt_aa_p_bb = np.sqrt(aa + bb) / (a * b)
    return sqrt_aa_p_bb * Ep / (2. * np.pi)

# set the threading layer before any parallel target compilation
#config.THREADING_LAYER = 'threadsafe'


#@njit( parallel = True)
@njit()
def matvec_fast(uk, elemX, dimX,  elemY, dimY, nx, C_toeplotz_coe):
    #elemX = self.domain_INDX
    #elemY = self.codomain_INDX
    #nx  # number of element in x direction in the global mesh
    # dimX = elemX.size  # number of elements to consider on x axis
    # dimY = elemY.size  # number of elements to consider on y axis

    res = np.empty(dimY, dtype=np.float64)  # subvector result

    iY = np.floor_divide(elemY, nx)
    jY = elemY - nx * iY
    iX = np.floor_divide(elemX, nx)
    jX = elemX - nx * iX
    #row_temp = np.empty(dimX, dtype=np.float64)  # subvector result
    for iter1 in range(dimY):
        # assembly matrix row
        i1 = iY[iter1]
        j1 = jY[iter1]

        #row_temp[0:dimX] = C_toeplotz_coe[np.abs(j1 - jX) + nx * np.abs(i1 - iX)]
        res[iter1] = np.dot(C_toeplotz_coe[np.abs(j1 - jX) + nx * np.abs(i1 - iX)],uk)

    return res


class load_isotropic_elasticity_matrix_toepliz(LinearOperator):
    def __init__(self, Mesh, Ep):
        self.Ep = Ep
        const = (Ep / (8. * np.pi))
        self.const = const
        self.reload(Mesh)

    def reload(self, Mesh):
        hx = Mesh.hx
        hy = Mesh.hy
        a = hx / 2.
        b = hy / 2.
        nx = Mesh.nx
        ny = Mesh.ny
        self.a = a
        self.b = b
        self.nx = nx
        const = self.const

        ################
        # Cdot section #
        ################
        # diagonal value of the matrix
        self.diag_val = get_isotropic_el_self_eff(hx, hy, self.Ep)

        # define the size = number of elements in the mesh
        self.C_size_ = int(Mesh.nx * Mesh.ny)

        # define the total number of unknowns to be output by the matvet method
        self.matvec_size_ = self.C_size_

        # it is mandatory to define shape and dtype of the dot product
        self.dtype_ = float
        self.shape_ = (self.matvec_size_, self.matvec_size_)
        super().__init__(self.dtype_, self.shape_)

        self._set_domain_IDX(np.arange(self.C_size_))
        self._set_codomain_IDX(np.arange(self.C_size_))
        ################ END Cdot SECTION ###################
        """
        Let us make some definitions:
        cartesian mesh             := a structured rectangular mesh of (nx,ny) cells of rectaungular shape
        
                                            |<------------nx----------->|
                                        _    ___ ___ ___ ___ ___ ___ ___
                                        |   | . | . | . | . | . | . | . |
                                        |   |___|___|___|___|___|___|___|
                                        ny  | . | . | . | . | . | . | . |  
                                        |   |___|___|___|___|___|___|___|   y
                                        |   | . | . | . | . | . | . | . |   |
                                        -   |___|___|___|___|___|___|___|   |____x  
                                       
                                       the cell centers are marked by .
         
        set of unique distances    := given a set of cells in a cartesian mesh, consider the set of unique distances 
                                      between any pair of cell centers.
        set of unique coefficients := given a set of unique distances then consider the interaction coefficients
                                      obtained from them
                                      
        C_toeplotz_coe             := An array of size (nx*ny), populated with the unique coefficients. 
        
        Matematically speaking:
        for i in (0,ny) and j in (0,nx) take the set of combinations (i,j) such that [i^2 y^2 + j^2 x^2]^1/2 is unique
        """
        C_toeplotz_coe = np.empty(ny*nx, dtype=np.float64)
        xindrange = np.asarray(range(nx))
        xrange = xindrange * hx
        for i in range(ny):
            y = i*hy
            amx = a - xrange
            apx = a + xrange
            bmy = b - y
            bpy = b + y
            C_toeplotz_coe[i*nx:(i+1)*nx] = const * (np.sqrt(np.square(amx) + np.square(bmy)) / (amx * bmy)
                                                            + np.sqrt(np.square(apx) + np.square(bmy)) / (apx * bmy)
                                                            + np.sqrt(np.square(amx) + np.square(bpy)) / (amx * bpy)
                                                            + np.sqrt(np.square(apx) + np.square(bpy)) / (apx * bpy))
        self.C_toeplotz_coe = C_toeplotz_coe

    def __getitem__(self, elementsXY,different_strategy=True):
        """
        critical call: it should be as fast as possible
        :param elemX: (numpy array) columns to take
        :param elemY: (numpy array) rows to take
        :return: submatrix of C
        """

        elemX = elementsXY[1].flatten()
        elemY = elementsXY[0].flatten()
        dimX = elemX.size  # number of elements to consider on x axis
        dimY = elemY.size  # number of elements to consider on y axis

        if dimX == 0 or dimY == 0:
            return np.empty((dimY, dimX),dtype=np.float32)
        else:
            nx = self.nx  # number of element in x direction in the global mesh
            C_sub = np.empty((dimY, dimX), dtype=np.float32)  # submatrix of C
            localC_toeplotz_coe = np.copy(self.C_toeplotz_coe)  # local access is faster
            if dimX != dimY:
                iY = np.floor_divide(elemY, nx)
                jY = elemY - nx * iY
                iX = np.floor_divide(elemX, nx)
                jX = elemX - nx * iX
                #if not different_strategy:
                # strategy 1
                for iter1 in range(dimY):
                    i1 = iY[iter1]
                    j1 = jY[iter1]
                    C_sub[iter1, 0:dimX] = localC_toeplotz_coe[np.abs(j1 - jX) + nx * np.abs(i1 - iX)]
                # else:
                    # strategy 2 - more memory expensive than 1
                    #C_sub=np.take(localC_toeplotz_coe, np.abs(np.array([jY]).T- jX)+ nx * np.abs(np.array([iY]).T- iX))

                    # strategy 3 - more memory expensive than 1
                    # J1, J2 = np.meshgrid(jX, jY)
                    # I1, I2 = np.meshgrid(iX, iY)
                    # C_sub = np.take(localC_toeplotz_coe, np.abs(J1 - J2) + nx * np.abs(I1 - I2))

                    # strategy 4 - slower
                    #C_sub = np.asarray(list(map(lambda x: localC_toeplotz_coe[np.abs(jY[x] - jX) + nx * np.abs(iY[x] - iX)],range(dimY))))
                return C_sub

            elif dimX == dimY and np.all((elemY == elemX)):
                i = np.floor_divide(elemX, nx)
                j = elemX - nx * i
                # if not different_strategy:
                    # strategy 1
                for iter1 in range(dimX):
                    i1 = i[iter1]
                    j1 = j[iter1]
                    C_sub[iter1, 0:dimX] = localC_toeplotz_coe[np.abs(j - j1) + nx * np.abs(i - i1)]
                # else:
                        # strategy 2 - more memory expensive than 1
                        #C_sub = np.take(localC_toeplotz_coe, np.abs(np.array([j]).T - j) + nx * np.abs(np.array([i]).T - i))

                        # strategy 3 - more memory expensive than 1
                        # J1, J2 = np.meshgrid(j, j)
                        # I1, I2 = np.meshgrid(i, i)
                        # C_sub = np.take(localC_toeplotz_coe, np.abs(J1-J2) + nx * np.abs(I1-I2))

                        # strategy 4 - slower
                        # C_sub = np.asarray(list(map(lambda x: localC_toeplotz_coe[np.abs(j[x] - j) + nx * np.abs(i[x] - i)],range(dimY))))
                return C_sub

            else:
                iY = np.floor_divide(elemY, nx)
                jY = elemY - nx * iY
                iX = np.floor_divide(elemX, nx)
                jX = elemX - nx * iX
                # if not different_strategy:
                    # strategy 1
                for iter1 in range(dimY):
                    i1 = iY[iter1]
                    j1 = jY[iter1]
                    C_sub[iter1, 0:dimX] = localC_toeplotz_coe[np.abs(j1 - jX) + nx * np.abs(i1 - iX)]
                # else:
                    # strategy 2 - more memory expensive than 1
                    #C_sub=np.take(localC_toeplotz_coe, np.abs(np.array([jY]).T- jX)+ nx * np.abs(np.array([iY]).T- iX))

                    # strategy 3 - more memory expensive than 1
                    # J1, J2 = np.meshgrid(jX, jY)
                    # I1, I2 = np.meshgrid(iX, iY)
                    # C_sub = np.take(localC_toeplotz_coe, np.abs(J2 - J1) + nx * np.abs(I2 - I1))

                    # strategy 4 - slower
                    # C_sub = np.asarray(list(map(lambda x: localC_toeplotz_coe[np.abs(jY[x] - jX) + nx * np.abs(iY[x] - iX)],range(dimY))))

                return C_sub

    def _matvec(self,uk):
        return matvec_fast(uk, self.domain_INDX,self.domain_INDX.size, self.codomain_INDX, self.codomain_INDX.size , self.nx, self.C_toeplotz_coe)
        # elemX = self.domain_INDX
        # elemY = self.codomain_INDX
        # dimX = elemX.size  # number of elements to consider on x axis
        # dimY = elemY.size  # number of elements to consider on y axis
        # nx = self.nx  # number of element in x direction in the global mesh
        # row_temp = np.empty(dimX, dtype=np.float64)  # subvector result
        # res = np.empty(dimY, dtype=np.float64)  # subvector result
        #
        # localC_toeplotz_coe = self.C_toeplotz_coe
        #
        # iY = np.floor_divide(elemY, nx)
        # jY = elemY - nx * iY
        # iX = np.floor_divide(elemX, nx)
        # jX = elemX - nx * iX
        #
        # for iter1 in range(dimY):
        #     # assembly matrix row
        #     i1 = iY[iter1]
        #     j1 = jY[iter1]
        #     row_temp[0:dimX] = localC_toeplotz_coe[np.abs(j1 - jX) + nx * np.abs(i1 - iX)]
        #     res[iter1] = np.dot(row_temp,uk)
        #
        # return res

    # def _matvec(self, uk):
    #     """
    #     E.uk
    #     (E + DiagTipCorrection).uk
    #     """
    #     traction = self._matvec_par(uk)
    #
    #     # TIP CORRECTION TO BE LOOKED AGAIN
    #     # if self.enable_tip_corr:
    #     #     # make tip correction
    #     #     effective_corrINDX = np.intersect1d(self.tipcorrINDX, self.domain_INDX) #take the correction only in the column where needed
    #     #     #corr_array = np.zeros(self.C_size_) #zero correction array
    #     #     #corr_array[effective_corrINDX] = self.tipcorr[effective_corrINDX] #fill it with the corrected val
    #     #     corr_array = copy.deepcopy(self.tipcorr[effective_corrINDX])  # zero correction array
    #     #     corr_array = np.multiply(corr_array,uk) #multiply
    #     #     traction = traction + corr_array #<---- to be changed
    #
    #     return traction

    # def _set_tipcorr(self, correction_val, correction_INDX):
    #     self.tipcorr = np.zeros(self.C_size_) #initialize to zero
    #     self.tipcorr[correction_INDX] = correction_val #set the correction where needed
    #     self.tipcorrINDX = correction_INDX #save the indexes of the nonzero val
    #     self.enable_tip_corr = True

    def _set_domain_IDX(self, domainIDX):
        """
        General example:
        domain indexes are [1 , 2] of NON ZERO elements used to make the dot product
        codomain indexes are [0, 2] of elements returned after the dot product
        o o o o    0 <-0    x <-0
        o o o o    x <-1  = o <-1
        o o o o    x <-2    x <-2
        o o o o    0 <-3    o <-3
        """
        self.domain_INDX = domainIDX

    def _set_codomain_IDX(self, codomainIDX):
        """
        General example:
        domain indexes are [1 , 2] of NON ZERO elements used to make the dot product
        codomain indexes are [0, 2] of elements returned after the dot product
        o o o o    0 <-0    x <-0
        o o o o    x <-1  = o <-1
        o o o o    x <-2    x <-2
        o o o o    0 <-3    o <-3
        """
        self.codomain_INDX = codomainIDX
        self._changeShape(codomainIDX.size)

    def _changeShape(self, shape_):
        self.matvec_size_ = shape_
        self.shape_ = (shape_, shape_)
        super().__init__(self.dtype_, self.shape_)



# -----------------------------------------------------------------------------------------------------------------------
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
def load_TI_elasticity_matrix(Mesh, mat_prop, sim_prop):
    """
    Create the elasticity matrix for transversely isotropic materials.  It is under development and will be refactored
    soon.

    Args:
        Mesh (object CartesianMesh):        -- a mesh object describing the domain.
        mat_prop (MaterialProperties):      -- the MaterialProperties object giving the material properties.
        sim_prop (SimulationProperties):    -- the SimulationProperties object giving the numerical parameters to be
                                               used in the simulation.

    Returns:
        C (ndarray):                        -- the elasticity matrix.
    """
    log = logging.getLogger('PyFrac.load_TI_elasticity_matrix')
    data = {'Solid parameters': {'C11': mat_prop.Cij[0][0],
                                 'C12': mat_prop.Cij[0][1],
                                 'C13': mat_prop.Cij[0][2],
                                 'C33': mat_prop.Cij[2][2],
                                 'C44': mat_prop.Cij[3][3]},
            'Mesh':             {'L1': Mesh.Lx,
                                 'L3': Mesh.Ly,
                                 'n1': Mesh.nx,
                                 'n3': Mesh.ny}
            }

    log.info('Writing parameters to a file...')
    curr_directory = os.getcwd()
    os.chdir(sim_prop.TI_KernelExecPath)
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
    try:
        file = open('StrainResult.bin', "rb")
        C = array('d')
        C.fromfile(file, pow(data['Mesh']['n1'] * data['Mesh']['n3'], 2))
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

# -----------------------------------------------------------------------------------------------------------------------

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