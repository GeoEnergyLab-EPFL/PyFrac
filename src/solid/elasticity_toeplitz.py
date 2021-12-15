# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo on Tue Nov 2 15:09:38 2021.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""
# external imports
import numpy as np
from numba import njit, prange
from scipy.sparse import coo_matrix
from numba import config
from numba.typed import List
from scipy.sparse.linalg import LinearOperator
import time
# import random
# import time
# import multiprocessing
# import copy
# from math import floor

# local imports
from solid.elasticity_isotropic_HMAT_hook import Hdot_3DR0opening
from utilities.utility import append_new_line


# set the threading layer before any parallel target compilation
# 'workqueue' is builtin
# config.THREADING_LAYER = 'workqueue' #'workqueue' , 'threadsafe' ,'tbb', 'omp'
config.THREADING_LAYER = 'workqueue'  # 'workqueue', 'threadsafe' ,'tbb', 'omp'


@njit(parallel=True, fastmath=True, nogil=True)  # <------parallel compilation
def matvec_fast(uk, elemX, elemY, dimY, nx, C_toeplitz_coe, C_precision):
    # uk (numpy array), vector to which multiply the matrix C
    # nx (int), n. of element in x direction in the cartesian mesh
    # elemX (numpy array), IDs of elements to consider on x axis of the mesh
    # elemY (numpy array), IDs of elements to consider on y axis of the mesh
    # dimY (int), length(elemY) = length(elemX)
    # C_toeplitz_coe (numpy array), array containing the N unique coefficients of the matrix C of size NxN
    # C_precision (e.g.: float)

    # 1) vector where to store the result of the dot product
    res = np.empty(dimY, dtype=C_precision)

    # 2) some indexes to build the row of a submatrix of C from the array of its unique entries
    iY = np.floor_divide(elemY, nx)
    jY = elemY - nx * iY
    iX = np.floor_divide(elemX, nx)
    jX = elemX - nx * iX

    iX *= nx
    iY *= nx

    # 3)loop over the rows of the matrix
    for iter1 in prange(dimY):
        # 4) get the indexes to access the array of C unique entries
        # 5) assembly a matrix row and execute the dot product
        # 6) execute the dot product
        res[iter1] = np.dot(C_toeplitz_coe[np.abs(jX - jY[iter1]) + np.abs(iX - iY[iter1])], uk)
    return res
#
# @njit(parallel=True, fastmath=True, nogil=True)  # <------parallel compilation
# def matvec_fast(uk, elemX, elemY, dimY, nx, C_toeplitz_coe, C_precision):
#     # uk (numpy array), vector to which multiply the matrix C
#     # nx (int), n. of element in x direction in the cartesian mesh
#     # elemX (numpy array), IDs of elements to consider on x axis of the mesh
#     # elemY (numpy array), IDs of elements to consider on y axis of the mesh
#     # dimY (int), length(elemY) = length(elemX)
#     # C_toeplitz_coe (numpy array), array containing the N unique coefficients of the matrix C of size NxN
#     # C_precision (e.g.: float)
#
#     # 1) vector where to store the result of the dot product
#     res = np.zeros(dimY, dtype=C_precision)
#
#     # 2) some indexes to build the row of a submatrix of C from the array of its unique entries
#     iY = np.floor_divide(elemY, nx)
#     jY = elemY - nx * iY
#     iX = np.floor_divide(elemX, nx)
#     jX = elemX - nx * iX
#
#     iX *= nx
#     iY *= nx
#
#     chunksize = 300
#     splitrange = np.floor_divide(dimY, chunksize)
#     residualrange = dimY - chunksize
#     for iter1 in prange(splitrange):
#         res[iter1] += np.dot(C_toeplitz_coe[np.abs(jX - jY[iter1]) + np.abs(iX - iY[iter1])], uk)
#     return res


@njit(fastmath=True, nogil=True, parallel=True)
def getFast(elemX, elemY, nx, C_toeplitz_coe, C_precision):
    dimX = elemX.size  # number of elements to consider on x axis
    dimY = elemY.size  # number of elements to consider on y axis

    if dimX == 0 or dimY == 0:
        return np.empty((dimY, dimX), dtype=C_precision)
    else:
        C_sub = np.empty((dimY, dimX), dtype=C_precision)  # submatrix of C
        # localC_toeplotz_coe = np.copy(C_toeplitz_coe)  # local access is faster
        localC_toeplotz_coe = C_toeplitz_coe
        if dimX != dimY:
            iY = np.floor_divide(elemY, nx)
            jY = elemY - nx * iY
            iX = np.floor_divide(elemX, nx)
            jX = elemX - nx * iX
            for iter1 in prange(dimY):
                i1 = iY[iter1]
                j1 = jY[iter1]
                C_sub[iter1, 0:dimX] = localC_toeplotz_coe[np.abs(j1 - jX) + nx * np.abs(i1 - iX)]
            return C_sub

        elif dimX == dimY and np.all((elemY == elemX)):
            i = np.floor_divide(elemX, nx)
            j = elemX - nx * i

            for iter1 in prange(dimX):
                i1 = i[iter1]
                j1 = j[iter1]
                C_sub[iter1, 0:dimX] = localC_toeplotz_coe[np.abs(j - j1) + nx * np.abs(i - i1)]
            return C_sub

        else:
            iY = np.floor_divide(elemY, nx)
            jY = elemY - nx * iY
            iX = np.floor_divide(elemX, nx)
            jX = elemX - nx * iX

            for iter1 in prange(dimY):
                i1 = iY[iter1]
                j1 = jY[iter1]
                C_sub[iter1, 0:dimX] = localC_toeplotz_coe[np.abs(j1 - jX) + nx * np.abs(i1 - iX)]
            return C_sub


@njit(fastmath=True, nogil=True, parallel=True)
def getFast_bandedC(coeff9stencilC, elmts, nx, dtype=np.float64):
    # coeff9stencilC contains [C_0dx_0dy, C_1dx_0dy, C_0dx_1dy, C_1dx_1dy]
    i = np.floor_divide(elmts, nx)
    j = elmts - nx * i
    dimX = len(elmts)

    data = dimX * [coeff9stencilC[0]]
    rows = [ii for ii in range(dimX)]
    cols = [ii for ii in range(dimX)]
    for iter1 in prange(dimX):
        i1 = i[iter1]
        j1 = j[iter1]
        delta_j = np.abs(j - j1)
        delta_i = np.abs(i - i1)
        coeff9stencilC_array_coord = delta_j ** 3 + 2 * delta_i ** 3
        coeff9stencilC_array_coord_bool = coeff9stencilC_array_coord < 4
        for iter2 in range(iter1 + 1, dimX):
            if coeff9stencilC_array_coord_bool[iter2]:
                cols.append(iter2)
                rows.append(iter1)
                data.append(coeff9stencilC[coeff9stencilC_array_coord[iter2]])
                cols.append(iter1)
                rows.append(iter2)
                data.append(coeff9stencilC[coeff9stencilC_array_coord[iter2]])

    return coo_matrix((data, (rows, cols)), shape=(dimX, dimX), dtype=dtype).tocsc()


@njit(fastmath=True, nogil=True) # <-- here parallel can not be set to True because currently appending to list is not threadsafe
def getFast_sparseC(C_toeplitz_coe, C_toeplitz_coe_decay, elmts, nx, decay_tshold=0.9, probability=0.05):
    i = np.floor_divide(elmts, nx)
    j = elmts - nx * i
    dimX = len(elmts)
    self_c = C_toeplitz_coe[0]
    #myR = range(dimX)
    data = List()
    rows = List()
    cols = List()
    i *= nx
    for iter1 in prange(dimX):
        index = np.abs(j - j[iter1]) + np.abs(i - i[iter1])
        # self effect
        data.append(self_c)
        rows.append(iter1)
        cols.append(iter1)
        for iter2 in range(iter1 + 1, dimX):
            ii2 = index[iter2]
            if C_toeplitz_coe_decay[ii2] > decay_tshold:# and random.random() < probability:
            #if C_toeplitz_coe_decay[ii2] > decay_tshold:
                cols.append(iter2)
                rows.append(iter1)
                data.append(C_toeplitz_coe[ii2])
                # symmetry
                rows.append(iter2)
                cols.append(iter1)
                data.append(C_toeplitz_coe[ii2])


    # import matplotlib
    # matplotlib.pyplot.spy(coo_matrix((data, (rows, cols)), shape=(dimX, dimX), dtype=dtype))

    # fill ratio:
    # print('fill ratio ' + str(100*len(data)/(dimX*dimX)))
    return data, rows, cols, dimX


class elasticity_matrix_toepliz(LinearOperator):
    """
    This class implements the isotropic elasticity matrix.
    It takes advantage of if toeplitz structure due to the cartesian mesh that it refers to.

    Quick features description:
        - get item via "[]"
        - rebuild the matrix on a different mesh via "reload"
        - set domain indexes via "_set_domain_IDX"
        - set codomain indexes via "_set_codomain_IDX"
        - set tip correction for the diagonal values of the partially filled elements
        - matrix vector multiplication (_matvec_fast or _matvec) both are parallelized.
            i) "_matvec_fast" directly returns the dot product between the matrix and a vector.
               one can change the domanin and codomain
            ii) "_matvec" checks if the user asked for:
                o tip correction
                o preconditioner
                if not the behaviour is same as "_matvec_fast"

    """

    def __init__(self, Mesh, mat_prop, elas_prop_HMAT, C_precision=np.float64, useHMATdot=False, kerneltype = 'Isotropic'):
        """
            Arguments:
                Mesh:                           -- Cartesian Mesh object
                    hx (float):                 -- x size of a mesh cell
                    hy (float):                 -- y size of a mesh cell
                    nx (float):                 -- num. of elements in x dir
                    ny (float):                 -- num. of elements in y dir
                Ep (float):                     -- plain strain modulus.
                C_precision (type):             -- accuracy of the entries e.g.: np.float64
        """
        self.C_precision = C_precision
        self.useHMATdot = useHMATdot
        if useHMATdot:
            self.updateHMATuponRemeshing = True
        else:
            self.updateHMATuponRemeshing = False
        self.mat_prop = mat_prop
        self.kerneltype = "Isotropic"

        self.elas_prop_HMAT = elas_prop_HMAT
        self.reload(Mesh)

        # ---- TIP CORRECTION ----
        self.enable_tip_corr = False  # one needs to specifically activate it in case it is needed
        self.tipcorrINDX = None  # list of tip elem. IDs
        self.tipcorr = None

        # ---- JACOBI PREC ----
        self.left_precJ = False
        self.right_precJ = False

    def reload(self, Mesh, len_eltcrack=0.):
        hx = Mesh.hx; hy = Mesh.hy
        nx = Mesh.nx; ny = Mesh.ny
        Lx = Mesh.Lx; Ly = Mesh.Ly
        self.nx = nx
        self.C_toeplitz_coe = self.reload_toepliz_Coe(Lx, Ly, nx, ny, hx, hy, self.mat_prop)
        #time_HMAT_build = -time.time()
        self.reload_HMAT_Coe(Mesh, self_eff = self.C_toeplitz_coe[0])
        #time_HMAT_build = time_HMAT_build + time.time()
        # file_name = '/home/peruzzo/Desktop/test_EHL_direct_vs_iter/iterT_recomputingHMAT.csv'
        # append_new_line(file_name,
        #                 str(time_HMAT_build) + ','
        #                 + str(len_eltcrack) + ','
        #                 + str(Mesh.NumberOfElts))
        self.reload_dot(Mesh)

    def reload_HMAT_Coe(self, Mesh, self_eff = None):
        #################### HMAT dot SECTION ###################
        if self.updateHMATuponRemeshing:
            self.max_leaf_size = 100
            self.eta = 5
            self.eps_aca = 1.e-6
            self.HMATtract = None

            data = [self.max_leaf_size, self.eta, self.eps_aca,
                    self.elas_prop_HMAT, Mesh.VertexCoor, Mesh.Connectivity, Mesh.hx, Mesh.hy, self_eff]

            if self.kerneltype == "Isotropic":
                self.HMAT = Hdot_3DR0opening()
            else:
                SystemExit("HMAT not implemented for kerneltype different from Isotropic")

            # HMATcreationTime = -time.time()
            self.HMAT.set(data)
            # self.HMATcreationTime.append(HMATcreationTime + time.time())
            # self._get_full_blocks(Mesh.VertexCoor, Mesh.Connectivity, elas_prop)
        ################ END HMAT dot SECTION ######################

    def reload_toepliz_Coe(self, Lx, Ly, nx, ny, hx, hy, mat_prop):
        raise NotImplementedError()

    def reload_dot(self, Mesh):
        #################### TOEPLITZ Cdot SECTION ###################
        nx = Mesh.nx

        # to build a fast precoditioner for iterative solver see (Peirce A., 2015)
        C_toeplitz_coe_exp = np.log(np.abs(self.C_toeplitz_coe))
        C_toeplitz_coe_exp = C_toeplitz_coe_exp - C_toeplitz_coe_exp[-1]
        C_toeplitz_coe_exp = C_toeplitz_coe_exp / C_toeplitz_coe_exp[0]
        self.C_toeplitz_coe_decay = List(C_toeplitz_coe_exp.tolist())  # between 0 and 1
        self.coeff9stencilC = [self.C_toeplitz_coe[0],  # 0 dx 0 dy
                               self.C_toeplitz_coe[1],  # 1 dx 0 dy
                               self.C_toeplitz_coe[nx],  # 0 dx 1 dy
                               self.C_toeplitz_coe[nx + 1]  # 1 dx 1 dy
                               ]

        # diagonal value of the matrix
        self.diag_val = self.C_toeplitz_coe[0]

        # define the size = number of elements in the mesh
        self.C_size_ = int(Mesh.nx * Mesh.ny)

        # define the total number of unknowns to be output by the matvet method
        self.matvec_size_ = self.C_size_

        # it is mandatory to define shape and dtype of the dot product
        self.dtype_ = self.C_precision
        self.shape_ = (self.matvec_size_, self.matvec_size_)
        super().__init__(self.dtype_, self.shape_)

        self._set_domain_IDX(np.arange(self.C_size_))
        self._set_codomain_IDX(np.arange(self.C_size_))
        ################ END TOEPLITZ Cdot SECTION ######################


    def __getitem__(self, elementsXY):
        """
        critical call: it should be as fast as possible
        :param elemX: (numpy array) columns to take
        :param elemY: (numpy array) rows to take
        :return: submatrix of C
        """

        elemX = elementsXY[1].flatten()
        elemY = elementsXY[0].flatten()
        return getFast(elemX, elemY, self.nx, self.C_toeplitz_coe, self.C_precision)


    def _matvec_fast(self, uk):
        if self.useHMATdot and len(uk) > 10000:
            return self.HMAT._matvec(uk)
        else:
            #mv_time = - time.time()
            aa= matvec_fast(np.float64(uk), self.domain_INDX, self.codomain_INDX, self.codomain_INDX.size, self.nx,
                               self.C_toeplitz_coe, self.C_precision)
            #mv_time = mv_time + time.time()

            #file_name = '/home/carlo/Desktop/test_EHL_direct_vs_iter/Ex_time.csv'
            #append_new_line(file_name,str(len(aa)) + ',' + str(mv_time))
            return aa


    def _matvec(self, uk):
        # if self.C_precision == np.float32:
        #     uk = uk.astype('float32')
        if not self.enable_tip_corr and not self.left_precJ and not self.right_precJ:
            return matvec_fast(uk, self.domain_INDX, self.codomain_INDX, self.codomain_INDX.size, self.nx,
                               self.C_toeplitz_coe, self.C_precision)
        elif self.enable_tip_corr:
            # find the positions of the tip elements in the codomain
            tip_codomain, INDX_codomain, waste = np.intersect1d(self.codomain_INDX, self.tipcorrINDX,
                                                                assume_unique=True, return_indices=True)
            if self.right_precJ:
                uk = self._precJm1dotvec(uk, INDX_codomain=INDX_codomain)
                res = matvec_fast(uk, self.domain_INDX, self.codomain_INDX, self.codomain_INDX.size, self.nx,
                                  self.C_toeplitz_coe, self.C_precision)
            else:
                # the tip correction acts only on the diagonal elements, so:
                # (A+tipcorr*I)*uk = A*uk+tipcorr*I*uk
                # compute A*uk
                res = matvec_fast(uk, self.domain_INDX, self.codomain_INDX, self.codomain_INDX.size, self.nx,
                                  self.C_toeplitz_coe, self.C_precision)

            if self.left_precJ:
                # assuming that we are solving the problem for the whole fracture and that we are considering all the tip elem
                # (A+tipcorr*I)*uk
                res[INDX_codomain] = res[INDX_codomain] + uk[INDX_codomain] * self.tipcorr[self.tipcorrINDX]
                res = self._precJm1dotvec(res, INDX_codomain=INDX_codomain)
            elif self.right_precJ:
                # assuming that we are solving the problem for the whole fracture and that we are considering all the tip elem
                # (A+tipcorr*I)*res
                res[INDX_codomain] = res[INDX_codomain] + uk[INDX_codomain] * self.tipcorr[self.tipcorrINDX]
            else:
                # compute tipcorr * I * uk
                # find the tip elements in common between domain and codomain
                tip_domain_and_codomain = np.intersect1d(self.domain_INDX, tip_codomain, assume_unique=True)

                # find the positions of the common tip elements between codomain and domain, in the codomain
                waste, partial_INDX_codomain, waste = np.intersect1d(tip_codomain, tip_domain_and_codomain,
                                                                     assume_unique=True, return_indices=True)
                INDX_codomain = INDX_codomain[partial_INDX_codomain]
                if len(INDX_codomain) > 0:
                    res[INDX_codomain] = res[INDX_codomain] + uk[INDX_codomain] * self.tipcorr[tip_domain_and_codomain]
            return res

        else:
            raise SystemExit('preconditioner without  tip correction is not currently implemented')


    def _precJm1dotvec(self, uk, INDX_codomain=None):
        # preconditioning:
        # P~A, P=A_ii
        # P^-1 * (A+tipcorr*I)*uk = P^-1 * res, P^-1 is a diag matrix =>  P^-1 * res == P^-1_ii * res_i
        #
        # the precontioner accounts for the correction of the tip due to partially filled elements

        Pm1_vec = np.full(len(uk), 1. / self.diag_val)
        if INDX_codomain is None:
            # find the positions of the tip elements in the codomain
            waste, INDX_codomain, waste = np.intersect1d(self.codomain_INDX, self.tipcorrINDX, assume_unique=True,
                                                         return_indices=True)
        Pm1_vec[INDX_codomain] = 1. / (self.diag_val + self.tipcorr[self.tipcorrINDX])
        return Pm1_vec * uk


    def _precJdotvec(self, uk, INDX_codomain=None):
        # preconditioning:
        # P~A, P=A_ii
        # the precontioner accounts for the correction of the tip due to partially filled elements
        #
        # returning P*vec

        Pm1_vec = np.full(len(uk), self.diag_val)
        if INDX_codomain is None:
            # find the positions of the tip elements in the codomain
            waste, INDX_codomain, waste = np.intersect1d(self.codomain_INDX, self.tipcorrINDX, assume_unique=True,
                                                         return_indices=True)
        Pm1_vec[INDX_codomain] = (self.diag_val + self.tipcorr[self.tipcorrINDX])
        return Pm1_vec * uk


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
        if self.useHMATdot:
            return self.HMAT._set_domain_IDX(domainIDX)


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
        if self.useHMATdot:
            return self.HMAT._set_codomain_IDX(codomainIDX)


    def _set_tipcorr(self, correction_val, correction_INDX):
        """
        :param correction_val: (array) contains the factors to be applied to each diagonal val. of the specified indexes based on the filling fraction of the cell
        :param correction_INDX: (array) specified indexes where the correction_val should apply
        :return:
        """

        self.enable_tip_corr = True
        self.tipcorrINDX = correction_INDX  # list of tip elem. IDs
        self.tipcorr = np.full(self.C_size_, np.nan)
        self.tipcorr[correction_INDX] = correction_val * self.diag_val


    def _changeShape(self, shape_):
        self.matvec_size_ = shape_
        self.shape_ = (shape_, shape_)
        super().__init__(self.dtype_, self.shape_)


    def _get9stencilC(self, elmts, decay_tshold=0.9, probability=0.15):
        data, rows, cols, dimX = getFast_sparseC(self.C_toeplitz_coe, self.C_toeplitz_coe_decay,
                                                 elmts, self.nx,
                                                 decay_tshold=decay_tshold, probability=probability)
        return coo_matrix((data, (rows, cols)), shape=(dimX, dimX), dtype=self.C_precision).tocsc()

    # def _get_full_blocks(self, VertexCoor, Connectivity, elas_prop):
    #     from pypart import Bigwhamio
    #     from pypart import pyGetFullBlocks
    #     from Hdot import applyPermutation
    #     self.HMATtract.set(VertexCoor,
    #                           Connectivity,
    #                           self.tractionKernel,
    #                           elas_prop,
    #                           self.max_leaf_size,
    #                           self.eta,
    #                           self.eps_aca)
    #     # ---> the memory here consist mainly of the Hmat
    #     myget = pyGetFullBlocks()
    #     myget.set(self.HMATtract)
    #     # ---> the memory here consist at most of 3 * Hmat
    #     col_ind_tract = myget.getColumnN()
    #     row_ind_tract = myget.getRowN()
    #     # here we need to permute the rows and columns
    #     # ---> the memory here consist at most of 3 * Hmat
    #     [row_ind_tract, col_ind_tract] = applyPermutation(self.HMATtract, row_ind_tract, col_ind_tract )
    #     # ---> the memory here consist at most of 5 * Hmat for a while and it goes back to 4 Hmat
    #     values_tract = myget.getValList()
    #     del myget
# -----------------------------------------------------------------------------------------------------------------------