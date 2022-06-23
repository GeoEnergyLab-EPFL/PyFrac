# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo on Tue Nov 2 15:09:38 2021.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""
# external imports
import numpy as np
from numba import njit, prange, boolean, uint16, uint64, int64
import numba as nb
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
from solid.elasticity_tip_correction import tip_correction_factors
from utilities.utility import append_new_line


# set the threading layer before any parallel target compilation
# 'workqueue' is builtin
# config.THREADING_LAYER = 'workqueue' #'workqueue' , 'threadsafe' ,'tbb', 'omp'
config.THREADING_LAYER = 'omp' #'tbb'   'workqueue', 'threadsafe' ,'tbb'


@njit(parallel=True, cache = True, nogil=True, fastmath=True)  # <------parallel compilation
def matvec_fast(uk, elemX, elemY, dimY, nx, C_toeplitz_coe, C_precision):
    # uk (numpy array), vector to which multiply the matrix C
    # nx (int), n. of element in x direction in the cartesian mesh
    # elemX (numpy array), IDs of elements to consider on x axis of the mesh
    # elemY (numpy array), IDs of elements to consider on y axis of the mesh
    # dimY (int), length(elemY) != length(elemX)
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
# @njit(parallel=True, cache = True, nogil=True, fastmath=True)  # <------parallel compilation
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


@njit(parallel=True, cache = True, nogil=True, fastmath=True)
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

@njit(fastmath=True, nogil=True, cache=True)
def _concat_equal1(array1, array2, out):
    N = len(array1)
    for i in range(N):
        out[i] = array1[i]

    for i in range(N):
        out[N+i] = array2[i]
    return out

@njit(fastmath=True, nogil=True, cache=True)
def concat_equal1(array1, array2):
    out = np.empty(shape=(2 * len(array1)))
    return _concat_equal1(array1, array2, out)

@njit(cache = True, nogil=True, fastmath=True) # <-- here parallel can not be set to True because currently appending to list is not threadsafe
def getSuperFast_sparseC_smooth(C_toeplitz_coe, C_toeplitz_coe_decay, elmts, nx, decay_tshold=0.9, probability=0.05):
    # this is not faster than its non smooth version but it provides a better approximation for the preconditioner
    # this has been disable in order to achieve better performances for 10^6 elements
    i = np.floor_divide(elmts, nx)
    j = elmts - nx * i
    dimX = len(elmts)
    self_c = 0.5 * C_toeplitz_coe[0]
    # self_c = C_toeplitz_coe[0]
    #myR = range(dimX)
    data = List()
    rows = List()
    cols = List()
    i *= nx
    for iter1 in prange(dimX):
        #index = np.abs(j - j[iter1]) + np.abs(i - i[iter1])
        index = np.abs(j[:iter1] - j[iter1]) + np.abs(i[:iter1] - i[iter1])
        # self effect
        data.append(self_c)
        rows.append(iter1)
        cols.append(iter1)
        #for iter2 in range(iter1 + 1, dimX):
        for iter2 in range(iter1):
            ii2 = index[iter2]
            if C_toeplitz_coe_decay[ii2] > decay_tshold:# and random.random() < probability:
            #if C_toeplitz_coe_decay[ii2] > decay_tshold:
                cols.append(iter2)
                rows.append(iter1)
                data.append(C_toeplitz_coe[ii2])
                # symmetry
                # rows.append(iter2)
                # cols.append(iter1)
                # data.append(C_toeplitz_coe[ii2])
    # symmetry
    rows_new = concat_equal1(rows,cols)
    cols = concat_equal1(cols,rows)
    data = concat_equal1(data,data)

    # import matplotlib
    # matplotlib.pyplot.spy(coo_matrix((data, (rows, cols)), shape=(dimX, dimX), dtype=dtype))

    # fill ratio:
    # print('fill ratio ' + str(100*len(data)/(dimX*dimX)))
    return data, rows_new, cols, dimX
    #return data, rows, cols, dimX

@njit(cache = True, nogil=True, fastmath=True) # <-- here parallel can not be set to True because currently appending to list is not threadsafe
def getSuperFast_sparseC(coeff9stencilC, elmts, nx):
    i = np.floor_divide(elmts, nx)
    j = elmts - nx * i
    dimX = len(elmts)
    self_c = 0.5 * coeff9stencilC[4]

    data = List()
    rows = List()
    cols = List()

    for iter1 in prange(dimX):
        dj = (j[:iter1] - j[iter1])
        di = (i[:iter1] - i[iter1])
        # self effect
        data.append(self_c)
        rows.append(iter1)
        cols.append(iter1)
        for iter2 in range(iter1):
            ii2_a = dj[iter2]
            ii2_b = di[iter2]
            if ii2_a < 2 and ii2_b < 2 and ii2_a > -2 and ii2_b > -2 :
                cols.append(iter2)
                rows.append(iter1)
                data.append(coeff9stencilC[([[0, 1, 2], [3, 4, 5], [6, 7, 8]])[ii2_a + 1][ii2_b + 1]])

    # symmetry
    rows_new = concat_equal1(rows,cols)
    cols = concat_equal1(cols,rows)
    data = concat_equal1(data,data)

    # import matplotlib
    # matplotlib.pyplot.spy(coo_matrix((data, (rows, cols)), shape=(dimX, dimX), dtype=dtype))

    # fill ratio:
    # print('fill ratio ' + str(100*len(data)/(dimX*dimX)))
    return data, rows_new, cols, dimX
    #return data, rows, cols, dimX

#@njit(parallel=True, cache = True, nogil=True, fastmath=True)
def getFast_sparseC(coeff9stencilC, elmts, nx):
    """
    Relative position and numbering:
    o---o---o---o
    | 0 | 1 | 2 |
    o---o---o---o
    | 3 | 4 | 5 |
    o---o---o---o
    | 6 | 7 | 8 |
    o---o---o---o
                            0,   1,   2,   3,   4,   5,   6,  7,  8
    coeff9stencilC =     [diag, dy, diag, dx, self, dx, diag, dy, diag ]
    o---o---o---o
    | 3 | 2 | 3 |
    o---o---o---o
    | 1 | 0 | 1 |
    o---o---o---o
    | 3 | 2 | 3 |
    o---o---o---o
    """
    dimX = len(elmts)
    interactions_BOOL = np.full((dimX, 9), False) #, dtype=boolean)
    interactions_PAIRS = np.zeros((dimX, 9))


    i = np.floor_divide(elmts, nx)
    j = elmts - nx * i

    for ii in prange(0, int(dimX*(dimX+1)/2)):
        position_table = np.asarray([[0, 1, 2], [3, 4, 5], [6, 7, 8]])#,dtype=uint16)
        source = int(np.floor_divide((-1 + np.sqrt(1 + 8 * ii)),2))
        receiver = int(ii - source*(source+1)/2)
        #source_ID = elmts[source]
        #receiver_ID = elmts[receiver]
        dip1 = i[source] - i[receiver] + 1
        djp1 = j[source] - j[receiver] + 1

        if dip1 in range(3):
            if djp1 in range(3):
                interactions_BOOL[source,position_table[dip1, djp1]] = True
                interactions_PAIRS[source, position_table[dip1, djp1]] = receiver


    size_of_output_symm = np.sum(interactions_BOOL)
    size_of_output_NoSelfEff = 2 * (size_of_output_symm - dimX)
    size_of_output = size_of_output_NoSelfEff + dimX
    data = np.zeros(size_of_output, dtype=np.float64)
    rows = np.zeros(size_of_output)
    cols = np.zeros(size_of_output)


    index_results = np.zeros(dimX, dtype=int) #, dtype=int64)
    cumulative_index = 0
    for ii in range(dimX):
        cumulative_index = int(cumulative_index + 2 * (np.sum(interactions_BOOL[ii, :]) - 1))
        index_results[ii] = int(cumulative_index)

    for ii in prange(dimX):
        starting_index = index_results[ii]
        for jj in [0,1,2,3,5,6,7,8]: # missing 4 because it is the self effect
            if interactions_BOOL[ii,jj]:
                data[starting_index] = coeff9stencilC[jj]
                rows[starting_index] = interactions_PAIRS[ii,jj]
                cols[starting_index] = ii
                starting_index = starting_index + 1
                data[starting_index] = coeff9stencilC[jj]
                rows[starting_index] = ii
                cols[starting_index] = interactions_PAIRS[ii,jj]
                starting_index = starting_index + 1

    for ii in prange(dimX):
        # writing the self effect
        index =  size_of_output_NoSelfEff + ii
        data[index] = coeff9stencilC[4]
        rows[index] = ii
        cols[index] = ii

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

    def __init__(self, Mesh, mat_prop, elas_prop_HMAT, C_precision=np.float64, useHMATdot=False,
                 kerneltype = 'Isotropic_R0',
                 HMATparam = None,
                 f_matvec_fast = matvec_fast,
                 f_getFast = getFast,
                 f_getFast_sparseC = getSuperFast_sparseC #getFast_sparseC
                 ):
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
        # this is because R4 requires different functions whereas R0 and TI does not
        self.f_matvec_fast = f_matvec_fast
        self.f_getFast = f_getFast
        self.f_getFast_sparseC = f_getFast_sparseC

        self.C_precision = C_precision
        self.useHMATdot = useHMATdot
        self.HMAT_tshold = 10000 # size of the dot product above which to use the HMAT
        if useHMATdot:
            self.updateHMATuponRemeshing = True
        else:
            self.updateHMATuponRemeshing = False
        self.mat_prop = mat_prop
        self.kerneltype = kerneltype

        self.elas_prop_HMAT = elas_prop_HMAT
        self.HMATparam = HMATparam

        # ---- TIP CORRECTION ----
        self.enable_tip_corr = False  # one needs to specifically activate it in case it is needed
        self.EltTip = None  # list of tip elem. IDs
        self.tipcorr = None # vector of size & ordering as self.EltTip with the correction factors
        self.tipINDX_codomain = None # list of indexes of tip elem. in the codomain
        self.tip_in_codomain = None # list of tip elem. IDs in the codomain

        # ---- LOAD THE OBJ ----
        self.reload(Mesh)

        # ---- JACOBI PREC ----
        self.left_precJ = False
        self.right_precJ = False

    def reload(self, Mesh, len_eltcrack=0.):
        hx = Mesh.hx; hy = Mesh.hy; self.hx = hx; self.hy = hy
        nx = Mesh.nx; ny = Mesh.ny; self.nx = nx; self.ny = ny
        Lx = Mesh.Lx; Ly = Mesh.Ly; self.Lx = Lx; self.Ly = Ly

        self.diag_val, self.C_toeplitz_coe = self.reload_toepliz_Coe(Lx, Ly, nx, ny, hx, hy, self.mat_prop)
        #time_HMAT_build = -time.time()
        self.reload_HMAT_Coe(Mesh, self_eff = self.diag_val)
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
            if self.HMATparam is None:
                self.max_leaf_size = 750
                self.eta = 8
                self.eps_aca = 1.e-4
                self.HMATtract = None
            else:
                self.max_leaf_size = self.HMATparam[0]
                self.eta = self.HMATparam[1]
                self.eps_aca = self.HMATparam[2]
                self.HMATtract = None


            data = [self.max_leaf_size, self.eta, self.eps_aca,
                    self.elas_prop_HMAT, Mesh.VertexCoor, Mesh.Connectivity, Mesh.hx, Mesh.hy, self_eff]

            if self.kerneltype == "Isotropic_R0":
                self.HMAT = Hdot_3DR0opening()
            else:
                SystemExit("HMAT not implemented for kerneltype different from Isotropic_R0")

            # HMATcreationTime = -time.time()
            hmattime = -time.time()
            self.HMAT.set(data)
            hmattime = hmattime + time.time()
            print(f'TOTAL TIME HMAT minutes : {hmattime/60. : 0.2f} ')
            # self.HMATcreationTime.append(HMATcreationTime + time.time())
            # self._get_full_blocks(Mesh.VertexCoor, Mesh.Connectivity, elas_prop)
        ################ END HMAT dot SECTION ######################

    def reload_toepliz_Coe(self, Lx, Ly, nx, ny, hx, hy, mat_prop):
        raise NotImplementedError()

    def get_normal_traction_at(self,  xy_obs, xy_crack):
        raise NotImplementedError()

    def reload_dot(self, Mesh):
        #################### TOEPLITZ Cdot SECTION ###################
        nx = Mesh.nx

        # to build a fast precoditioner for iterative solver see (Peirce A., 2015)
        if len(self.C_toeplitz_coe.shape) == 1:
            C_toeplitz_coe_exp = np.log(np.abs(self.C_toeplitz_coe))
            C_toeplitz_coe_exp = C_toeplitz_coe_exp - C_toeplitz_coe_exp[-1]
            C_toeplitz_coe_exp = C_toeplitz_coe_exp / C_toeplitz_coe_exp[0]
            self.C_toeplitz_coe_decay = List(C_toeplitz_coe_exp.tolist())  # between 0 and 1
        else:
            C_toeplitz_coe_exp = np.log(np.abs(self.C_toeplitz_coe[0,:]))
            C_toeplitz_coe_exp = C_toeplitz_coe_exp - C_toeplitz_coe_exp[-1]
            C_toeplitz_coe_exp = C_toeplitz_coe_exp / C_toeplitz_coe_exp[0]
            self.C_toeplitz_coe_decay = List(C_toeplitz_coe_exp.tolist())  # between 0 and 1


        #coeff9stencilC = [diag, dy, diag, dx, self, dx, diag, dy, diag]
        self.coeff9stencilC = np.asarray([self.C_toeplitz_coe[nx + 1],  # 1 dx 1 dy - diag
                               self.C_toeplitz_coe[nx],      # 0 dx 1 dy - dy
                               self.C_toeplitz_coe[nx + 1],  # 1 dx 1 dy - diag
                               self.C_toeplitz_coe[1],       # 1 dx 0 dy - dx
                               self.C_toeplitz_coe[0],       # 0 dx 0 dy - self
                               self.C_toeplitz_coe[1],       # 1 dx 0 dy - dx
                               self.C_toeplitz_coe[nx + 1],  # 1 dx 1 dy - diag
                               self.C_toeplitz_coe[nx],      # 0 dx 1 dy - dy
                               self.C_toeplitz_coe[nx + 1]   # 1 dx 1 dy - diag
                               ])

        # define the size = number of elements in the mesh
        self.C_size_ = int(Mesh.nx * Mesh.ny)

        # define the total number of unknowns to be output by the matvet method
        self.matvec_size_ = self.C_size_

        # it is mandatory to define shape and dtype of the dot product
        self.dtype_ = self.C_precision
        self.shape_ = (self.matvec_size_, self.matvec_size_)
        super().__init__(self.dtype_, self.shape_)

        self._set_domain_and_codomain_IDX(np.arange(self.C_size_), np.arange(self.C_size_), same_domain_and_codomain= True)
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
        return self.f_getFast(elemX, elemY, self.nx, self.C_toeplitz_coe, self.C_precision)


    def _matvec_fast(self, uk):
        if self.useHMATdot and len(uk) > self.HMAT_tshold:
            return self.HMAT._matvec(uk)
        else:
            #mv_time = - time.time()
            aa= self.f_matvec_fast(np.float64(uk), self.domain_INDX, self.codomain_INDX, self.codomain_INDX.size, self.nx,
                               self.C_toeplitz_coe, self.C_precision)
            #mv_time = mv_time + time.time()

            #file_name = '/home/carlo/Desktop/test_EHL_direct_vs_iter/Ex_time.csv'
            #append_new_line(file_name,str(len(aa)) + ',' + str(mv_time))
            return aa


    def _matvec(self, uk):
        if not self.enable_tip_corr and not self.left_precJ and not self.right_precJ:
            if self.useHMATdot and len(uk) > self.HMAT_tshold:
                return self.HMAT._matvec(uk)
            else:
                return self.f_matvec_fast(np.float64(uk),
                                          self.domain_INDX,
                                          self.codomain_INDX,
                                          self.codomain_INDX.size,
                                          self.nx,
                                          self.C_toeplitz_coe,
                                          self.C_precision)

        elif self.enable_tip_corr:
            if self.right_precJ:
                uk = self._precJm1dotvec(uk, tipINDX_codomain=self.tipINDX_codomain)

            # the tip correction acts only on the diagonal elements, so:
            # (A+tipcorr*I)*uk = A*uk+tipcorr*I*uk
            # compute A*uk
            if self.useHMATdot and len(uk) > self.HMAT_tshold:
                res = self.HMAT._matvec(uk)
            else:
                res = self.f_matvec_fast(np.float64(uk),
                                          self.domain_INDX,
                                          self.codomain_INDX,
                                          self.codomain_INDX.size,
                                          self.nx,
                                          self.C_toeplitz_coe,
                                          self.C_precision)

            if self.left_precJ:
                # TIPCORRECTION & LEFT PRECONDITIONER
                # (A+tipcorr*I)*uk
                res[self.tipINDX_codomain] = res[self.tipINDX_codomain] + uk[self.tipINDX_codomain] * self.tipcorr[self.tip_in_codomain]
                res = self._precJm1dotvec(res, tipINDX_codomain = self.tipINDX_codomain)
            elif self.right_precJ:
                # TIPCORRECTION & RIGHT PRECONDITIONER
                # (A+tipcorr*I)*res
                res[self.tipINDX_codomain] = res[self.tipINDX_codomain] + uk[self.tipINDX_codomain] * self.tipcorr[self.tip_in_codomain]
            else:
                # ONLY TIPCORRECTION - NO PRECONDITIONER
                if len(self.tipINDX_codomain_and_domain) > 0:
                    res[self.tipINDX_codomain_and_domain] = res[self.tipINDX_codomain_and_domain] + uk[self.tipINDX_codomain_and_domain] * self.tipcorr[self.tip_in_domain_and_codomain]
            return res
        else:
            raise SystemExit('elasticity matvec: case not currently implemented')


    def _precJm1dotvec(self, uk, tipINDX_codomain=None):
        # preconditioning:
        # P~A, P=A_ii
        # P^-1 * (A+tipcorr*I)*uk = P^-1 * res, P^-1 is a diag matrix =>  P^-1 * res == P^-1_ii * res_i
        #
        # the precontioner accounts for the correction of the tip due to partially filled elements

        Pm1_vec = np.full(len(uk), 1. / self.diag_val)
        if tipINDX_codomain is None:
            # find the positions of the tip elements in the codomain
            waste, tipINDX_codomain, waste = np.intersect1d(self.codomain_INDX, self.EltTip, assume_unique=True, return_indices=True)
        Pm1_vec[tipINDX_codomain] = 1. / (self.diag_val + self.tipcorr[self.EltTip])
        return Pm1_vec * uk


    def _precJdotvec(self, uk, tipINDX_codomain=None):
        # preconditioning:
        # P~A, P=A_ii
        # the precontioner accounts for the correction of the tip due to partially filled elements
        #
        # returning P*vec

        Pm1_vec = np.full(len(uk), self.diag_val)
        if tipINDX_codomain is None:
            # find the positions of the tip elements in the codomain
            waste, tipINDX_codomain, waste = np.intersect1d(self.codomain_INDX, self.EltTip, assume_unique=True,
                                                         return_indices=True)
        Pm1_vec[tipINDX_codomain] = (self.diag_val + self.tipcorr[self.EltTip])
        return Pm1_vec * uk


    def _set_domain_and_codomain_IDX(self, domainIDX, codomainIDX, same_domain_and_codomain = False):
        """
        General example:
        domain indexes are [1 , 2] of NON ZERO elements used to make the dot product
        codomain indexes are [0, 2] of elements returned after the dot product
        o o o o    0 <-0    x <-0
        o o o o    x <-1  = o <-1
        o o o o    x <-2    x <-2
        o o o o    0 <-3    o <-3
        """
        # set domain
        self.domain_INDX = domainIDX

        # set codomain
        self.codomain_INDX = codomainIDX
        self._changeShape(codomainIDX.size)

        if self.enable_tip_corr:
            # find the positions of the tip elements in the codomain
            self.tip_in_codomain, self.tipINDX_codomain, waste = np.intersect1d(self.codomain_INDX, self.EltTip, assume_unique=True, return_indices=True)

            if not same_domain_and_codomain:
                # find the tip elements in common between domain and codomain
                self.tip_in_domain_and_codomain = np.intersect1d(self.domain_INDX, self.tip_in_codomain, assume_unique=True)

                # find the positions of the common tip elements between codomain and domain, in the codomain
                waste, partial_INDX_codomain, waste = np.intersect1d(self.tip_in_codomain, self.tip_in_domain_and_codomain, assume_unique=True, return_indices=True)
                self.tipINDX_codomain_and_domain = self.tipINDX_codomain[partial_INDX_codomain]
            else:
                self.tip_in_domain_and_codomain = self.tip_in_codomain
                self.tipINDX_codomain_and_domain = self.tipINDX_codomain
        else:
            self.tip_in_domain_and_codomain = []
            self.tipINDX_codomain_and_domain = []

        if self.useHMATdot:
            self.HMAT._set_domain_and_codomain_IDX(domainIDX, codomainIDX)


    def _set_tipcorr(self, FillFrac, EltTip):
        """
        Tip correction as Rider & Napier, 1985.

        :param correction_val: (array) contains the factors to be applied to each diagonal val. of the specified indexes based on the filling fraction of the cell
        :param correction_INDX: (array) specified indexes where the correction_val should apply
        :return:
        """
        # ATTENTION!
        # the following flag needs to be se to 0 when one wants to disable the tip correction
        self.enable_tip_corr = True
        # list of tip elem. IDs:
        self.EltTip = EltTip
        # build an array for all the elements in the mesh and fill it with the known values
        self.tipcorr = np.full(self.C_size_, np.nan)
        self.tipcorr[EltTip] = tip_correction_factors(FillFrac) * self.diag_val


    def _changeShape(self, shape_):
        self.matvec_size_ = shape_
        self.shape_ = (shape_, shape_)
        super().__init__(self.dtype_, self.shape_)


    def _get9stencilC(self, elmts, decay_tshold=0.9, probability=0.15):
        # data1, rows1, cols1, dimX1 = getSuperFast_sparseC_smooth(self.C_toeplitz_coe, self.C_toeplitz_coe_decay,
        #                                          elmts, self.nx,
        #                                          decay_tshold=decay_tshold, probability=probability)

        data, rows, cols, dimX = self.f_getFast_sparseC(self.coeff9stencilC,elmts, self.nx)

        # to see the matrix
        # import matplotlib.pylab as plt
        # plt.spy(coo_matrix((data, (rows, cols)), shape=(dimX, dimX), dtype=self.C_precision).tocsc())
        return coo_matrix((data, (rows, cols)), shape=(dimX, dimX), dtype=self.C_precision).tocsc()


    def _set_kerneltype_as_R0(self):
        # this function exist because R0 is performing better in case of injection in the same footprint when tip correction is needed.
        # the counter function is "_set_kerneltype_as_it_used_to_be"
        if self.kerneltype == "Isotropic_R4":
            # set the functions needed for the dot product
            self.f_matvec_fast = matvec_fast
            self.f_getFast = getFast
            #self.f_getFast_sparseC = getFast_sparseC
            self.f_getFast_sparseC = getSuperFast_sparseC

            # set the matrix coeff.
            self.diag_val_pause = self.diag_val
            self.C_toeplitz_coe_pause =  self.C_toeplitz_coe
            self.diag_val = self.C_toeplitz_coe[2,0]
            self.C_toeplitz_coe =  self.C_toeplitz_coe[2,:]


    def _set_kerneltype_as_it_used_to_be(self):
        from solid.elasticity_kernels.isotropic_R4_elem import matvec_fast_R4, getFast_R4, getFast_sparseC_R4
        if self.kerneltype == "Isotropic_R4":
            # set the functions needed for the dot product back
            self.f_matvec_fast = matvec_fast_R4
            self.f_getFast = getFast_R4
            self.f_getFast_sparseC = getFast_sparseC_R4

            # set the matrix coeff. back
            self.diag_val = self.diag_val_pause
            self.C_toeplitz_coe =  self.C_toeplitz_coe_pause

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