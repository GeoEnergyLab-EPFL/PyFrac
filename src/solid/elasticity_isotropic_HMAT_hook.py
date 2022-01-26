# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo on Tue Nov 2 15:09:38 2021.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# External imports
import numpy as np
from scipy.sparse.linalg import LinearOperator

class Hdot_3DR0opening(LinearOperator):
    """
    This function provides the dot product between the Hmatrix approx of the elasticity matrix and a vector

    """
    def __init__(self):
        #from solid.bigwhamPybind import Bigwhamio
        from solid.pypart import Bigwhamio
        self.unknowns_number_ = None
        self.matvec_size_ = None
        self.HMAT_size_ = None
        self.shape_ = None
        self.dtype_ = float
        self.HMATtract = Bigwhamio()
        self.tractionKernel = "3DR0opening"
        self.diag_val = None
        self.domain_INDX = None
        self.codomain_INDX = None
        self.tipcorr = None
        self.tipcorrINDX = None
        self.enable_tip_corr = False
        #------
        self.max_leaf_size = None
        self.eta = None
        self.eps_aca = None

    def set(self, data):
        # properties = [youngs_mod, nu]
        max_leaf_size, eta, eps_aca,  properties, coor2D, conn, hx, hy, self_eff = data

        self.diag_val = self_eff

        # number of vertexes in the mesh
        NoV = coor2D.shape[0]

        # number of elements in the mesh
        self.NoE = conn.shape[0]

        # define the HMAT size
        # define the total number of unknowns to be output by the matvet method
        unknowns_per_element_ = 1
        self.HMAT_size_ = int(self.NoE  * unknowns_per_element_)
        self.matvec_size_ = self.HMAT_size_

        # it is mandatory to define shape and dtype of the dot product
        self.shape_ = (self.matvec_size_, self.matvec_size_)
        super().__init__(self.dtype_, self.shape_)

        # size of the vector containing the coordinates in 3D
        size_coor3D = NoV * 3
        coor3D = np.zeros(size_coor3D)

        # setting the coordinate z to be 0
        for i in range(NoV):
            #put 0 at z
            coor3D[i*3] = coor2D[i,0]
            coor3D[i*3+1] = coor2D[i,1]
            coor3D[i*3+2]= 0.

        # it is mandatory to flatten the array
        coor3D = coor3D.flatten()               # coor : is an array with the coordinates of all the vertexes of the elements of the mesh
        conn3D = conn.flatten()               # conn : is an array with all the connectivity of the mesh

        # save the parameters in case of mesh extension
        self.max_leaf_size = max_leaf_size
        self.eta = eta
        self.eps_aca = eps_aca

        # set the object
        self.HMATtract.set(coor3D,
                              conn3D,
                              self.tractionKernel,
                              properties,
                              max_leaf_size,
                              eta,
                              eps_aca)

        self.compressionratio = self.HMATtract.getCompressionRatio()

        self._set_domain_and_codomain_IDX(np.arange(self.HMAT_size_), np.arange(self.HMAT_size_))


    def _matvec(self, uk):
        """
        E.uk
        (E + DiagTipCorrection).uk
        """
        uk_full = np.zeros(self.HMAT_size_)
        uk_full[self.domain_INDX] = uk
        traction = np.asarray(self.HMATtract.hdotProduct(uk_full))

        # TIP CORRECTION TO BE LOOKED AGAIN
        # if self.enable_tip_corr:
        #     # make tip correction
        #     effective_corrINDX = np.intersect1d(self.tipcorrINDX, self.domain_INDX)
        #     corr_array = np.zeros(self.HMAT_size_)
        #     corr_array[effective_corrINDX] = self.tipcorr[effective_corrINDX]
        #     corr_array = np.multiply(corr_array,uk_full)
        #     traction = traction + corr_array

        return traction[self.codomain_INDX]

    # def _set_tipcorr(self, correction_val, correction_INDX):
    #     self.tipcorr = np.zeros(self.HMAT_size_)
    #     self.tipcorr[correction_INDX] = correction_val
    #     self.tipcorrINDX = correction_INDX
    #     self.enable_tip_corr = True

    def _set_domain_and_codomain_IDX(self, domainIDX, codomainIDX):
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
        self.codomain_INDX = codomainIDX
        self._changeShape(codomainIDX.size)

    def _changeShape(self, shape_):
        self.shape_ = (shape_, shape_)
        super().__init__(self.dtype_, self.shape_)

