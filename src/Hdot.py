
#external
import numpy as np
import timeit
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spilu
from scipy.sparse.linalg import LinearOperator

#from scipy.sparse.linalg import splu #used for testing purposes
import logging

class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
        self.threshold = 10
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            if self.niter == self.threshold:
                print('WARNING: GMRES has not converged in '+str(self.niter)+' iter, monitoring the residual')
            if self.niter > self.threshold:
                print('iter %3i\trk = %s' % (self.niter, str(rk)))

def getMemUse():
    # some memory statistics
    import os, psutil

    process = psutil.Process(os.getpid())
    byte_use = process.memory_info().rss  # byte
    GiByte_use = byte_use / 1024 / 1024 / 1024  # GiB
    print("  -> Current memory use: " + str(GiByte_use) + " GiB")
    return GiByte_use

def getPermutation(HMATobj):
    permut = HMATobj.getPermutation()
    pdof = []
    for i in range(len(permut)):
        pdof.extend([3 * permut[i], 3 * permut[i] + 1, 3 * permut[i] + 2])
    return pdof

def applyPermutation(HMATobj, row_ind, col_ind):
    pdof = np.asarray(getPermutation(HMATobj))
    col_ind = np.asarray(pdof)[col_ind]
    row_ind = np.asarray(pdof)[row_ind]
    del pdof
    return row_ind, col_ind

def deleteNonUsedRows(row_ind, col_ind, values, toBeSavedROWs): #todo: this method looks too fill up the memory (~HMAT)
    ### explicit TEST of the function ###
    # row_indexes = np.asarray([1,4,56,6,7,8,1,1,3,4,90])
    # save_those = np.asarray([1,6,7,2,3,10,12,34,56])
    # save_those_bin = np.in1d(row_indexes, save_those, assume_unique=False)
    # final = row_indexes[save_those_bin]
    #####################################
    toBeSavedRows_bin = np.in1d(row_ind, toBeSavedROWs, assume_unique=False)
    row_ind=row_ind[toBeSavedRows_bin]
    col_ind=col_ind[toBeSavedRows_bin]
    values=values[toBeSavedRows_bin]
    del toBeSavedRows_bin
    return row_ind, col_ind, values

def checkOvelappingEntries(row_ind_tract, row_ind_displ):
    row_ind_tract = np.unique(row_ind_tract)
    row_ind_displ = np.unique(row_ind_displ)
    commonRows = np.intersect1d(row_ind_tract,row_ind_displ,assume_unique=True)
    if len(commonRows) > 0:
        print("ERROR: common Rows have been detected")
        import time
        time.sleep(1.)
        exit()

##############################
# Mdot operator for GMRES    #
##############################
class Mdot(LinearOperator):
  def __init__(self, blockHmat_iLU):
    self.dtype_ = float
    self.shape_ = blockHmat_iLU.shape
    self.blockHmat_iLU = blockHmat_iLU
    self.HMAT_size_ = self.shape_[0]
    super().__init__(self.dtype_, self.shape_)
    self.Hdot = None

  def setHdot(self, Hdot):
      self.Hdot=Hdot

  def _precvec(self,v):
      all_v = np.zeros(self.HMAT_size_)
      all_v[self.rhsOUTindx] = v
      Rhs = self.blockHmat_iLU.solve(all_v)
      return Rhs[self.rhsOUTindx]

  def _matvec(self, v):
    """
    This function implements the dot product.
    :param v: vector expected to be of size unknowns_number_
    :return: HMAT.v, where HMAT is a matrix obtained by selecting equations from either HMATtract or HMATdispl
    """
    all_v = np.zeros(self.HMAT_size_)
    all_v[self.rhsOUTindx] = v
    Rhs = self.blockHmat_iLU.solve(all_v)
    if self.Hdot == None:
        return Rhs[self.rhsOUTindx]
    else:
        return self.Hdot._matvec(Rhs[self.rhsOUTindx])

  def _setRhsOUTindx(self, RhsOUTindx):
    """
    This function sets the index to be in output from the multiplication in _matvet
    :param RhsOUTindx: indexes to output
    :return: -
    """
    self.rhsOUTindx = RhsOUTindx
    self._changeShape(RhsOUTindx.size)

  def _changeShape(self, shape_):
    self.shape_ = (shape_,shape_)
    super().__init__(self.dtype_, self.shape_)

  @property
  def _init_shape(self):
    return self.shape_

  def _init_dtype(self):
    return self.dtype_

##############################
# Hdot operator for GMRES    #
##############################
class Hdot(LinearOperator):
  """
    This is a special Hdot operator.
    When it is instantiated for the first time, it builds two Hmatrices. These are related respectively to a traction kernel and a displacement kernel
    In this contest, a traction kernel is a linear operator that links the displacement discontinuities (DDs) to traction, while
    a displacement kernel links DDs to displacements.
    Then the Hdot provides the multiplication of a vector of DD over a set of equation taken from the two Kernels pending some restrictions.
    Bearing in mind that the equations are selected by row indexes, the restrictions are:
        1-You can not take the same index (or equation) form both the kernel
        2-The final set of equation selected must be equal to the number of rows equal to the one of each kernel

  """
  def __init__(self):
      import pypart
      from pypart import Bigwhamio
      self.unknowns_number_ = None
      self.matvec_size_ = None
      self.HMAT_size_ = None
      self.shape_ = None
      self.dtype_ = float
      self.HMATtract = Bigwhamio()
      self.HMATdispl = Bigwhamio()

  def set(self, data):
    from pypart import pyGetFullBlocks

    # instantiating the objects and variables

    # unpaking the data
    coor, conn, properties, \
    max_leaf_size_tr, eta_tr, \
    max_leaf_size_disp, eta_disp, \
    eps_aca, \
    tractionIDX, displacemIDX, use_preconditioner = data

    # checks
    nodes_per_element_ = 4
    n_of_elts_ = int(len(conn) / nodes_per_element_)
    if len(conn) % nodes_per_element_ != 0 :
        print(" ERROR: \n ")
        print(" wrong connectivity dimension \n ")

    # define the HMAT size
    # define the total number of unknowns to be output by the matvet method
    unknowns_per_element_ = 3
    self.HMAT_size_ = int(n_of_elts_ * unknowns_per_element_)
    self.matvec_size_ = self.HMAT_size_

    # it is mandatory to define shape and dtype of the dot product
    self.shape_ = (self.matvec_size_, self.matvec_size_)
    super().__init__(self.dtype_, self.shape_)

    # set the equation indexes to make the mixed traction-displacement system
    self._setEquationIDX(tractionIDX, displacemIDX)

    # set the objects
    print("  ")
    print(" --------------------------------------- ")
    self.HMATtract.set(coor.tolist(),
                       conn.tolist(),
                       "3DR0",
                       properties,
                       max_leaf_size_tr,
                       eta_tr,
                       eps_aca)
    print(" --------------------------------------- ")
    print("  ")
    print("   -> KERNEL: 3DR0 compr. ratio = "+ str(self.HMATtract.getCompressionRatio()))

    if use_preconditioner:
        # ---> the memory here consist mainly of the Hmat
        myget = pyGetFullBlocks()
        myget.set(self.HMATtract)
        # ---> the memory here consist at most of 3 * Hmat
        col_ind_tract = myget.getColumnN()
        row_ind_tract = myget.getRowN()
        # here we need to permute the rows and columns
        # ---> the memory here consist at most of 3 * Hmat
        [row_ind_tract, col_ind_tract] = applyPermutation(self.HMATtract, row_ind_tract, col_ind_tract )
        # ---> the memory here consist at most of 5 * Hmat for a while and it goes back to 4 Hmat
        values_tract = myget.getValList()
        del myget

        ##### Test #####
        # test the blockHmat VS HMATtraction
        # blockHmat = csc_matrix((values_tract, (row_ind_tract, col_ind_tract)), shape=self.shape_)
        # v= np.ones(self.shape_[0])
        # array1 = blockHmat.dot(v)
        # array2 = self.HMATtract.hdotProduct(v.tolist())
        # relerr = np.linalg.norm(array1-array2)
        # print(relerr)
        # del v, array1, array2, relerr
        ################

        # the following methow quickly fills up the memory leaving it unchanged before and after its application
        [row_ind_tract, col_ind_tract, values_tract] = deleteNonUsedRows(row_ind_tract, col_ind_tract, values_tract,
                                                                         tractionIDX)
    print("  ")
    print(" --------------------------------------- ")
    self.HMATdispl.set(coor,
                       conn,
                       "3DR0_displ", #kernel
                       properties,
                       max_leaf_size_disp,
                       eta_disp,
                       eps_aca)
    print(" --------------------------------------- ")
    print("  ")
    print("   -> KERNEL: 3DR0_displ compr. ratio = "+ str(self.HMATdispl.getCompressionRatio()))

    if use_preconditioner:
        # ---> the memory here consist mainly of 2 Hmat
        myget = pyGetFullBlocks()
        myget.set(self.HMATdispl)
        # ---> the memory here consist at most of 4 Hmat
        col_ind_displ = myget.getColumnN()
        row_ind_displ = myget.getRowN()
        # here we need to permute the rows and columns
        # ---> the memory here consist at most of 4 Hmat
        [row_ind_displ, col_ind_displ] = applyPermutation(self.HMATdispl, row_ind_displ, col_ind_displ )
        # ---> the memory here consist at most of 5 Hmat
        values_displ = myget.getValList()
        del myget

        ##### Test #####
        # test the blockHmat VS HMATtraction
        # blockHmat = csc_matrix((values_displ, (row_ind_displ, col_ind_displ)), shape=self.shape_)
        # v= np.ones(self.shape_[0])
        # array1 = blockHmat.dot(v)
        # array2 = self.HMATdispl.hdotProduct(v.tolist())
        # relerr = np.linalg.norm(array1-array2)
        # print(relerr)
        # del v, array1, array2, relerr
        ################

        # ---> the memory here consist at most of 5 times the Hmat
        # the following methow quickly fills up the memory
        [row_ind_displ, col_ind_displ, values_displ] = deleteNonUsedRows(row_ind_displ, col_ind_displ, values_displ,
                                                                         displacemIDX)
        # ---> the memory here consist at most of 5 times the Hmat
        # the following method uses 1 times the Hmat of memory to apply
        checkOvelappingEntries(row_ind_tract, row_ind_displ)
        # ---> the memory here consist at most of 5 times the Hmat
        # the following concatenations are slow and fills the memory up
        row_ind_tract = np.concatenate((row_ind_tract,row_ind_displ))
        del row_ind_displ
        col_ind_tract = np.concatenate((col_ind_tract,col_ind_displ))
        del col_ind_displ
        values_tract = np.concatenate((values_tract,values_displ))
        del values_displ
        # ---> the memory here consist at most of 3 times the Hmat -> can not be less
        #blockHmat = csr_matrix((values_tract, (row_ind_tract, col_ind_tract)), shape=self.shape_)
        blockHmat = csc_matrix((values_tract, (row_ind_tract, col_ind_tract)), shape=self.shape_)
        del values_tract, row_ind_tract, col_ind_tract

        ### TEST ###
        # we want to check if the blockHmat has been properly computed
        # v = np.ones(self.HMAT_size_)
        # arrayD = np.asarray(self.HMATdispl.hdotProduct(v))
        # arrayT = np.asarray(self.HMATtract.hdotProduct(v))
        # arrayTD = np.asarray(blockHmat.dot(v))
        #
        # arrayD_d = arrayD[displacemIDX]
        # arrayT_t = arrayT[tractionIDX]
        # arrayTD_t = arrayTD[tractionIDX]
        # arrayTD_d = arrayTD[displacemIDX]
        #
        # err = np.linalg.norm(arrayD_d - arrayTD_d) + np.linalg.norm(arrayT_t - arrayTD_t)
        # del v, arrayD, arrayT, arrayTD, arrayD_d, arrayT_t, arrayTD_t, arrayTD_d, err
        ############

        ### Compute an incomplete LU decomposition for a sparse, square matrix. ###
        # The resulting object is an approximation to the inverse of blockHmat.
        # Drop tolerance (0 <= tol <= 1) for an incomplete LU decomposition. (default: 1e-4)
        # Specifies the fill ratio upper bound (>= 1.0) for ILU. (default: 10)
        # To improve the better approximation to the inverse, you may need to increase fill_factor AND decrease drop_tol.
        # A test using a small matrix (~550x550) requires to use drop_tol=1e-15, fill_factor=1000 to achieve a preconditioner equivalent to the inverse
        #
        print("START: creation of the ILU approx ")
        print("   -> Size of the problem: "+str(self.shape_))
        memuse = (self.shape_[0]**2)*8/(1024**3)
        print("   -> Memory [GiB]: " + str(round(memuse,3)))
        tic = timeit.default_timer()
        blockHmat_iLU = spilu(blockHmat, drop_tol=1e-13, fill_factor=5)
        toc = timeit.default_timer()
        tictoc=(toc - tic)/60.
        print("END: creation of the ILU approx, time: "+ str(round(tictoc,2))+" minutes")

        # defining the accuracy of the matrix inverse
        v = np.ones(self.HMAT_size_)
        self._setRhsOUTindx(np.arange(self.HMAT_size_))
        test1 = self._matvec(v)
        test1 = blockHmat_iLU.solve(test1)
        approx_err = np.linalg.norm(test1-v)/self.HMAT_size_
        print("   -> The max difference between the approx. val. ")
        print("   -> and the matrix inverse is: " + str(round(approx_err,3)))

        ### TEST ###
        # this test helps checking that all the slices are correct and that the preconditioner works correctly
        # LU = splu(blockHmat) # <---- this will compute the true HMAT
        #
        # v = np.ones(self.HMAT_size_)
        # self._setRhsOUTindx(np.arange(self.HMAT_size_))
        # test1 = self._matvec(v)
        #
        # test1 = LU.solve(test1)
        # test2 = blockHmat_iLU.solve(test1)
        ###

        del blockHmat
        return blockHmat_iLU
    else :
        return None

  def _matvec(self, v):
    """
    This function implements the dot product.
    :param v: vector expected to be of size unknowns_number_
    :return: HMAT.v, where HMAT is a matrix obtained by selecting equations from either HMATtract or HMATdispl
    """
    all_v = np.zeros(self.HMAT_size_)
    all_v[self.rhsOUTindx] = v
    Rhs_t = self.HMATtract.hdotProduct(all_v)
    Rhs_d = self.HMATdispl.hdotProduct(all_v)
    Rhs = self._selectEquations(Rhs_t,Rhs_d)
    return Rhs[self.rhsOUTindx]

  def _matvec_full(self, v):
    """
    This function implements the dot product.
    :param v: vector expected to be of size self.HMAT_size_
    :return: HMAT.v, where HMAT is a matrix obtained by selecting equations from either HMATtract or HMATdispl
    """
    Rhs_t = self.HMATtract.hdotProduct(v)
    Rhs_d = self.HMATdispl.hdotProduct(v)
    Rhs = self._selectEquations(Rhs_t,Rhs_d)
    return Rhs[self.rhsOUTindx]

  def _selectEquations(self, Rhs_t, Rhs_d):
    """
    This functions select the results from two arrays according to the choice expressed by Rhs_t, Rhs_d
    :param Rhs_t: vector of indexes of the equations from the traction HMAT to be considered
    :param Rhs_d: vector of indexes of the equations from the displacement HMAT to be considered
    :return: Rhs is a vector with the required values and size unknowns_number_
    """
    Rhs = np.zeros(self.HMAT_size_)
    Rhs[self.tractionIDX] = np.asarray(Rhs_t)[self.tractionIDX]
    Rhs[self.displacemIDX] = np.asarray(Rhs_d)[self.displacemIDX]
    return Rhs

  def _setRhsOUTindx(self, RhsOUTindx):
    """
    This function sets the index to be in output from the multiplication in _matvet
    :param RhsOUTindx: indexes to output
    :return: -
    """
    self.rhsOUTindx = RhsOUTindx
    self._changeShape(RhsOUTindx.size)

  def _setEquationIDX(self, tractionIDX, displacemIDX):
      # this function sets the indexes in order to assembly the final system according to the boundary conditions
      # all the crack plane is assumed to have a traction BC
      self.displacemIDX = displacemIDX
      self.tractionIDX = tractionIDX

  def _changeShape(self, shape_):
      self.shape_ = (shape_,shape_)
      super().__init__(self.dtype_, self.shape_)

  @property
  def _init_shape(self):
    return self.shape_

  def _init_dtype(self):
    return self.dtype_

#--------------------------------